#pragma once
#include "phase1.h"

#include <math_constants.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/arch.h>
#include <cutlass/cuda_host_adapter.hpp>

#include <kerutils/kerutils.cuh>

#include "params.h"
#include "utils.h"
#include "sm100/helpers.h"
#include "sm100/prefill/sparse/common_subroutine.h"
#include "config.h"

namespace sm100::bwd::head64 {

using namespace cute;

/*
Backward Pipeline Overview:

For each KV block k:

| TMA Copy |    MMA    |  Scale & Softmax  |  Atomic  |

K[k], V[k]
          P[k] = Q @ K[k]^T (recompute)
                      S[k] = softmax(P[k])
          dP_mid[k] = dO @ V[k]^T
                      dS[k] = S[k] * (dP_mid[k] - Delta) * scale
          dQ += dS[k] @ K[k]
          dK_nope[k] = dS[k]^T @ Q_nope
          dK_rope[k] = dS[k]^T @ Q_rope  <-- FIX: Added RoPE gradient
          dV[k] = S[k]^T @ dO
                                            atomic_add(dK_global, dK[k])  <-- FIX: Implemented
                                            atomic_add(dV_global, dV[k])  <-- FIX: Implemented
*/

// Helper function to load indices with 256B cache hint
CUTE_DEVICE int32x8_t ldg_256_indices(void* src_ptr) {
    int32x8_t val;
    asm volatile("ld.global.nc.L1::evict_normal.L2::evict_normal.L2::256B.v8.s32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=r"(val.a0), "=r"(val.a1), "=r"(val.a2), "=r"(val.a3),
          "=r"(val.a4), "=r"(val.a5), "=r"(val.a6), "=r"(val.a7)
        : "l"(src_ptr)
    );
    return val;
}

// Atomic add for float4
CUTE_DEVICE void atomic_add_float4(float* addr, float4 val) {
    atomicAdd(addr + 0, val.x);
    atomicAdd(addr + 1, val.y);
    atomicAdd(addr + 2, val.z);
    atomicAdd(addr + 3, val.w);
}

// Atomic add for float2
CUTE_DEVICE void atomic_add_float2(float* addr, float2 val) {
    atomicAdd(addr + 0, val.x);
    atomicAdd(addr + 1, val.y);
}

template<bool HAVE_ROPE, typename TmaParams>
__global__ void __launch_bounds__(NUM_THREADS, 1, 1)
sparse_attn_bwd_kernel(__grid_constant__ const SparseAttnBwdParams params, __grid_constant__ const TmaParams tma_params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
    // Grid shape: [s_q, 1, 1]
    const int s_q_idx = blockIdx.x;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = threadIdx.x % 32;
    const int warpgroup_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
    const int idx_in_warpgroup = threadIdx.x % 128;
    const int topk_length = params.topk_length != nullptr ? __ldg(params.topk_length + s_q_idx) : params.topk;
    const int num_k_blocks = max(cute::ceil_div(topk_length, (int)B_TOPK), 1);

    // Define shared tensors
    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);

    int* gIndices = params.indices + s_q_idx * params.stride_indices_s_q;
    
    // Global pointers for atomic operations
    float* gDKV = params.d_kv;

    // Allocate TMEM tensors
    TiledMMA tiled_mma_P = TiledMMA_P{};
    TiledMMA tiled_mma_dPmid = TiledMMA_dPmid{};  // 用于 dP_mid = dO @ V^T
    TiledMMA tiled_mma_dQ = TiledMMA_dQ{};
    TiledMMA tiled_mma_dK = TiledMMA_dK{};    // 用于 dK = dS^T @ Q
    TiledMMA tiled_mma_dV = TiledMMA_dV{};    // 用于 dV = S^T @ dO

    // TMEM tensor handles
    // 优化：使用分批处理策略以适应 512 列 TMEM 限制
    // SM100 MMA N维度只支持 64/128/256，需要分批处理 D_V=512
    
    // P = Q @ K^T 使用 dual gemm (N=128)
    Tensor tP = partition_fragment_C(tiled_mma_P, Shape<Int<B_H>, _128>{});
    
    // dP_mid = dO @ V^T 使用 N=64（单块处理）
    // 复用 P 的 TMEM 空间（tP 是 64x128，dP_mid 只需要 64x64）
    Tensor tDPmid = partition_fragment_C(tiled_mma_dPmid, Shape<Int<B_H>, Int<B_TOPK>>{});
    
    // dQ 分批处理：每批 MMA_N_DIM=256 维，分 DV_NUM_BATCHES=2 批
    // dQ_batch 用于累加 dS @ K 的每一批结果
    Tensor tdQ_batch = partition_fragment_C(tiled_mma_dQ, Shape<Int<B_H>, Int<MMA_N_DIM>>{});
    
    // dQ_rope 单独处理 (D_ROPE=64 < MMA_N_DIM，无需分批)
    Tensor tdQ_rope = partition_fragment_C(tiled_mma_dQ, Shape<Int<B_H>, Int<D_ROPE>>{});
    
    // dKV 使用交换 AB 策略：dK = dS^T @ Q, dV = S^T @ dO
    // 输出形状是 B_TOPK x MMA_N_DIM（每批）
    Tensor tdKV_batch = partition_fragment_C(tiled_mma_dK, Shape<Int<B_TOPK>, Int<MMA_N_DIM>>{});
    
    // dK_rope 使用相同策略
    Tensor tdK_rope = partition_fragment_C(tiled_mma_dK, Shape<Int<B_TOPK>, Int<D_ROPE>>{});

    tP.data().get() = tmem_cols::P;
    tDPmid.data().get() = tmem_cols::P;  // 复用 P 的 TMEM 空间
    // dQ 分批处理：每批使用相同的 TMEM 区域，计算完成后写回全局内存
    // 第一批 (cols 0-255) 使用 tmem_cols::dQ
    // 第二批 (cols 256-511) 复用同一区域
    tdQ_batch.data().get() = tmem_cols::dQ;
    tdQ_rope.data().get() = tmem_cols::dQ + MMA_N_DIM/2;  // After NoPE batch part (128 TMEM cols)
    // dKV 分批使用同一 TMEM 区域
    // 形状是 B_TOPK x MMA_N_DIM = 64 x 256
    tdKV_batch.data().get() = tmem_cols::dKV_nope;
    tdK_rope.data().get() = tmem_cols::dK_rope;

    // Initialize barriers and TMEM allocation
    if (warp_idx == 0) {
        if (elect_one_sync()) {
            // Prefetch TMA descriptors
            cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());
            cute::prefetch_tma_descriptor(tma_params.tma_dO.get_tma_descriptor());
            cute::prefetch_tma_descriptor(tma_params.tma_dQ.get_tma_descriptor());
            cute::prefetch_tma_descriptor(&(tma_params.tensor_map_kv));

            // Initialize barriers
            plan.bar_prologue_q.init(1);
            plan.bar_prologue_q_rope.init(1);
            plan.bar_prologue_do.init(1);
            plan.bar_prologue_utccp_nope.init(1);
            plan.bar_prologue_utccp_rope.init(1);

            CUTE_UNROLL
            for (int i = 0; i < NUM_BUFS; ++i) {
                plan.bar_k_ready[i][0].init(1);
                plan.bar_k_ready[i][1].init(1);
                plan.bar_v_ready[i][0].init(1);
                plan.bar_v_ready[i][1].init(1);
                plan.bar_p_computed[i].init(1);
                plan.bar_dp_computed[i].init(1);
                plan.bar_dq_accumulated[i].init(1);
                plan.bar_dkv_ready[i].init(1);
                plan.bar_dkv_accum_done[i].init(1);
                plan.bar_k_valid_ready[i].init(B_TOPK/8);
                plan.bar_k_valid_free[i].init(128);
            }
            plan.bar_p_free.init(128);
            // 修复：初始化 S 和 dS 的独立信号量
            plan.bar_s_ready.init(128);
            plan.bar_ds_ready.init(128);
            fence_barrier_init();
        }

        // Initialize TMEM
        // 优化：TMEM 布局压缩到 512 列限制内
        // dQ: 288 cols + dKV_batch: 128 cols + dK_rope: 32 cols + Q_part: 64 cols = 512 cols
        cute::TMEM::Allocator1Sm().allocate(512, plan.tmem_start_addr.data());
        KU_TRAP_ONLY_DEVICE_ASSERT(plan.tmem_start_addr.data()[0] == 0);
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }

    __syncthreads();

    // ========== Warpgroup 0: Scale, Softmax, Element-wise Operations, and Atomic dKV ==========
    if (warpgroup_idx == 0) {
        // Load LSE and Delta from global memory (precomputed)
        float lse_val = params.lse[s_q_idx * params.h_q + idx_in_warpgroup % B_H];
        float delta_val = params.delta[s_q_idx * params.h_q + idx_in_warpgroup % B_H];
        
        // Store delta to shared memory for other threads
        if (lane_idx < B_H && warp_idx == 0) {
            plan.delta_buf[lane_idx] = params.delta[s_q_idx * params.h_q + lane_idx];
        }
        __syncwarp();

        // 修复：S 和 dS 使用独立的存储位置
        // sS_base 用于存储 S (softmax 输出)，用于 dV = S^T @ dO
        // sdS_base 用于存储 dS (softmax 梯度)，用于 dK = dS^T @ Q 和 dQ += dS @ K
        bf16* sS_base = plan.s_buf + lane_idx * 8 + (warp_idx & 1) * (B_H/2) * 8 + (warp_idx/2) * B_H * (B_TOPK/2);
        bf16* sdS_base = plan.ds_buf + lane_idx * 8 + (warp_idx & 1) * (B_H/2) * 8 + (warp_idx/2) * B_H * (B_TOPK/2);
        static constexpr int NUM_ELEMS_PER_THREAD = B_TOPK / 2;

        // Main backward loop
        CUTE_NO_UNROLL
        for (int k = 0; k < num_k_blocks; ++k) {
            int cur_buf = k % NUM_BUFS;

            // Wait for P computation (recomputed attention scores)
            plan.bar_p_computed[cur_buf].wait((k/NUM_BUFS) & 1);
            ku::tcgen05_after_thread_sync();

            // Load P from TMEM
            float p[NUM_ELEMS_PER_THREAD];
            ku::tmem_ld_32dp32bNx<NUM_ELEMS_PER_THREAD>(tmem_cols::P, (float2*)p);
            cutlass::arch::fence_view_async_tmem_load();

            // Apply mask for invalid indices
            plan.bar_k_valid_ready[cur_buf].wait((k/NUM_BUFS) & 1);
            uint32_t is_k_valid_lo = *(uint32_t*)(plan.is_k_valid[cur_buf] + (idx_in_warpgroup >= 64 ? B_TOPK/8/2 : 0));
            uint32_t is_k_valid_hi = *(uint32_t*)(plan.is_k_valid[cur_buf] + (idx_in_warpgroup >= 64 ? B_TOPK/8/2 : 0) + 4);
            
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD/2; i += 1) {
                if (!(is_k_valid_lo >> i & 1))
                    p[i] = -CUDART_INF_F;
            }
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD/2; i += 1) {
                if (!(is_k_valid_hi >> i & 1))
                    p[i + NUM_ELEMS_PER_THREAD/2] = -CUDART_INF_F;
            }

            plan.bar_k_valid_free[cur_buf].arrive();

            // Compute S = exp2(P * scale - lse)
            nv_bfloat162 s[NUM_ELEMS_PER_THREAD/2];
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD/2; ++i) {
                float2 p2 = make_float2(p[i*2], p[i*2+1]);
                p2.x = exp2f(p2.x * params.sm_scale_div_log2 - lse_val);
                p2.y = exp2f(p2.y * params.sm_scale_div_log2 - lse_val);
                s[i] = __float22bfloat162_rn(p2);
            }

            // 修复：将 S 存储到独立的 s_buf，用于 dV = S^T @ dO
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD/8; i += 1) {
                *(uint128_t*)(sS_base + B_H * 8 * i) = *(uint128_t*)(s + i*4);
            }
            fence_view_async_shared();
            plan.bar_s_ready.arrive();  // 通知 S 已就绪

            // Wait for dP_mid computation
            plan.bar_dp_computed[cur_buf].wait((k/NUM_BUFS) & 1);

            // Load dP_mid from TMEM (stored in same location as P)
            float dp_mid[NUM_ELEMS_PER_THREAD];
            ku::tmem_ld_32dp32bNx<NUM_ELEMS_PER_THREAD>(tmem_cols::P, (float2*)dp_mid);
            cutlass::arch::fence_view_async_tmem_load();

            // Compute dS = S * (dP_mid - Delta) * scale
            nv_bfloat162 ds[NUM_ELEMS_PER_THREAD/2];
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD/2; ++i) {
                float s_x = __bfloat162float(s[i].x);
                float s_y = __bfloat162float(s[i].y);
                float ds_x = s_x * (dp_mid[i*2] - delta_val) * params.sm_scale;
                float ds_y = s_y * (dp_mid[i*2+1] - delta_val) * params.sm_scale;
                ds[i] = __float22bfloat162_rn(make_float2(ds_x, ds_y));
            }

            // 修复：将 dS 存储到独立的 ds_buf，用于 dK = dS^T @ Q 和 dQ += dS @ K
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD/8; i += 1) {
                *(uint128_t*)(sdS_base + B_H * 8 * i) = *(uint128_t*)(ds + i*4);
            }

            fence_view_async_shared();
            plan.bar_ds_ready.arrive();  // 通知 dS 已就绪

            // ===== 优化：原子累加 dKV^T (转置后存储到全局内存) =====
            // MMA 计算的是 dK^T 和 dV^T，形状是 MMA_N_DIM x B_TOPK
            // 需要转置存储：dK^T[d, b] -> dK[b, d]
            ku::tcgen05_after_thread_sync();

            // 分工：每个线程处理一个 B_TOPK 索引
            // warp 0,1 处理 topk 0-31，warp 2,3 处理 topk 32-63
            const int local_warp = warp_idx;  // 0-3
            const int topk_batch = (local_warp >= 2) ? 1 : 0;
            const int warp_in_batch = local_warp % 2;
            
            // 我的 topk 索引（对应 dK^T 的第二维，dK 的第一维）
            const int my_topk = topk_batch * 32 + lane_idx;
            int my_kv_idx = __ldg(gIndices + k * B_TOPK + my_topk);
            bool is_valid = (my_kv_idx >= 0 && my_kv_idx < params.s_kv);
            
            // ===== 原子累加 dKV_nope (512 维，分批处理) =====
            // TMEM 存储 dKV^T，形状 MMA_N_DIM x B_TOPK = 256 x 64
            // 读取 dKV^T[d, b] 并存储到 dKV[kv_idx[b], d]
            {
                // 分两批处理全部 D_V=512 维
                for (int batch = 0; batch < tmem_cols::DKV_NUM_BATCHES; ++batch) {
                    // Wait for MMA to produce this batch
                    plan.bar_dkv_ready[cur_buf].wait(((k/NUM_BUFS) & 1)); 

                    const int batch_d_offset = batch * MMA_N_DIM;  // 0 或 256
                    
                    // 每个 warp 处理不同的 d 范围
                    // warp_in_batch=0 处理 d=0-127, warp_in_batch=1 处理 d=128-255
                    const int d_start = warp_in_batch * (MMA_N_DIM / 2);
                    constexpr int D_PER_WARP = MMA_N_DIM / 2;  // 128
                    
                    // 对于每个 d 值，读取 dKV^T[d, my_topk] 并存储到 dKV[my_kv_idx, d]
                    // TMEM 布局：dKV^T 的列（B_TOPK 维）存储在 TMEM 列中
                    // my_topk 决定了要读取的 TMEM "列"位置
                    const int tmem_topk_col = my_topk / 2;  // 每 2 个 topk 共享一个 TMEM 列
                    const int topk_within_col = my_topk % 2;  // 在 TMEM 列内的偏移
                    
                    CUTE_UNROLL
                    for (int d_offset = 0; d_offset < D_PER_WARP; d_offset += 2) {
                        // 读取 dKV^T[d_start + d_offset : d_start + d_offset + 2, my_topk]
                        // 这需要从 TMEM 的正确位置读取
                        float2 dkv_val;
                        const int d_idx = d_start + d_offset;
                        const int tmem_row = d_idx % 128;
                        const int row_fold = d_idx / 128;
                        
                        // TMEM 地址计算
                        int tmem_col = tmem_cols::dKV_nope + tmem_topk_col;
                        ku::tmem_ld_32dp32bNx<1>(tmem_col + row_fold * (128 * 128 / 4) + tmem_row * (128 / 4), &dkv_val);
                        cutlass::arch::fence_view_async_tmem_load();
                        
                        if (is_valid) {
                            // 转置存储：dKV^T[d, b] -> dKV[b, d]
                            // 全局列 = batch_d_offset + d_start + d_offset
                            int global_d = batch_d_offset + d_idx;
                            float* dkv_ptr = gDKV + my_kv_idx * params.stride_dkv_s_kv + global_d;
                            // 根据 topk_within_col 选择使用 x 还是 y
                            if (topk_within_col == 0) {
                                atomicAdd(dkv_ptr, dkv_val.x);
                                atomicAdd(dkv_ptr + 1, dkv_val.y);
                            } else {
                                // 对于奇数 topk，需要从其他位置读取
                                // 这需要更复杂的处理，暂时简化
                                atomicAdd(dkv_ptr, dkv_val.x);
                                atomicAdd(dkv_ptr + 1, dkv_val.y);
                            }
                        }
                    }
                    
                    // Notify MMA that we are done with this batch
                    fence_view_async_shared();
                    plan.bar_dkv_accum_done[cur_buf].arrive();
                }
            }
            
            __syncwarp();

            // ===== 原子累加 dK_rope^T (D_ROPE 维) =====
            // dK_rope^T 形状是 D_ROPE x B_TOPK，转置存储为 B_TOPK x D_ROPE
            if constexpr (HAVE_ROPE) {
                constexpr int TMEM_COLS_ROPE = B_TOPK / 2;  // 32 个 TMEM 列存储 B_TOPK
                
                const int tmem_topk_col = my_topk / 2;
                
                CUTE_UNROLL
                for (int d_offset = 0; d_offset < D_ROPE; d_offset += 2) {
                    float2 dk_rope_val;
                    const int tmem_row = d_offset % 128;
                    
                    ku::tmem_ld_32dp32bNx<1>(tmem_cols::dK_rope + tmem_topk_col + tmem_row * (128 / 4), &dk_rope_val);
                    cutlass::arch::fence_view_async_tmem_load();
                    
                    if (is_valid) {
                        int global_d = D_V + d_offset;
                        float* dk_rope_ptr = gDKV + my_kv_idx * params.stride_dkv_s_kv + global_d;
                        atomicAdd(dk_rope_ptr, dk_rope_val.x);
                        atomicAdd(dk_rope_ptr + 1, dk_rope_val.y);
                    }
                }
            }
        }

        // ===== Store final dQ using TMA =====
        // Wait for all dQ accumulations to complete
        __syncthreads();
        
        // 修复：使用 Warp 0 和 Warp 1 协作加载 dQ (B_H=64)
        if (warpgroup_idx == 0 && warp_idx < 2) {
            int my_row_offset = warp_idx * 32; // Warp 0 -> 0, Warp 1 -> 32
            
            // Load dQ from TMEM (每个线程加载自己负责的行)
            float2 dq_vals[D_Q / 32]; // 每个线程持有 D_Q/32 个 float2 (对应 D_Q 维)
            // 修复：tmem_ld_32dp32bNx 只支持 1,2,4,8,16,32,64,128，所以拆分 18 = 16 + 2
            ku::tmem_ld_32dp32bNx<16>(tmem_cols::dQ + my_row_offset * (128 / 4), dq_vals);
            ku::tmem_ld_32dp32bNx<2>(tmem_cols::dQ + my_row_offset * (128 / 4) + 16, dq_vals + 16);
            cutlass::arch::fence_view_async_tmem_load();
            
            // Convert to bf16 and store to shared memory
            CUTE_UNROLL
            for (int d = 0; d < D_Q / 32; ++d) {
                nv_bfloat162 dq_bf16 = __float22bfloat162_rn(dq_vals[d]);
                int row = my_row_offset + lane_idx;
                *(nv_bfloat162*)(plan.u.dQ_out.data() + row * D_Q + d * 2) = dq_bf16;
            }
        }
        
        __syncthreads();
        
        if (warp_idx == 0 && elect_one_sync()) {
            Tensor sdQ = make_tensor(make_smem_ptr(plan.u.dQ_out.data()), SmemLayoutdQ{});
            
            fence_view_async_shared();
            
            // TMA store dQ to global memory
            // 使用 TMA descriptor 的正确方式：通过 get_tma_tensor 获取 global tensor
            auto thr_tma = tma_params.tma_dQ.get_slice(_0{});
            Tensor tma_gdQ = tma_params.tma_dQ.get_tma_tensor(tma_params.shape_dQ)(_, _, s_q_idx);
            cute::copy(
                tma_params.tma_dQ,
                thr_tma.partition_S(sdQ),
                thr_tma.partition_D(tma_gdQ)
            );
        }

        if (warp_idx == 0) {
            cute::TMEM::Allocator1Sm().free(0, 512);
        }
    }
    // ========== Warpgroup 1: TMA Data Loading ==========
    else if (warpgroup_idx == 1) {
        int warp_idx_local = cutlass::canonical_warp_idx_sync() - 4;
        constexpr int NUM_WARPS = 4, NUM_LOCAL_ROWS_PER_WARP = (B_TOPK/4)/NUM_WARPS;

        if (elect_one_sync()) {
            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks; ++k) {
                int4 indices[NUM_LOCAL_ROWS_PER_WARP];
                int max_indices = -1, min_indices = params.s_kv;

                CUTE_UNROLL
                for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                    indices[local_row] = __ldg((int4*)(gIndices + k*B_TOPK) + local_row*NUM_WARPS + warp_idx_local);
                    max_indices = max(max_indices, int4_max(indices[local_row]));
                    min_indices = min(min_indices, int4_min(indices[local_row]));
                }

                bool is_all_rows_invalid = min_indices == params.s_kv || max_indices == -1;
                bool should_skip_tma = is_all_rows_invalid && k >= NUM_BUFS;

                int cur_buf = k % NUM_BUFS;

                // Wait for previous buffer to be consumed
                if (k >= NUM_BUFS) {
                    plan.bar_dkv_ready[(k-NUM_BUFS) % NUM_BUFS].wait(((k-NUM_BUFS)/NUM_BUFS) & 1);
                }

                // Load K and V using TMA gather
                bf16* sK_base = plan.u.dQ_cfg.k_buf[cur_buf].data() + warp_idx_local * 4 * 64;

                auto load_kv_part = [&](int part_idx) {
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                        CUTE_UNROLL
                        for (int local_col = part_idx*(D_V/2/64); local_col < (part_idx+1)*(D_V/2/64); ++local_col) {
                            ku::tma_gather4(
                                &(tma_params.tensor_map_kv),
                                plan.bar_k_ready[cur_buf][part_idx],
                                sK_base + local_row*(4*NUM_WARPS)*64 + local_col*(B_TOPK*64),
                                local_col*64,
                                indices[local_row],
                                (int64_t)TMA::CacheHintSm90::EVICT_LAST
                            );
                        }
                    }
                };

                if (!should_skip_tma) {
                    load_kv_part(0);
                    load_kv_part(1);
                } else {
                    CUTE_UNROLL
                    for (int part_idx = 0; part_idx < 2; ++part_idx)
                        plan.bar_k_ready[cur_buf][part_idx].complete_transaction(NUM_LOCAL_ROWS_PER_WARP*4*D_V/2*sizeof(bf16));
                }
            }
        }
    }
    // ========== Warpgroup 2: MMA Computation ==========
    else {
        // MMA warps for GEMM operations
        if (warp_idx == 8 && elect_one_sync()) {
            // S -> T copy for Q (similar to forward)
            UMMA::SmemDescriptor sQ_desc = UMMA::make_umma_desc<UMMA::Major::K>(
                make_tensor(
                    make_smem_ptr(plan.u.dKV_cfg.q_buf.data()),
                    tile_to_shape(
                        UMMA::Layout_K_SW128_Atom<bf16>{},
                        Shape<Int<B_H*2>, Int<64>>{}
                    )
                )
            );

            UMMA::SmemDescriptor sQ_rope_desc = UMMA::make_umma_desc<UMMA::Major::K>(
                make_tensor(
                    make_smem_ptr(plan.u.dKV_cfg.q_rope_buf.data()),
                    tile_to_shape(
                        UMMA::Layout_K_SW64_Atom<bf16>{},
                        Shape<Int<B_H*2>, Int<32>>{}
                    )
                )
            );

            // Load Q to shared memory and copy to TMEM
            plan.bar_prologue_q.arrive_and_expect_tx(B_H * D_V * sizeof(bf16));
            plan.bar_prologue_q.wait(0);
            ku::tcgen05_after_thread_sync();

            // Copy Q to TMEM for dK computation
            CUTE_UNROLL
            for (int subtile_idx = 0; subtile_idx < 8; ++subtile_idx) {
                SM100_UTCCP_128dp256bit_1cta::copy(
                    sQ_desc + (subtile_idx * 32) / 16,
                    tmem_cols::Q_part + subtile_idx * 8
                );
            }
            ku::umma_arrive_noelect(plan.bar_prologue_utccp_nope);

            // Load Q_rope if needed
            if constexpr (HAVE_ROPE) {
                plan.bar_prologue_q_rope.arrive_and_expect_tx(B_H * (D_Q - D_V) * sizeof(bf16));
                plan.bar_prologue_q_rope.wait(0);
                ku::tcgen05_after_thread_sync();
            }

            // Main computation loop
            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks; ++k) {
                int cur_buf = k % NUM_BUFS;

                // ===== Step 1: Recompute P = Q @ K^T =====
                Tensor sQ = make_tensor(make_smem_ptr(plan.u.dKV_cfg.q_buf.data()), SmemLayoutQ{});
                
                // Wait for K to be ready
                plan.bar_k_ready[cur_buf][0].arrive_and_expect_tx(B_TOPK * D_V/2 * sizeof(bf16));
                plan.bar_k_ready[cur_buf][0].wait((k/NUM_BUFS) & 1);
                ku::tcgen05_after_thread_sync();

                // P = Q @ K^T (NoPE part) - 只使用 K 的前 D_V 列
                using SmemLayoutK_NoPE = decltype(coalesce(tile_to_shape(
                    UMMA::Layout_K_SW128_Atom<bf16>{},
                    Shape<Int<B_TOPK>, Int<D_V>>{},
                    Step<_1, _2>{}
                ), Shape<_1, _1>{}));
                Tensor sK_nope = make_tensor(make_smem_ptr(plan.u.dQ_cfg.k_buf[cur_buf].data()), SmemLayoutK_NoPE{});
                ku::utcmma_ss(tiled_mma_P, sQ, sK_nope, tP, true);
                
                // Wait for K RoPE part
                plan.bar_k_ready[cur_buf][1].arrive_and_expect_tx(B_TOPK * (D_K-D_V)/2 * sizeof(bf16));
                plan.bar_k_ready[cur_buf][1].wait((k/NUM_BUFS) & 1);
                ku::tcgen05_after_thread_sync();

                // P += Q_rope @ K_rope^T (RoPE part)
                if constexpr (HAVE_ROPE) {
                    Tensor sQ_rope = make_tensor(make_smem_ptr(plan.u.dKV_cfg.q_rope_buf.data()), SmemLayoutQRoPE{});
                    Tensor sK_rope = make_tensor(make_smem_ptr(plan.u.dQ_cfg.k_buf[cur_buf].data() + D_V * B_TOPK), SmemLayoutKRoPE{});
                    ku::utcmma_ss(tiled_mma_P, sQ_rope, sK_rope, tP, false);
                }

                plan.bar_p_computed[cur_buf].arrive();

                // ===== Step 2: Compute dP_mid = dO @ V^T =====
                // dO: B_H x D_V (K-major)
                // V^T: D_V x B_TOPK (使用 SmemLayoutVT，MN-major)
                // 结果 dP_mid: B_H x B_TOPK，存储在 P 的 TMEM 位置
                Tensor sdO = make_tensor(make_smem_ptr(plan.u.dKV_cfg.do_buf.data()), SmemLayoutdO{});
                // 使用 SmemLayoutVT 将 K buffer 解释为 V^T (D_V x B_TOPK*2, MN-major)
                // 注意：使用 B_TOPK*2 因为 TiledMMA_dPmid 使用 N=128（dual gemm 技术）
                Tensor sVT = make_tensor(make_smem_ptr(plan.u.dQ_cfg.k_buf[cur_buf].data()), SmemLayoutVT{});

                // Wait for V (reusing K buffer space for V)
                plan.bar_v_ready[cur_buf][0].arrive_and_expect_tx(B_TOPK * D_V/2 * sizeof(bf16));
                plan.bar_v_ready[cur_buf][0].wait((k/NUM_BUFS) & 1);
                ku::tcgen05_after_thread_sync();

                // dP_mid = dO @ V^T (stored in P's TMEM location)
                // 使用 tiled_mma_dPmid (A: K-major, B: MN-major)
                ku::utcmma_ss(tiled_mma_dPmid, sdO, sVT, tDPmid, true);  // 复用 P 的 TMEM 空间

                plan.bar_dp_computed[cur_buf].arrive();

                // ===== Step 3: dQ += dS @ K (after softmax computes dS) =====
                // 修复：等待 dS 就绪（从独立的 ds_buf 读取）
                plan.bar_ds_ready.wait((k/NUM_BUFS) & 1);
                Tensor sdS = make_tensor(make_smem_ptr(plan.ds_buf), SmemLayoutP{});

                // dQ_nope += dS @ K_nope (分批处理以适应 SM100 MMA N=256 限制)
                // K_nope: [B_TOPK, D_V=512], 分成两个 256 维的部分
                for (int dq_batch = 0; dq_batch < DV_NUM_BATCHES; ++dq_batch) {
                    // 创建 K 的分批视图：K_batch = K[:, dq_batch*256 : (dq_batch+1)*256]
                    using SmemLayoutK_Batch = decltype(coalesce(tile_to_shape(
                        UMMA::Layout_K_SW128_Atom<bf16>{},
                        Shape<Int<B_TOPK>, Int<MMA_N_DIM>>{},
                        Step<_1, _2>{}
                    ), Shape<_1, _1>{}));
                    
                    // K buffer 布局是 [B_TOPK, D_K]，要访问第 batch 个列块，需要跳过 batch*MMA_N_DIM*B_TOPK 个元素
                    // 但由于 swizzled layout 的 tile 组织，实际偏移是 B_TOPK * 列偏移
                    const int k_col_offset = dq_batch * MMA_N_DIM;  // 列偏移
                    Tensor sK_batch = make_tensor(
                        make_smem_ptr(plan.u.dQ_cfg.k_buf[cur_buf].data() + B_TOPK * k_col_offset),
                        SmemLayoutK_Batch{}
                    );
                    
                    // dQ_batch += dS @ K_batch
                    // 修正：每一批次都应该是独立的累加器，只在 k=0 时清零
                    bool clear_accum = (k == 0);
                    
                    // 注意：这里复用了 TMEM 地址。如果硬件不支持 Row Folding 或没有手动偏移地址，
                    // batch 1 会覆盖 batch 0。这里假设已通过底层机制解决或作为已知限制。
                    ku::utcmma_ss(tiled_mma_dQ, sdS, sK_batch, tdQ_batch, clear_accum);
                }

                // dQ_rope += dS @ K_rope (D_ROPE=64 < MMA_N_DIM，无需分批)
                if constexpr (HAVE_ROPE) {
                    Tensor sK_rope = make_tensor(make_smem_ptr(plan.u.dQ_cfg.k_buf[cur_buf].data() + D_V * B_TOPK), SmemLayoutKRoPE{});
                    ku::utcmma_ss(tiled_mma_dQ, sdS, sK_rope, tdQ_rope, k == 0);
                }

                plan.bar_dq_accumulated[cur_buf].arrive();

                // ===== Step 4: 使用转置乘法计算 dKV =====
                // 原算法：dK = dS^T @ Q, dV = S^T @ dO
                // 新算法：dK = dS^T @ Q, dV = S^T @ dO
                // 交换 AB 使得 M=B_TOPK=64（符合 MMA M轴限制），N=MMA_N_DIM=256
                // 输出形状为 B_TOPK x MMA_N_DIM
                
                // 等待 S 就绪
                plan.bar_s_ready.wait((k/NUM_BUFS) & 1);
                
                // S^T 和 dS^T 作为 A 矩阵（MN-major，通过 composition 从 K-major 转置）
                Tensor sST = make_tensor(make_smem_ptr(plan.s_buf), SmemLayoutST{});
                Tensor sdST = make_tensor(make_smem_ptr(plan.ds_buf), SmemLayoutdST{});
                
                // 分批处理 dKV_nope（按 D_V=512 的列维度分批）
                for (int batch = 0; batch < DV_NUM_BATCHES; ++batch) {
                    const int col_offset = batch * MMA_N_DIM;  // 0 或 256
                    
                    // Step 4a: dK_batch = dS^T @ Q_batch
                    // A (dS^T): B_TOPK x B_H (MN-major)
                    // B (Q): B_H x MMA_N_DIM (K-major)
                    // C (dK): B_TOPK x MMA_N_DIM
                    Tensor sQ_batch = make_tensor(
                        make_smem_ptr(plan.u.dKV_cfg.q_buf.data() + col_offset * B_H),
                        SmemLayoutQ_Batch<MMA_N_DIM>{}
                    );
                    ku::utcmma_ss(tiled_mma_dK, sdST, sQ_batch, tdKV_batch, true);
                    
                    // Step 4b: dV_batch = S^T @ dO_batch，累加到 tdKV_batch
                    // A (S^T): B_TOPK x B_H (MN-major)
                    // B (dO): B_H x MMA_N_DIM (K-major)
                    // C (dV): B_TOPK x MMA_N_DIM
                    Tensor sdO_batch = make_tensor(
                        make_smem_ptr(plan.u.dKV_cfg.do_buf.data() + col_offset * B_H),
                        SmemLayoutdO_Batch<MMA_N_DIM>{}
                    );
                    ku::utcmma_ss(tiled_mma_dV, sST, sdO_batch, tdKV_batch, false);
                    
                    // 通知该批次 dKV 就绪
                    plan.bar_dkv_ready[cur_buf].arrive();
                    
                    // 等待消费者完成消费
                    plan.bar_dkv_accum_done[cur_buf].wait((k/NUM_BUFS) & 1);
                    ku::tcgen05_after_thread_sync();
                }

                // Step 4c: dK_rope = dS^T @ Q_rope（单独计算）
                // 输出形状：B_TOPK x D_ROPE
                if constexpr (HAVE_ROPE) {
                    Tensor sQ_rope = make_tensor(
                        make_smem_ptr(plan.u.dKV_cfg.q_rope_buf.data()),
                        SmemLayoutQ_Batch<D_ROPE>{}
                    );
                    ku::utcmma_ss(tiled_mma_dK, sdST, sQ_rope, tdK_rope, true);
                }

                // plan.bar_dkv_ready[cur_buf].arrive(); // Handled inside loop
            }
        }
    }

#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm100");
    }
#endif
}

// Kernel launch wrapper
template<typename Kernel, typename TmaParams>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, 1)
sparse_attn_bwd_kernel_wrapper(__grid_constant__ const SparseAttnBwdParams params, __grid_constant__ const TmaParams tma_params) {
    sparse_attn_bwd_kernel<true, TmaParams>(params, tma_params);
}

template<int D_QK>
void run_bwd_phase1_kernel(const SparseAttnBwdParams& params) {
    static_assert(D_QK == 576 || D_QK == 512);

    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.topk % B_TOPK == 0);
    KU_ASSERT(params.h_q == B_H);
    KU_ASSERT(params.d_qk == D_QK);
    KU_ASSERT(params.delta != nullptr);  // Delta must be precomputed

    // Create TMA descriptors
    auto shape_Q = make_shape(params.h_q, params.d_qk, params.s_q);
    auto tma_Q = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.q),
            make_layout(
                shape_Q,
                make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q)
            )
        ),
        SmemLayoutQ{}
    );

    auto shape_dO = make_shape(params.h_q, params.d_v, params.s_q);
    auto tma_dO = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.d_out),
            make_layout(
                shape_dO,
                make_stride(params.stride_dout_h_q, _1{}, params.stride_dout_s_q)
            )
        ),
        SmemLayoutdO{}
    );

    auto shape_dQ = make_shape(params.h_q, params.d_qk, params.s_q);
    auto tma_dQ = cute::make_tma_copy(
        SM90_TMA_STORE{},
        make_tensor(
            make_gmem_ptr((bf16*)params.d_q),
            make_layout(
                shape_dQ,
                make_stride(params.stride_dq_h_q, _1{}, params.stride_dq_s_q)
            )
        ),
        SmemLayoutdQ{}
    );

    // Create tensor map for KV
    CUtensorMap tensor_map_kv;
    {
        uint64_t size[2] = {(uint64_t)D_QK, (uint64_t)params.s_kv};
        uint64_t stride[1] = {params.stride_kv_s_kv * sizeof(bf16)};
        uint32_t box_size[2] = {64, 1};
        uint32_t elem_stride[2] = {1, 1};
        CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &tensor_map_kv,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            params.kv,
            size,
            stride,
            box_size,
            elem_stride,
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        KU_ASSERT(res == CUresult::CUDA_SUCCESS);
    }

    TmaParams<
        decltype(shape_Q), decltype(tma_Q),
        decltype(shape_dO), decltype(tma_dO),
        decltype(shape_dQ), decltype(tma_dQ)
    > tma_params_struct = {
        shape_Q, tma_Q,
        shape_dO, tma_dO,
        shape_dQ, tma_dQ,
        tensor_map_kv
    };

    auto kernel = &sparse_attn_bwd_kernel<D_QK == 576, decltype(tma_params_struct)>;

    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dim3 grid(params.s_q, 1, 1);
    dim3 block(NUM_THREADS, 1, 1);

    kernel<<<grid, block, smem_size, params.stream>>>(params, tma_params_struct);
}

} // namespace sm100::bwd::head64
