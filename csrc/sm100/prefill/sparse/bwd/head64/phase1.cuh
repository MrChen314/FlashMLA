/**
 * @file phase1.cuh
 * @brief SM100 稀疏注意力反向传播核函数实现 (head_dim=64)
 * 
 * 本文件实现了针对 NVIDIA Blackwell (SM100) 架构优化的稀疏 MLA 注意力反向计算，
 * 使用 TMA (Tensor Memory Accelerator) 和 TMEM (Tensor Memory) 进行高效数据搬运。
 * 
 * 基于 TileLang 反向实现 (tilelang/examples/deepseek_v32/sparse_mla_bwd.py) 的计算流程:
 * - D=512, D_tail=64, block_H=64, BS=32
 * - 3 个 WarpGroup 并行计算
 */

#pragma once
#include "phase1.h"

#include <math_constants.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/arch.h>
#include <cutlass/cuda_host_adapter.hpp>

#include <kerutils/kerutils.cuh>

#include "params.h"
#include "sm100/helpers.h"
#include "sm100/prefill/sparse/common_subroutine.h"
#include "config.h"

namespace sm100::bwd::head64 {

using namespace cute;

/*
反向传播流水线概览 (Backward Pipeline Overview):
==================================================

本核函数采用 3 个 WarpGroup 并行计算模式:

WG0 (数据加载):
  - TMA 预取加载 Q/dO
  - 循环加载 KV blocks
  - 输出 dQ 到全局内存

WG1 (P/dKV 计算):
  1. QK 矩阵乘 (NoPE + RoPE) → P (TMEM)
  2. 计算 softmax(P) → S
  3. 存储 S 到 SMEM，通知 WG2
  4. S^T × dO → dKV (TMEM)
  5. 等待 WG2 的 ds
  6. ds^T × Q → dKV (累加)
  7. 原子写回 dKV 前半部分

WG2 (dP/dQ 计算):
  1. 计算 delta = sum(O * dO)
  2. dO × K^T → dP (TMEM)
  3. 等待 WG1 的 s
  4. ds = s * (dP - delta) * scale
  5. 存储 ds 到 SMEM，通知 WG1
  6. ds × K → dQ (TMEM)
  7. 原子写回 dKV 后半部分
*/

/**
 * @brief 稀疏注意力反向核函数
 * @tparam HAVE_ROPE 是否包含RoPE位置编码
 * @tparam TmaParams TMA参数类型
 * @param params 注意力反向计算参数
 * @param tma_params TMA描述符参数
 * 
 * 线程组织结构:
 * - Grid: [s_q, 1, 1] - 每个block处理一个query token
 * - Block: 384线程 = 3个warpgroup (每个128线程)
 *   - Warpgroup 0: 数据加载 (TMA)
 *   - Warpgroup 1: P/dKV 计算 (MMA)
 *   - Warpgroup 2: dP/dQ 计算 (MMA)
 */
template<bool HAVE_ROPE, typename TmaParams>
__global__ void __launch_bounds__(NUM_THREADS, 1, 1)
sparse_attn_bwd_kernel(__grid_constant__ const SparseAttnBwdParams params, __grid_constant__ const TmaParams tma_params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
    // Grid形状: [s_q, 1, 1] - 每个block处理一个query位置

    // === 线程索引计算 ===
    const int s_q_idx = blockIdx.x;                                     // 当前处理的query索引
    const int warp_idx = cutlass::canonical_warp_idx_sync();            // warp索引 (0-11)
    const int lane_idx = threadIdx.x % 32;                              // lane索引 (0-31)
    const int warpgroup_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);  // warpgroup索引 (0-2)
    const int idx_in_warpgroup = threadIdx.x % 128;                     // warpgroup内的线程索引
    const int topk_length = params.topk_length != nullptr ? __ldg(params.topk_length + s_q_idx) : params.topk;
    const int num_k_blocks = max(cute::ceil_div(topk_length, (int)B_TOPK), 1);

    // === 共享内存设置 ===
    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);

    int* gIndices = params.indices + s_q_idx * params.stride_indices_s_q;

    // === TMEM 张量分配 ===
    TiledMMA tiled_mma_P = TiledMMA_P{};
    TiledMMA tiled_mma_dQ = TiledMMA_dQ{};
    TiledMMA tiled_mma_dQ_rope = TiledMMA_dQ_RoPE{};
    TiledMMA tiled_mma_dKV = TiledMMA_dKV{};
    TiledMMA tiled_mma_dKV_rope = TiledMMA_dKV_RoPE{};

    // P 矩阵片段 (注意力分数): [B_H, B_TOPK] = [64, 32]
    Tensor tP = partition_fragment_C(tiled_mma_P, Shape<Int<B_H>, Int<B_TOPK>>{});
    
    // dP 矩阵片段: [B_H, B_TOPK] = [64, 32]
    Tensor tdP = partition_fragment_C(tiled_mma_P, Shape<Int<B_H>, Int<B_TOPK>>{});
    
    // dQ 累加器片段: [B_H, D_V] = [64, 512]
    Tensor tdQ = partition_fragment_C(tiled_mma_dQ, Shape<Int<B_H>, Int<D_V>>{});
    // dQ_RoPE 累加器片段: [B_H, D_ROPE] = [64, 64]
    Tensor tdQ_rope = partition_fragment_C(tiled_mma_dQ_rope, Shape<Int<B_H>, Int<D_ROPE>>{});
    
    // dKV 累加器片段: [B_TOPK, D_V] = [32, 512]
    Tensor tdKV = partition_fragment_C(tiled_mma_dKV, Shape<Int<B_TOPK>, Int<D_V>>{});
    // dKV_RoPE 累加器片段: [B_TOPK, D_ROPE] = [32, 64]
    Tensor tdKV_rope = partition_fragment_C(tiled_mma_dKV_rope, Shape<Int<B_TOPK>, Int<D_ROPE>>{});

    // 设置 TMEM 基地址
    tP.data().get() = tmem_cols::P;
    tdP.data().get() = tmem_cols::dP;
    tdQ.data().get() = tmem_cols::dQ;
    tdQ_rope.data().get() = tmem_cols::dQ_RoPE;
    tdKV.data().get() = tmem_cols::dKV;
    tdKV_rope.data().get() = tmem_cols::dKV_RoPE;

    // === Warp 0: 序言阶段 - 初始化barrier和加载Q/dO ===
    if (warp_idx == 0) {
        if (elect_one_sync()) {
            // --- 预取 TMA 描述符 ---
            cute::prefetch_tma_descriptor(tma_params.tma_Q_nope.get_tma_descriptor());
            if constexpr (HAVE_ROPE) {
                cute::prefetch_tma_descriptor(tma_params.tma_Q_rope.get_tma_descriptor());
            }
            cute::prefetch_tma_descriptor(tma_params.tma_dO.get_tma_descriptor());
            cute::prefetch_tma_descriptor(tma_params.tma_dQ.get_tma_descriptor());

            // --- 初始化 barrier ---
            plan.bar_prologue_q_nope.init(1);
            plan.bar_prologue_q_rope.init(1);
            plan.bar_prologue_dO.init(1);
            plan.bar_prologue_utccp.init(1);
            plan.bar_qk_nope_done.init(1);
            plan.bar_qk_rope_done.init(1);
            plan.bar_dp_done.init(1);
            plan.bar_dq_done.init(1);
            plan.bar_dkv_done.init(1);
            plan.bar_kv_nope_ready.init(1);
            plan.bar_kv_rope_ready.init(1);
            plan.bar_p_free.init(128);
            plan.bar_dp_free.init(128);
            plan.bar_s_do_ready.init(128);
            plan.bar_kv_valid_ready.init(B_TOPK/8);
            plan.bar_kv_valid_free.init(128);
            fence_barrier_init();

            // --- 使用 TMA 加载 Q 矩阵 ---
            // 加载 Q NoPE 部分
            Tensor gQ_nope = tma_params.tma_Q_nope.get_tma_tensor(tma_params.shape_Q_nope)(_, _, s_q_idx);
            Tensor sQ_nope = make_tensor(make_smem_ptr(plan.u.q_kv.q_nope.data()), SmemLayoutQNoPE{});
            ku::launch_tma_copy(tma_params.tma_Q_nope, gQ_nope, sQ_nope, plan.bar_prologue_q_nope, TMA::CacheHintSm90::EVICT_FIRST);

            // 加载 Q RoPE 部分
            if constexpr (HAVE_ROPE) {
                Tensor gQ_rope = tma_params.tma_Q_rope.get_tma_tensor(tma_params.shape_Q_rope)(_, _, s_q_idx);
                Tensor sQ_rope = make_tensor(make_smem_ptr(plan.u.q_kv.q_rope.data()), SmemLayoutQRoPE{});
                ku::launch_tma_copy(tma_params.tma_Q_rope, gQ_rope, sQ_rope, plan.bar_prologue_q_rope, TMA::CacheHintSm90::EVICT_FIRST);
            }

            // --- 使用 TMA 加载 dO 矩阵 ---
            Tensor gdO = tma_params.tma_dO.get_tma_tensor(tma_params.shape_dO)(_, _, s_q_idx);
            Tensor sdO = make_tensor(make_smem_ptr(plan.dO.data()), SmemLayoutdO{});
            ku::launch_tma_copy(tma_params.tma_dO, gdO, sdO, plan.bar_prologue_dO, TMA::CacheHintSm90::EVICT_FIRST);

            // 预取 KV TMA 描述符
            cute::prefetch_tma_descriptor(&(tma_params.tensor_map_kv_nope));
            if constexpr (HAVE_ROPE) {
                cute::prefetch_tma_descriptor(&(tma_params.tensor_map_kv_rope));
            }
        }

        // --- 分配 TMEM ---
        cute::TMEM::Allocator1Sm().allocate(512, plan.tmem_start_addr.data());
        TRAP_ONLY_DEVICE_ASSERT(plan.tmem_start_addr.data()[0] == 0);
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }

    __syncthreads();

    // ========================================
    // Warpgroup 0: 数据加载单元
    // 负责: TMA 加载 KV，TMA 输出 dQ
    // ========================================
    if (warpgroup_idx == 0) {
        int local_warp_idx = warp_idx;
        constexpr int NUM_WARPS = 4, NUM_LOCAL_ROWS_PER_WARP = (B_TOPK/4)/NUM_WARPS;

        // === 主循环: 加载 KV blocks ===
        if (elect_one_sync()) {
            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks; ++k) {
                // --- 加载 TopK 索引 ---
                int4 indices[NUM_LOCAL_ROWS_PER_WARP];
                int max_indices = -1, min_indices = params.s_kv;
                CUTE_UNROLL
                for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                    indices[local_row] = __ldg((int4*)(gIndices + k*B_TOPK) + local_row*NUM_WARPS + local_warp_idx);
                    max_indices = max(max_indices, int4_max(indices[local_row]));
                    min_indices = min(min_indices, int4_min(indices[local_row]));
                }
                bool is_all_rows_invalid = min_indices == params.s_kv || max_indices == -1;
                bool should_skip_tma = is_all_rows_invalid && k >= NUM_BUFS;

                // 等待上一轮计算完成
                if (k > 0) {
                    plan.bar_dkv_done.wait((k-1)&1);
                }

                // --- TMA gather 加载 KV NoPE ---
                bf16* sKV_nope_base = plan.u.q_kv.kv_nope.data() + local_warp_idx*4*64;
                if (!should_skip_tma) {
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                        CUTE_UNROLL
                        for (int local_col = 0; local_col < D_V/64; ++local_col) {
                            ku::tma_gather4(
                                &(tma_params.tensor_map_kv_nope),
                                plan.bar_kv_nope_ready,
                                sKV_nope_base + local_row*(4*NUM_WARPS)*64 + local_col*(B_TOPK*64),
                                local_col*64,
                                indices[local_row],
                                (int64_t)TMA::CacheHintSm90::EVICT_LAST
                            );
                        }
                    }
                } else {
                    plan.bar_kv_nope_ready.complete_transaction(NUM_LOCAL_ROWS_PER_WARP*4*D_V*sizeof(bf16));
                }

                // --- TMA gather 加载 KV RoPE ---
                if constexpr (HAVE_ROPE) {
                    bf16* sKV_rope_base = plan.u.q_kv.kv_rope.data() + local_warp_idx*4*64;
                    if (!should_skip_tma) {
                        CUTE_UNROLL
                        for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                            ku::tma_gather4(
                                &(tma_params.tensor_map_kv_rope),
                                plan.bar_kv_rope_ready,
                                sKV_rope_base + local_row*(4*NUM_WARPS)*64,
                                D_V,  // RoPE 部分从 D_V 偏移开始
                                indices[local_row],
                                (int64_t)TMA::CacheHintSm90::EVICT_LAST
                            );
                        }
                    } else {
                        plan.bar_kv_rope_ready.complete_transaction(NUM_LOCAL_ROWS_PER_WARP*4*D_ROPE*sizeof(bf16));
                    }
                }
            }
        }

        // === Epilogue: 输出 dQ ===
        // 等待所有计算完成
        plan.bar_dq_done.wait((num_k_blocks-1)&1);
        ku::tcgen05_after_thread_sync();

        // 从 TMEM 读取 dQ 并转换为 bf16 存储到 SMEM
        if (local_warp_idx < 2) {
            constexpr int CHUNK_SIZE = 64;
            float2 dq_chunk[CHUNK_SIZE/2];
            
            // 处理 dQ NoPE 部分
            CUTE_UNROLL
            for (int chunk_idx = 0; chunk_idx < D_V/CHUNK_SIZE; ++chunk_idx) {
                ku::tmem_ld_32dp32bNx<CHUNK_SIZE>(tmem_cols::dQ + chunk_idx*CHUNK_SIZE, dq_chunk);
                cutlass::arch::fence_view_async_tmem_load();
                
                bf16* sdQ_base = plan.u.dq.dq_nope.data() + idx_in_warpgroup*D_V + chunk_idx*CHUNK_SIZE;
                CUTE_UNROLL
                for (int i = 0; i < CHUNK_SIZE/2; ++i) {
                    nv_bfloat162 dq_bf16 = __float22bfloat162_rn(dq_chunk[i]);
                    *(nv_bfloat162*)(sdQ_base + i*2) = dq_bf16;
                }
            }
            
            // 处理 dQ RoPE 部分
            if constexpr (HAVE_ROPE) {
                ku::tmem_ld_32dp32bNx<D_ROPE>(tmem_cols::dQ_RoPE, dq_chunk);
                cutlass::arch::fence_view_async_tmem_load();
                
                bf16* sdQ_rope_base = plan.u.dq.dq_rope.data() + idx_in_warpgroup*D_ROPE;
                CUTE_UNROLL
                for (int i = 0; i < D_ROPE/2; ++i) {
                    nv_bfloat162 dq_bf16 = __float22bfloat162_rn(dq_chunk[i]);
                    *(nv_bfloat162*)(sdQ_rope_base + i*2) = dq_bf16;
                }
            }
        }

        fence_view_async_shared();
        NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);

        // TMA 存储 dQ 到全局内存
        if (local_warp_idx == 0 && elect_one_sync()) {
            Tensor gdQ = tma_params.tma_dQ.get_tma_tensor(tma_params.shape_dQ)(_, _, s_q_idx);
            Tensor sdQ_nope = make_tensor(make_smem_ptr(plan.u.dq.dq_nope.data()), SmemLayoutdQNoPE{});
            cute::copy(tma_params.tma_dQ, sdQ_nope, gdQ);
        }

        // 释放 TMEM
        if (local_warp_idx == 0) {
            cute::TMEM::Allocator1Sm().free(0, 512);
        }

    // ========================================
    // Warpgroup 1: P/dKV 计算单元
    // 负责: QK→P, softmax, S×dO→dKV, ds×Q→dKV
    // ========================================
    } else if (warpgroup_idx == 1) {
        int local_warp_idx = warp_idx - 4;

        // 构建 SMEM 张量
        Tensor sQ_nope = make_tensor(make_smem_ptr(plan.u.q_kv.q_nope.data()), SmemLayoutQNoPE{});
        Tensor sQ_rope = make_tensor(make_smem_ptr(plan.u.q_kv.q_rope.data()), SmemLayoutQRoPE{});
        Tensor sKV_nope = make_tensor(make_smem_ptr(plan.u.q_kv.kv_nope.data()), SmemLayoutKVNoPE{});
        Tensor sKV_rope = make_tensor(make_smem_ptr(plan.u.q_kv.kv_rope.data()), SmemLayoutKVRoPE{});
        Tensor sdO = make_tensor(make_smem_ptr(plan.dO.data()), SmemLayoutdO{});
        Tensor sS = make_tensor(make_smem_ptr(plan.s_ds.s.data()), SmemLayoutS{});

        // 等待 Q 加载完成
        plan.bar_prologue_q_nope.arrive_and_expect_tx(B_H*D_V*sizeof(bf16));
        plan.bar_prologue_q_nope.wait(0);
        if constexpr (HAVE_ROPE) {
            plan.bar_prologue_q_rope.arrive_and_expect_tx(B_H*D_ROPE*sizeof(bf16));
            plan.bar_prologue_q_rope.wait(0);
        }
        plan.bar_prologue_dO.arrive_and_expect_tx(B_H*D_V*sizeof(bf16));
        plan.bar_prologue_dO.wait(0);

        // S 矩阵在共享内存中的基地址
        bf16* sS_base = plan.s_ds.s.data() + lane_idx*8 + (local_warp_idx&1)*(B_H/2)*8 + (local_warp_idx/2)*B_H*(B_TOPK/2);
        constexpr int NUM_ELEMS_PER_THREAD = B_TOPK / 2;

        // === 主循环: 计算 P, softmax, dKV ===
        CUTE_NO_UNROLL
        for (int k = 0; k < num_k_blocks; ++k) {
            // --- 等待 KV 数据就绪 ---
            plan.bar_kv_nope_ready.arrive_and_expect_tx(B_TOPK*D_V*sizeof(bf16));
            plan.bar_kv_nope_ready.wait(k&1);
            if constexpr (HAVE_ROPE) {
                plan.bar_kv_rope_ready.arrive_and_expect_tx(B_TOPK*D_ROPE*sizeof(bf16));
                plan.bar_kv_rope_ready.wait(k&1);
            }
            plan.bar_kv_valid_ready.wait(k&1);
            ku::tcgen05_after_thread_sync();

            // --- 计算 P = Q × K^T ---
            // 等待 P 缓冲区空闲
            plan.bar_p_free.wait(k&1^1);
            ku::tcgen05_after_thread_sync();

            // NoPE 部分
            ku::utcmma_ss(tiled_mma_P, sQ_nope, sKV_nope, tP, true);  // clear_accum=true

            // RoPE 部分
            if constexpr (HAVE_ROPE) {
                ku::utcmma_ss(tiled_mma_P, sQ_rope, sKV_rope, tP, false);  // 累加
            }
            ku::umma_arrive_noelect(plan.bar_qk_nope_done);

            // --- 等待 P 计算完成，计算 softmax ---
            plan.bar_qk_nope_done.wait(k&1);
            ku::tcgen05_after_thread_sync();

            // 从 TMEM 加载 P 并应用 mask
            float p[NUM_ELEMS_PER_THREAD];
            retrieve_mask_and_reduce_p<
                NUM_ELEMS_PER_THREAD,
                tmem_cols::P,
                NamedBarriers::wg0_warp02_sync,
                NamedBarriers::wg0_warp13_sync,
                false
            >(
                plan.is_kv_valid,
                local_warp_idx, lane_idx,
                [&]() { plan.bar_p_free.arrive(); },
                plan.p_exchange_buf,
                p
            );
            plan.bar_kv_valid_free.arrive();

            // 计算 softmax: S = exp2(P*scale - LSE)
            float lse = params.lse[s_q_idx * params.h_q + idx_in_warpgroup];
            nv_bfloat162 s[NUM_ELEMS_PER_THREAD/2];
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD/2; ++i) {
                float2 p_scaled;
                p_scaled.x = exp2f(p[i*2] * params.sm_scale_div_log2 - lse);
                p_scaled.y = exp2f(p[i*2+1] * params.sm_scale_div_log2 - lse);
                s[i] = __float22bfloat162_rn(p_scaled);
            }

            // 将 S 存储到 SMEM
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD/8; i += 1) {
                *(uint128_t*)(sS_base + B_H*8*i) = *(uint128_t*)(s + i*4);
            }
            fence_view_async_shared();
            plan.bar_s_do_ready.arrive();  // 通知 WG2: S 已就绪

            // --- 计算 dKV 第一部分: S^T × dO → dKV ---
            NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
            ku::tcgen05_after_thread_sync();

            // dKV = S^T × dO (clear_accum=true for first iteration)
            ku::utcmma_ss(tiled_mma_dKV, sS, sdO, tdKV, k == 0);

            // --- 等待 WG2 的 ds，计算 dKV 第二部分 ---
            plan.bar_dp_done.wait(k&1);
            ku::tcgen05_after_thread_sync();

            Tensor sDS = make_tensor(make_smem_ptr(plan.s_ds.ds.data()), SmemLayoutS{});

            // dKV += ds^T × Q_nope
            ku::utcmma_ss(tiled_mma_dKV, sDS, sQ_nope, tdKV, false);

            // dKV_RoPE = ds^T × Q_rope (只在有 RoPE 时)
            if constexpr (HAVE_ROPE) {
                ku::utcmma_ss(tiled_mma_dKV_rope, sDS, sQ_rope, tdKV_rope, k == 0);
            }

            // --- 原子写回 dKV 前半部分 ---
            ku::tcgen05_after_thread_sync();

            // 从 TMEM 读取 dKV 并原子写回全局内存
            constexpr int HALF_ROWS = B_TOPK / SPLIT_STORE;
            float2 dkv_chunk[32];
            
            // 写回 NoPE 部分前半
            CUTE_UNROLL
            for (int col_chunk = 0; col_chunk < D_V/64; ++col_chunk) {
                ku::tmem_ld_32dp32bNx<64>(tmem_cols::dKV + col_chunk*64, dkv_chunk);
                cutlass::arch::fence_view_async_tmem_load();
                
                // 原子累加到全局内存
                CUTE_UNROLL
                for (int row = 0; row < HALF_ROWS; ++row) {
                    int kv_idx = gIndices[k*B_TOPK + row];
                    if (kv_idx >= 0 && kv_idx < params.s_kv) {
                        float* dKV_ptr = params.dKV + kv_idx * params.stride_dKV_s_kv + col_chunk*64;
                        CUTE_UNROLL
                        for (int j = 0; j < 64/4; ++j) {
                            atomicAdd((float4*)(dKV_ptr + j*4), *(float4*)(dkv_chunk + row*32 + j*2));
                        }
                    }
                }
            }

            // 写回 RoPE 部分前半
            if constexpr (HAVE_ROPE) {
                ku::tmem_ld_32dp32bNx<D_ROPE>(tmem_cols::dKV_RoPE, dkv_chunk);
                cutlass::arch::fence_view_async_tmem_load();
                
                CUTE_UNROLL
                for (int row = 0; row < HALF_ROWS; ++row) {
                    int kv_idx = gIndices[k*B_TOPK + row];
                    if (kv_idx >= 0 && kv_idx < params.s_kv) {
                        float* dKV_ptr = params.dKV + kv_idx * params.stride_dKV_s_kv + D_V;
                        CUTE_UNROLL
                        for (int j = 0; j < D_ROPE/4; ++j) {
                            atomicAdd((float4*)(dKV_ptr + j*4), *(float4*)(dkv_chunk + row*32 + j*2));
                        }
                    }
                }
            }

            ku::umma_arrive_noelect(plan.bar_dkv_done);
        }

    // ========================================
    // Warpgroup 2: dP/dQ 计算单元
    // 负责: delta计算, dO×K→dP, ds计算, ds×K→dQ
    // ========================================
    } else {
        int local_warp_idx = warp_idx - 8;

        // 构建 SMEM 张量
        Tensor sQ_nope = make_tensor(make_smem_ptr(plan.u.q_kv.q_nope.data()), SmemLayoutQNoPE{});
        Tensor sQ_rope = make_tensor(make_smem_ptr(plan.u.q_kv.q_rope.data()), SmemLayoutQRoPE{});
        Tensor sKV_nope = make_tensor(make_smem_ptr(plan.u.q_kv.kv_nope.data()), SmemLayoutKVNoPE{});
        Tensor sKV_rope = make_tensor(make_smem_ptr(plan.u.q_kv.kv_rope.data()), SmemLayoutKVRoPE{});
        Tensor sdO = make_tensor(make_smem_ptr(plan.dO.data()), SmemLayoutdO{});
        Tensor sS = make_tensor(make_smem_ptr(plan.s_ds.s.data()), SmemLayoutS{});
        Tensor sDS = make_tensor(make_smem_ptr(plan.s_ds.ds.data()), SmemLayoutS{});

        // === 计算 Delta = sum(O * dO) ===
        // 从全局内存加载 O，从 SMEM 读取 dO
        float delta = 0.0f;
        bf16* gO = params.o + s_q_idx * params.stride_o_s_q + idx_in_warpgroup * D_V;
        bf16* sdO_base = plan.dO.data() + idx_in_warpgroup * D_V;

        // 等待 dO 加载完成
        plan.bar_prologue_dO.arrive_and_expect_tx(B_H*D_V*sizeof(bf16));
        plan.bar_prologue_dO.wait(0);

        // 计算 delta = sum(O[h,:] * dO[h,:])
        CUTE_UNROLL
        for (int d = 0; d < D_V; d += 2) {
            nv_bfloat162 o_val = *(nv_bfloat162*)(gO + d);
            nv_bfloat162 dO_val = *(nv_bfloat162*)(sdO_base + d);
            float2 o_f = __bfloat1622float2(o_val);
            float2 dO_f = __bfloat1622float2(dO_val);
            delta += o_f.x * dO_f.x + o_f.y * dO_f.y;
        }

        // 存储 delta 到共享内存
        plan.rowwise_delta_buf[idx_in_warpgroup] = delta;

        // 等待 Q/KV 加载完成
        plan.bar_prologue_q_nope.arrive_and_expect_tx(B_H*D_V*sizeof(bf16));
        plan.bar_prologue_q_nope.wait(0);
        if constexpr (HAVE_ROPE) {
            plan.bar_prologue_q_rope.arrive_and_expect_tx(B_H*D_ROPE*sizeof(bf16));
            plan.bar_prologue_q_rope.wait(0);
        }

        // dS 矩阵在共享内存中的基地址
        bf16* sDS_base = plan.s_ds.ds.data() + lane_idx*8 + (local_warp_idx&1)*(B_H/2)*8 + (local_warp_idx/2)*B_H*(B_TOPK/2);
        constexpr int NUM_ELEMS_PER_THREAD = B_TOPK / 2;

        // === 主循环: 计算 dP, ds, dQ ===
        CUTE_NO_UNROLL
        for (int k = 0; k < num_k_blocks; ++k) {
            // --- 等待 KV 数据就绪 ---
            plan.bar_kv_nope_ready.arrive_and_expect_tx(B_TOPK*D_V*sizeof(bf16));
            plan.bar_kv_nope_ready.wait(k&1);
            if constexpr (HAVE_ROPE) {
                plan.bar_kv_rope_ready.arrive_and_expect_tx(B_TOPK*D_ROPE*sizeof(bf16));
                plan.bar_kv_rope_ready.wait(k&1);
            }
            ku::tcgen05_after_thread_sync();

            // --- 计算 dP_partial = dO × K^T ---
            plan.bar_dp_free.wait(k&1^1);
            ku::tcgen05_after_thread_sync();

            ku::utcmma_ss(tiled_mma_P, sdO, sKV_nope, tdP, true);
            ku::umma_arrive_noelect(plan.bar_dp_done);

            // --- 等待 WG1 的 S，计算 ds = s * (dP - delta) * scale ---
            plan.bar_s_do_ready.wait(k&1);
            ku::tcgen05_after_thread_sync();

            // 从 TMEM 加载 dP
            float dp[NUM_ELEMS_PER_THREAD];
            ku::tmem_ld_32dp32bNx<NUM_ELEMS_PER_THREAD>(tmem_cols::dP, dp);
            cutlass::arch::fence_view_async_tmem_load();
            ku::tcgen05_before_thread_sync();
            plan.bar_dp_free.arrive();

            // 从 SMEM 加载 S
            bf16* sS_base = plan.s_ds.s.data() + lane_idx*8 + (local_warp_idx&1)*(B_H/2)*8 + (local_warp_idx/2)*B_H*(B_TOPK/2);
            nv_bfloat162 s[NUM_ELEMS_PER_THREAD/2];
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD/8; ++i) {
                *(uint128_t*)(s + i*4) = *(uint128_t*)(sS_base + B_H*8*i);
            }

            // 读取 delta
            float my_delta = plan.rowwise_delta_buf[idx_in_warpgroup];

            // 计算 ds = s * (dp - delta) * scale
            nv_bfloat162 ds[NUM_ELEMS_PER_THREAD/2];
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD/2; ++i) {
                float2 s_f = __bfloat1622float2(s[i]);
                float2 ds_f;
                ds_f.x = s_f.x * (dp[i*2] - my_delta) * params.sm_scale;
                ds_f.y = s_f.y * (dp[i*2+1] - my_delta) * params.sm_scale;
                ds[i] = __float22bfloat162_rn(ds_f);
            }

            // 将 ds 存储到 SMEM
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD/8; ++i) {
                *(uint128_t*)(sDS_base + B_H*8*i) = *(uint128_t*)(ds + i*4);
            }
            fence_view_async_shared();
            ku::umma_arrive_noelect(plan.bar_dp_done);  // 通知 WG1: ds 已就绪

            // --- 计算 dQ = ds × K ---
            NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
            ku::tcgen05_after_thread_sync();

            // dQ += ds × K_nope
            ku::utcmma_ss(tiled_mma_dQ, sDS, sKV_nope, tdQ, k == 0);

            // dQ_RoPE += ds × K_rope
            if constexpr (HAVE_ROPE) {
                ku::utcmma_ss(tiled_mma_dQ_rope, sDS, sKV_rope, tdQ_rope, k == 0);
            }

            // --- 原子写回 dKV 后半部分 ---
            constexpr int HALF_ROWS = B_TOPK / SPLIT_STORE;
            float2 dkv_chunk[32];

            // 写回 NoPE 部分后半
            CUTE_UNROLL
            for (int col_chunk = 0; col_chunk < D_V/64; ++col_chunk) {
                ku::tmem_ld_32dp32bNx<64>(tmem_cols::dKV + col_chunk*64, dkv_chunk);
                cutlass::arch::fence_view_async_tmem_load();
                
                CUTE_UNROLL
                for (int row = HALF_ROWS; row < B_TOPK; ++row) {
                    int kv_idx = gIndices[k*B_TOPK + row];
                    if (kv_idx >= 0 && kv_idx < params.s_kv) {
                        float* dKV_ptr = params.dKV + kv_idx * params.stride_dKV_s_kv + col_chunk*64;
                        CUTE_UNROLL
                        for (int j = 0; j < 64/4; ++j) {
                            atomicAdd((float4*)(dKV_ptr + j*4), *(float4*)(dkv_chunk + (row-HALF_ROWS)*32 + j*2));
                        }
                    }
                }
            }

            // 写回 RoPE 部分后半
            if constexpr (HAVE_ROPE) {
                ku::tmem_ld_32dp32bNx<D_ROPE>(tmem_cols::dKV_RoPE, dkv_chunk);
                cutlass::arch::fence_view_async_tmem_load();
                
                CUTE_UNROLL
                for (int row = HALF_ROWS; row < B_TOPK; ++row) {
                    int kv_idx = gIndices[k*B_TOPK + row];
                    if (kv_idx >= 0 && kv_idx < params.s_kv) {
                        float* dKV_ptr = params.dKV + kv_idx * params.stride_dKV_s_kv + D_V;
                        CUTE_UNROLL
                        for (int j = 0; j < D_ROPE/4; ++j) {
                            atomicAdd((float4*)(dKV_ptr + j*4), *(float4*)(dkv_chunk + (row-HALF_ROWS)*32 + j*2));
                        }
                    }
                }
            }

            ku::umma_arrive_noelect(plan.bar_dq_done);
        }
    }

#else
    // 非SM100架构的错误处理
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm100");
    }
#endif
}

/**
 * @brief 启动反向Phase1核函数的主机端包装函数
 * @tparam D_QK Query/Key的维度 (576或512)
 * @param params 注意力反向计算参数结构体
 * 
 * 功能:
 * 1. 参数校验
 * 2. 创建TMA描述符 (Q_nope, Q_rope, dO, dQ, KV)
 * 3. 配置并启动CUDA核函数
 */
template<int D_QK>
void run_bwd_phase1_kernel(const SparseAttnBwdParams& params) {
    // === 参数校验 ===
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.topk % B_TOPK == 0);
    KU_ASSERT(params.h_q == B_H);
    KU_ASSERT(params.d_qk == D_QK);
    static_assert(D_QK == 576 || D_QK == 512);

    // === 创建 Q NoPE 的 TMA 描述符 ===
    auto shape_Q_nope = make_shape(params.h_q, D_V, params.s_q);
    auto tma_Q_nope = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.q),
            make_layout(
                shape_Q_nope,
                make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q)
            )
        ),
        SmemLayoutQNoPE{}
    );

    // === 创建 Q RoPE 的 TMA 描述符 ===
    auto shape_Q_rope = make_shape(params.h_q, D_ROPE, params.s_q);
    auto tma_Q_rope = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.q + D_V),
            make_layout(
                shape_Q_rope,
                make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q)
            )
        ),
        SmemLayoutQRoPE{}
    );

    // === 创建 dO 的 TMA 描述符 ===
    auto shape_dO = make_shape(params.h_q, params.d_v, params.s_q);
    auto tma_dO = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.dO),
            make_layout(
                shape_dO,
                make_stride(params.stride_dO_h_q, _1{}, params.stride_dO_s_q)
            )
        ),
        SmemLayoutdO{}
    );

    // === 创建 dQ 的 TMA 描述符 ===
    auto shape_dQ = make_shape(params.h_q, D_QK, params.s_q);
    auto tma_dQ = cute::make_tma_copy(
        SM90_TMA_STORE{},
        make_tensor(
            make_gmem_ptr((bf16*)params.dQ),
            make_layout(
                shape_dQ,
                make_stride(params.stride_dQ_h_q, _1{}, params.stride_dQ_s_q)
            )
        ),
        SmemLayoutdQNoPE{}
    );

    // === 创建 KV NoPE 的 TensorMap ===
    CUtensorMap tensor_map_kv_nope;
    {
        uint64_t size[2] = {D_V, (unsigned long)params.s_kv};
        uint64_t stride[1] = {params.stride_kv_s_kv*sizeof(bf16)};
        uint32_t box_size[2] = {64, 1};
        uint32_t elem_stride[2] = {1, 1};
        CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &tensor_map_kv_nope,
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

    // === 创建 KV RoPE 的 TensorMap ===
    CUtensorMap tensor_map_kv_rope;
    {
        uint64_t size[2] = {D_ROPE, (unsigned long)params.s_kv};
        uint64_t stride[1] = {params.stride_kv_s_kv*sizeof(bf16)};
        uint32_t box_size[2] = {64, 1};
        uint32_t elem_stride[2] = {1, 1};
        CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &tensor_map_kv_rope,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            (bf16*)params.kv + D_V,
            size,
            stride,
            box_size,
            elem_stride,
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        KU_ASSERT(res == CUresult::CUDA_SUCCESS);
    }

    // === 组装 TMA 参数结构体 ===
    TmaParams<
        decltype(shape_Q_nope), decltype(tma_Q_nope),
        decltype(shape_Q_rope), decltype(tma_Q_rope),
        decltype(shape_dO), decltype(tma_dO),
        decltype(shape_dQ), decltype(tma_dQ)
    > tma_params = {
        shape_Q_nope, tma_Q_nope,
        shape_Q_rope, tma_Q_rope,
        shape_dO, tma_dO,
        shape_dQ, tma_dQ,
        tensor_map_kv_nope,
        tensor_map_kv_rope
    };

    // D_QK==576 时启用 RoPE
    auto kernel = &sparse_attn_bwd_kernel<D_QK == 576, decltype(tma_params)>;

    // === 配置并启动核函数 ===
    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    // Grid: s_q 个 block，每个 block 处理一个 query token
    // Block: NUM_THREADS (384) 个线程
    kernel<<<params.s_q, NUM_THREADS, smem_size, params.stream>>>(params, tma_params);
    KU_CHECK_KERNEL_LAUNCH();
}

}  // namespace sm100::bwd::head64
