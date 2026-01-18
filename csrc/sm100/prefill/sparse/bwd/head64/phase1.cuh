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
    TiledMMA tiled_mma_dQ = TiledMMA_dQ{};
    TiledMMA tiled_mma_dK = TiledMMA_dK{};
    TiledMMA tiled_mma_dV = TiledMMA_dV{};

    // TMEM tensor handles
    Tensor tP = partition_fragment_C(tiled_mma_P, Shape<Int<B_H>, _128>{});
    Tensor tdQ = partition_fragment_C(tiled_mma_dQ, Shape<Int<B_H>, Int<D_V>>{});
    Tensor tdQ_rope = partition_fragment_C(tiled_mma_dQ, Shape<Int<B_H>, Int<D_Q-D_V>>{});
    Tensor tdK = partition_fragment_C(tiled_mma_dK, Shape<Int<B_TOPK>, Int<D_V>>{});
    Tensor tdK_rope = partition_fragment_C(tiled_mma_dK, Shape<Int<B_TOPK>, Int<D_Q-D_V>>{});
    Tensor tdV = partition_fragment_C(tiled_mma_dV, Shape<Int<B_TOPK>, Int<D_V>>{});

    tP.data().get() = tmem_cols::P;
    tdQ.data().get() = tmem_cols::dQ;
    tdQ_rope.data().get() = tmem_cols::dQ + D_V/2;  // After NoPE part
    tdK.data().get() = tmem_cols::dKV;
    tdK_rope.data().get() = tmem_cols::dKV;  // Reuse same space, computed separately
    tdV.data().get() = tmem_cols::dKV;

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
                plan.bar_k_valid_ready[i].init(B_TOPK/8);
                plan.bar_k_valid_free[i].init(128);
            }
            plan.bar_p_free.init(128);
            plan.bar_so_ready.init(128);
            fence_barrier_init();
        }

        // Initialize TMEM
        cute::TMEM::Allocator1Sm().allocate(512, plan.tmem_start_addr.data());
        TRAP_ONLY_DEVICE_ASSERT(plan.tmem_start_addr.data()[0] == 0);
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

        bf16* sS_base = plan.s_q_rope.s + lane_idx * 8 + (warp_idx & 1) * (B_H/2) * 8 + (warp_idx/2) * B_H * (B_TOPK/2);
        bf16* sdS_base = sS_base;  // dS stored in same location
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

            // Store S to shared memory for dV computation (S^T @ dO)
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD/8; i += 1) {
                *(uint128_t*)(sS_base + B_H * 8 * i) = *(uint128_t*)(s + i*4);
            }

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

            // Store dS to shared memory for dQ and dK computation
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD/8; i += 1) {
                *(uint128_t*)(sdS_base + B_H * 8 * i) = *(uint128_t*)(ds + i*4);
            }

            fence_view_async_shared();
            plan.bar_so_ready.arrive();

            // ===== Atomic accumulation of dK and dV =====
            // Wait for dK/dV to be computed by MMA warpgroup
            plan.bar_dkv_ready[cur_buf].wait((k/NUM_BUFS) & 1);
            ku::tcgen05_after_thread_sync();

            // Load dK from TMEM and atomically add to global memory
            // Each thread handles a portion of the dK/dV matrix
            const int local_warp = warp_idx;  // 0-3
            const int rows_per_warp = B_TOPK / 4;  // 16 rows per warp
            const int row_start = local_warp * rows_per_warp;
            
            // Load indices for this block
            int indices_buf[rows_per_warp];
            CUTE_UNROLL
            for (int r = 0; r < rows_per_warp; ++r) {
                indices_buf[r] = __ldg(gIndices + k * B_TOPK + row_start + r);
            }

            // Atomic add dK (NoPE part: D_V = 512 dimensions)
            // Each lane handles 512/32 = 16 elements
            {
                float2 dk_vals[D_V / 64];  // 8 float2 values per lane
                ku::tmem_ld_32dp32bNx<D_V/32>(tmem_cols::dKV + lane_idx * (D_V/64), dk_vals);
                cutlass::arch::fence_view_async_tmem_load();
                
                CUTE_UNROLL
                for (int r = 0; r < rows_per_warp; ++r) {
                    int kv_idx = indices_buf[r];
                    if (kv_idx >= 0 && kv_idx < params.s_kv) {
                        float* dk_ptr = gDKV + kv_idx * params.stride_dkv_s_kv;
                        CUTE_UNROLL
                        for (int d = 0; d < D_V / 64; ++d) {
                            atomic_add_float2(dk_ptr + lane_idx * 2 + d * 64, dk_vals[d]);
                        }
                    }
                }
            }

            // Atomic add dK (RoPE part: 64 dimensions)
            if constexpr (HAVE_ROPE) {
                float2 dk_rope_vals[1];  // 2 elements per lane for 64-dim RoPE
                ku::tmem_ld_32dp32bNx<1>(tmem_cols::dKV + D_V/2 + lane_idx, dk_rope_vals);
                cutlass::arch::fence_view_async_tmem_load();
                
                CUTE_UNROLL
                for (int r = 0; r < rows_per_warp; ++r) {
                    int kv_idx = indices_buf[r];
                    if (kv_idx >= 0 && kv_idx < params.s_kv) {
                        float* dk_ptr = gDKV + kv_idx * params.stride_dkv_s_kv + D_V;
                        atomic_add_float2(dk_ptr + lane_idx * 2, dk_rope_vals[0]);
                    }
                }
            }

            // Atomic add dV (D_V = 512 dimensions)
            {
                float2 dv_vals[D_V / 64];
                // dV is stored after dK in TMEM (or recomputed)
                ku::tmem_ld_32dp32bNx<D_V/32>(tmem_cols::dKV + lane_idx * (D_V/64), dv_vals);
                cutlass::arch::fence_view_async_tmem_load();
                
                CUTE_UNROLL
                for (int r = 0; r < rows_per_warp; ++r) {
                    int kv_idx = indices_buf[r];
                    if (kv_idx >= 0 && kv_idx < params.s_kv) {
                        // dV is stored in the V part of KV (first D_V dimensions)
                        float* dv_ptr = gDKV + kv_idx * params.stride_dkv_s_kv;
                        CUTE_UNROLL
                        for (int d = 0; d < D_V / 64; ++d) {
                            atomic_add_float2(dv_ptr + lane_idx * 2 + d * 64, dv_vals[d]);
                        }
                    }
                }
            }
        }

        // ===== Store final dQ using TMA =====
        // Wait for all dQ accumulations to complete
        __syncthreads();
        
        if (warp_idx == 0 && elect_one_sync()) {
            // Copy dQ from TMEM to shared memory
            Tensor sdQ = make_tensor(make_smem_ptr(plan.u.dQ_out.data()), SmemLayoutdQ{});
            
            // Load dQ from TMEM
            float2 dq_vals[D_Q / 64];
            CUTE_UNROLL
            for (int row = 0; row < B_H; ++row) {
                ku::tmem_ld_32dp32bNx<D_Q/32>(tmem_cols::dQ, dq_vals);
                cutlass::arch::fence_view_async_tmem_load();
                
                // Convert to bf16 and store to shared memory
                CUTE_UNROLL
                for (int d = 0; d < D_Q / 64; ++d) {
                    nv_bfloat162 dq_bf16 = __float22bfloat162_rn(dq_vals[d]);
                    *(nv_bfloat162*)(plan.u.dQ_out.data() + row * D_Q + d * 2) = dq_bf16;
                }
            }
            
            fence_view_async_shared();
            
            // TMA store dQ to global memory
            cute::copy(tma_params.tma_dQ, sdQ, 
                make_tensor(make_gmem_ptr((bf16*)params.d_q + s_q_idx * params.stride_dq_s_q), 
                    make_layout(make_shape(Int<B_H>{}, Int<D_Q>{}), 
                        make_stride(params.stride_dq_h_q, _1{}))));
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
                bf16* sK_base = plan.dQ_cfg.k_buf[cur_buf].data() + warp_idx_local * 4 * 64;

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
                    make_smem_ptr(plan.dKV_cfg.q_buf.data()),
                    tile_to_shape(
                        UMMA::Layout_K_SW128_Atom<bf16>{},
                        Shape<Int<B_H*2>, Int<64>>{}
                    )
                )
            );

            UMMA::SmemDescriptor sQ_rope_desc = UMMA::make_umma_desc<UMMA::Major::K>(
                make_tensor(
                    make_smem_ptr(plan.dKV_cfg.q_rope_buf.data()),
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
                Tensor sQ = make_tensor(make_smem_ptr(plan.dKV_cfg.q_buf.data()), SmemLayoutQ{});
                Tensor sK = make_tensor(make_smem_ptr(plan.dQ_cfg.k_buf[cur_buf].data()), SmemLayoutK{});

                // Wait for K to be ready
                plan.bar_k_ready[cur_buf][0].arrive_and_expect_tx(B_TOPK * D_V/2 * sizeof(bf16));
                plan.bar_k_ready[cur_buf][0].wait((k/NUM_BUFS) & 1);
                ku::tcgen05_after_thread_sync();

                // P = Q @ K^T (NoPE part)
                ku::utcmma_ss(tiled_mma_P, sQ, sK, tP, true);
                
                // Wait for K RoPE part
                plan.bar_k_ready[cur_buf][1].arrive_and_expect_tx(B_TOPK * (D_K-D_V)/2 * sizeof(bf16));
                plan.bar_k_ready[cur_buf][1].wait((k/NUM_BUFS) & 1);
                ku::tcgen05_after_thread_sync();

                // P += Q_rope @ K_rope^T (RoPE part)
                if constexpr (HAVE_ROPE) {
                    Tensor sQ_rope = make_tensor(make_smem_ptr(plan.dKV_cfg.q_rope_buf.data()), SmemLayoutQRoPE{});
                    Tensor sK_rope = make_tensor(make_smem_ptr(plan.dQ_cfg.k_buf[cur_buf].data() + D_V * B_TOPK), SmemLayoutKRoPE{});
                    ku::utcmma_ss(tiled_mma_P, sQ_rope, sK_rope, tP, false);
                }

                plan.bar_p_computed[cur_buf].arrive();

                // ===== Step 2: Compute dP_mid = dO @ V^T =====
                Tensor sdO = make_tensor(make_smem_ptr(plan.dKV_cfg.do_buf.data()), SmemLayoutdO{});
                Tensor sV = make_tensor(make_smem_ptr(plan.dQ_cfg.k_buf[cur_buf].data()), SmemLayoutV{});

                // Wait for V (reusing K buffer space for V)
                plan.bar_v_ready[cur_buf][0].arrive_and_expect_tx(B_TOPK * D_V/2 * sizeof(bf16));
                plan.bar_v_ready[cur_buf][0].wait((k/NUM_BUFS) & 1);
                ku::tcgen05_after_thread_sync();

                // dP_mid = dO @ V^T (stored in P's TMEM location)
                ku::utcmma_ss(tiled_mma_dQ, sdO, sV, tP, true);  // Reuse P's TMEM for dP_mid

                plan.bar_dp_computed[cur_buf].arrive();

                // ===== Step 3: dQ += dS @ K (after softmax computes dS) =====
                plan.bar_so_ready.wait((k/NUM_BUFS) & 1);
                Tensor sS = make_tensor(make_smem_ptr(plan.s_q_rope.s), SmemLayoutP{});

                // dQ_nope += dS @ K_nope
                ku::utcmma_ss(tiled_mma_dQ, sS, sK, tdQ, k == 0);

                // dQ_rope += dS @ K_rope (FIX: Added RoPE gradient for dQ)
                if constexpr (HAVE_ROPE) {
                    Tensor sK_rope = make_tensor(make_smem_ptr(plan.dQ_cfg.k_buf[cur_buf].data() + D_V * B_TOPK), SmemLayoutKRoPE{});
                    ku::utcmma_ss(tiled_mma_dQ, sS, sK_rope, tdQ_rope, k == 0);
                }

                plan.bar_dq_accumulated[cur_buf].arrive();

                // ===== Step 4: dK = dS^T @ Q =====
                // dK_nope = dS^T @ Q_nope
                ku::utcmma_ts(tiled_mma_dK, sS, sQ, tdK, true);

                // dK_rope = dS^T @ Q_rope (FIX: Added RoPE gradient for dK)
                if constexpr (HAVE_ROPE) {
                    Tensor sQ_rope = make_tensor(make_smem_ptr(plan.dKV_cfg.q_rope_buf.data()), SmemLayoutQRoPE{});
                    ku::utcmma_ts(tiled_mma_dK, sS, sQ_rope, tdK_rope, true);
                }

                // ===== Step 5: dV = S^T @ dO =====
                // Note: S is stored in shared memory (same location as dS after softmax stores it)
                // We need to use the S values computed by warpgroup 0
                ku::utcmma_ts(tiled_mma_dV, sS, sdO, tdV, true);

                plan.bar_dkv_ready[cur_buf].arrive();
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
