#pragma once
#include "phase1.h"

#include <math_constants.h>
#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/arch.h>
#include <cutlass/cuda_host_adapter.hpp>

#include <kerutils/kerutils.cuh>

#include "params.h"
#include "utils.h"
#include "sm100/helpers.h"

#include "config.h"
#include "preprocess_delta.cuh"

namespace sm100::bwd::head128 {

using namespace cute;
 
 /*
Backward Pipeline Overview - Part0:
====================================

This kernel implements the first phase of backward propagation with three WarpGroups:

| WG0 (Copy)  |  WG1 (MMA P=QK^T)  |  WG2 (Delta)  |

WG0: Data Loading
  1. TMA prefetch load Q and dO
  2. Loop load KV data

WG1: P = QK^T Computation
  1. First matmul (NoPE part): P = Q[0:D_V] @ K[0:D_V]^T
     - Uses Implicit Dual GEMM optimization
     - Instruction: utcmma_ss (SMEM-SMEM)
     - Output to tmem_cols::P
     - First part requires clear_accum
  2. Second matmul (RoPE part): P += Q[D_V:D_Q] @ K[D_V:D_Q]^T
     - Instruction: utcmma_ss
     - Accumulation mode

WG2: Delta Loading
  1. Load precomputed delta from global memory to SMEM
  2. Delta is computed by preprocess kernel before main kernel launch
*/
 
/**
 * @brief Sparse Attention Backward Kernel Device Function - Part0 (2CTA Mode)
 * @tparam TmaParams TMA parameter type
 * @param params Attention computation parameters
 * @param tma_params TMA descriptor parameters
 * 
 * Thread Organization (2CTA Mode):
 * - Grid: [2*s_q, 1, 1] - 2 CTAs per query token
 * - Cluster: [2, 1, 1] - 2 CTAs form a cluster
 * - Block: 384 threads = 3 warpgroups (128 threads each)
 *   - Warpgroup 0: Data loading (TMA)
 *   - Warpgroup 1: QK^T computation (MMA)
 *   - Warpgroup 2: Delta loading (from precomputed delta)
 * 
 * 2CTA Cooperation:
 * - CTA0 processes Q[0:B_H/2, :] and KV[0:B_TOPK/2, :]
 * - CTA1 processes Q[B_H/2:B_H, :] and KV[B_TOPK/2:B_TOPK, :]
 * - Both CTAs share TMEM through 2x1SM MMA
 */
template<int D_QK>
template<typename TmaParams>
__device__ void
KernelTemplate<D_QK>::sparse_attn_bwd_kernel_devfunc(const SparseAttnBwdParams &params, const TmaParams &tma_params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
    const int cta_idx = blockIdx.x % 2;
    const int s_q_idx = blockIdx.x / 2;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = threadIdx.x % 32;
    const int topk_length = params.topk_length != nullptr ? __ldg(params.topk_length + s_q_idx) : params.topk;
    const int num_k_blocks = max(cute::ceil_div(topk_length, (int)B_TOPK), 1);  // num_k_blocks always >= 1
    const int warpgroup_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
    const int idx_in_warpgroup = threadIdx.x % 128;

    // Prefetch TMA descriptors
    if (threadIdx.x == 0) {
        cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dO.get_tma_descriptor());
        cute::prefetch_tma_descriptor(&(tma_params.tensor_map_kv));
    }

    // Define shared tensors
    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);
    Tensor sQ = make_tensor(make_smem_ptr(plan.u.q_kv.q.data()), SmemLayoutQ{});
    Tensor sdO = make_tensor(make_smem_ptr(plan.dO.data()), SmemLayoutdO{});

    int* gIndices = params.indices + s_q_idx*params.stride_indices_s_q; // [topk]
 
    // Allocate tmem tensors
    TiledMMA tiled_mma_P = TiledMMA_P{};
    Tensor tP = partition_fragment_C(tiled_mma_P, Shape<Int<B_H/2>, Int<B_TOPK>>{});
    tP.data().get() = tmem_cols::P;
 
    if (warp_idx == 0) {
        if (elect_one_sync()) {

            // --- Initialize barriers (double buffering support) ---
            plan.bar_prologue_q.init(1);
            plan.bar_prologue_dO.init(1);
            plan.bar_prologue_utccp.init(1);
            
            // Single buffer mode initialization (debug mode: disable double buffering)
            plan.bar_qk_done[0].init(1);
            plan.bar_kv_ready[0].init(1);
            plan.bar_p_free[0].init(128*2);         // 2CTA sync
            plan.bar_kv_valid_ready[0].init(B_TOPK/8);
            plan.bar_kv_valid_free[0].init(128);
            
            // WG0-WG2 synchronization barriers
            plan.bar_p_ready[0].init(128*2);        // WG2通知WG0 p已准备好 (2CTA sync)
            plan.bar_dp_ready[0].init(128*2);       // WG2通知WG0 dp已准备好 (2CTA sync)
            plan.bar_s_ready[0].init(128*2);        // WG0通知WG2 s已准备好 (2CTA sync)
            plan.bar_ds_ready[0].init(128*2);       // WG0通知WG2 ds已准备好 (2CTA sync)
            plan.bar_dq_ready[0].init(128*2);       // WG2通知WG1 dQ已准备好 (2CTA sync)
            plan.bar_dkv_ready[0].init(128*2);     // WG2通知WG1 dKV已准备好 (2CTA sync)
            
            fence_barrier_init();
        }
    }

    cute::cluster_sync();   // We must add a cluster_sync() here, or TMA from CTA1 may launch before barrier initialization in CTA0

    if (warp_idx == 0) {
        if (elect_one_sync()) {
            // Copy Q (including NoPE and RoPE)
            Tensor gQ = flat_divide(
                tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx),
                Tile<Int<B_H/2>>{}
            )(_, cta_idx, _);
            ku::launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_prologue_q, TMA::CacheHintSm90::EVICT_FIRST);

            // --- Load dO ---
            Tensor gdO = flat_divide(
                tma_params.tma_dO.get_tma_tensor(tma_params.shape_dO)(_, _, s_q_idx),
                Tile<Int<B_H/2>>{}
            )(_, cta_idx, _);
            ku::launch_tma_copy(tma_params.tma_dO, gdO, sdO, plan.bar_prologue_dO, TMA::CacheHintSm90::EVICT_FIRST);
        }

        // Initialize TMEM
        cute::TMEM::Allocator2Sm().allocate(512, plan.tmem_start_addr.data());
        TRAP_ONLY_DEVICE_ASSERT(plan.tmem_start_addr.data()[0] == 0);
        cute::TMEM::Allocator2Sm().release_allocation_lock();
    }

    __syncthreads();    // Wait for TMEM allocation

    // ========================================
    // Warpgroup 0: Softmax and dS Computation (WG0) - 2CTA mode
    // Responsibility: Compute softmax(P), load delta, compute ds, output dQ
    // ========================================
    if (warpgroup_idx == 0) {
        cutlass::arch::warpgroup_reg_alloc<144>();
        
        // Load LSE from global memory (needed for softmax computation)
        // Each thread loads LSE for one row (one head)
        // LSE shape: [s_q, h_q], stored row-major
        float lse = 0.0f;
        if (idx_in_warpgroup < B_H/2) {
            const int global_row_idx = cta_idx * (B_H/2) + idx_in_warpgroup;
            // Load LSE: params.lse[s_q_idx * h_q + global_row_idx]
            const float* gLSE = params.lse + s_q_idx * params.h_q + global_row_idx;
            lse = __ldg(gLSE);
        }
        
        const float scale = params.sm_scale_div_log2;
        Tensor sS = make_tensor(make_smem_ptr(plan.s_ds.s.data()), SmemLayoutS{});
        Tensor sDS = make_tensor(make_smem_ptr(plan.s_ds.ds.data()), SmemLayoutS{});
        
        // Allocate TMEM tensors for dQ (will be accumulated across iterations)
        TiledMMA tiled_mma_dQ = TiledMMA_dQ{};
        Tensor tdQ = partition_fragment_C(tiled_mma_dQ, Shape<Int<B_H/2>, Int<D_Q>>{});
        tdQ.data().get() = tmem_cols::dQ;
        
        // Initialize dQ accumulator to zero
        CUTE_UNROLL
        for (int i = 0; i < size(tdQ); ++i) {
            tdQ(i) = 0.0f;
        }
        
        CUTE_NO_UNROLL
        for (int k = 0; k < num_k_blocks; ++k) {
            constexpr int cur_buf = 0;
            
            // Step 1: Wait for WG2 to compute P
            plan.bar_p_ready[cur_buf].wait((k & 1) ? 1 : 0);
            ku::tcgen05_after_thread_sync();
            
            // Step 2: Load P from TMEM
            float2 p[(B_TOPK/2)/2];
            ku::tmem_ld_32dp32bNx<B_TOPK/2>(tmem_cols::P, p);
            cutlass::arch::fence_view_async_tmem_load();
            ku::tcgen05_before_thread_sync();
            
            // Step 3: Compute softmax(P) = exp2(P*scale - LSE)
            // Load LSE for this row (if available, otherwise compute from P)
            float row_lse = lse;  // Use preloaded LSE or compute from P
            if (row_lse == 0.0f) {
                // Compute LSE from P if not provided
                float max_p = -CUDART_INF_F;
                float* p_float = (float*)p;
                CUTE_UNROLL
                for (int i = 0; i < B_TOPK/2; ++i) {
                    max_p = max(max_p, p_float[i]);
                }
                max_p *= scale;
                
                float sum_exp = 0.0f;
                CUTE_UNROLL
                for (int i = 0; i < (B_TOPK/2)/2; ++i) {
                    float2 scaled_p = ku::float2_mul(p[i], make_float2(scale, scale));
                    scaled_p.x = exp2f(scaled_p.x - max_p);
                    scaled_p.y = exp2f(scaled_p.y - max_p);
                    sum_exp += scaled_p.x + scaled_p.y;
                }
                row_lse = log2f(sum_exp) + max_p;
            }
            
            // Compute softmax values: s = exp2(P*scale - LSE)
            __nv_bfloat162 s_fp32[(B_TOPK/2)/2];
            float2 neg_lse = make_float2(-row_lse, -row_lse);
            CUTE_UNROLL
            for (int i = 0; i < (B_TOPK/2)/2; ++i) {
                float2 scaled_p = ku::float2_fma(p[i], make_float2(scale, scale), neg_lse);
                scaled_p.x = exp2f(scaled_p.x);
                scaled_p.y = exp2f(scaled_p.y);
                s_fp32[i] = __float22bfloat162_rn(scaled_p);
            }
            
            // Step 4: Store s to SMEM (convert fp32 to bf16)
            uint128_t* sS_base = (uint128_t*)plan.s_ds.s.data() + idx_in_warpgroup%64 + 64*((idx_in_warpgroup/64)*8);
            CUTE_UNROLL
            for (int i = 0; i < B_TOPK/2/8; ++i) {
                sS_base[64*i] = *(uint128_t*)(s_fp32 + i*4);
            }
            fence_view_async_shared();
            
            // Notify WG2 that s is ready
            plan.bar_s_ready[cur_buf].arrive(0u);
            
            // Step 5: Load delta from global memory (per row)
            float delta = 0.0f;
            if (idx_in_warpgroup < B_H/2) {
                const int global_row_idx = cta_idx * (B_H/2) + idx_in_warpgroup;
                const float* gDelta = params.delta + s_q_idx * params.stride_delta_s_q + global_row_idx * params.stride_delta_h_q;
                delta = __ldg(gDelta);
            }
            
            // Step 6: Wait for WG2 to compute dp
            plan.bar_dp_ready[cur_buf].wait((k & 1) ? 1 : 0);
            ku::tcgen05_after_thread_sync();
            
            // Step 7: Load dp from TMEM
            float2 dp[(B_TOPK/2)/2];
            ku::tmem_ld_32dp32bNx<B_TOPK/2>(tmem_cols::dP, dp);
            cutlass::arch::fence_view_async_tmem_load();
            ku::tcgen05_before_thread_sync();
            
            // Step 8: Compute ds = s * (dp - delta) * scale
            // Note: Use fp32 format of s (s_fp32), not bf16 format stored in SMEM
            __nv_bfloat162 ds_fp32[(B_TOPK/2)/2];
            float2 delta_float2 = make_float2(delta, delta);
            float2 scale_float2 = make_float2(scale, scale);
            CUTE_UNROLL
            for (int i = 0; i < (B_TOPK/2)/2; ++i) {
                // Convert s_fp32 back to float2 for computation
                float2 s_val = __bfloat1622float2(s_fp32[i]);
                float2 dp_val = dp[i];
                float2 dp_minus_delta = ku::float2_sub(dp_val, delta_float2);
                float2 ds_val = ku::float2_mul(ku::float2_mul(s_val, dp_minus_delta), scale_float2);
                ds_fp32[i] = __float22bfloat162_rn(ds_val);
            }
            
            // Step 9: Store ds to SMEM (convert fp32 to bf16)
            uint128_t* sDS_base = (uint128_t*)plan.s_ds.ds.data() + idx_in_warpgroup%64 + 64*((idx_in_warpgroup/64)*8);
            CUTE_UNROLL
            for (int i = 0; i < B_TOPK/2/8; ++i) {
                sDS_base[64*i] = *(uint128_t*)(ds_fp32 + i*4);
            }
            fence_view_async_shared();
            
            // Notify WG2 that ds is ready
            plan.bar_ds_ready[cur_buf].arrive(0u);
        }
        
        // Step 10: After all loops, output dQ using TMA
        // Wait for final dQ accumulation
        if (num_k_blocks > 0) {
            plan.bar_dq_ready[0].wait(((num_k_blocks-1) & 1) ? 1 : 0);
        }
        ku::tcgen05_after_thread_sync();
        
        // Copy dQ from TMEM to SMEM, then output via TMA
        Tensor sdQ = make_tensor(make_smem_ptr(plan.u.dq.dq.data()), SmemLayoutdQ{});
        
        // Copy dQ from TMEM to SMEM in chunks
        constexpr int CHUNK_SIZE = 64;
        CUTE_UNROLL
        for (int chunk_idx = 0; chunk_idx < D_Q / CHUNK_SIZE; ++chunk_idx) {
            float2 dq_chunk[CHUNK_SIZE/2];
            ku::tmem_ld_32dp32bNx<CHUNK_SIZE>(tmem_cols::dQ + chunk_idx * CHUNK_SIZE, dq_chunk);
            cutlass::arch::fence_view_async_tmem_load();
            ku::tcgen05_before_thread_sync();
            
            // Convert fp32 to bf16 and store to SMEM
            __nv_bfloat162 dq_bf16[CHUNK_SIZE/2];
            CUTE_UNROLL
            for (int i = 0; i < CHUNK_SIZE/2; ++i) {
                dq_bf16[i] = __float22bfloat162_rn(dq_chunk[i]);
            }
            
            // Store to SMEM
            uint128_t* sdQ_base = (uint128_t*)plan.u.dq.dq.data() + idx_in_warpgroup%64 + 64*((idx_in_warpgroup/64)*8) + chunk_idx*(CHUNK_SIZE/8);
            CUTE_UNROLL
            for (int i = 0; i < CHUNK_SIZE/8; ++i) {
                sdQ_base[64*i] = *(uint128_t*)(dq_bf16 + i*4);
            }
            fence_view_async_shared();
        }
        
        // Output dQ via TMA (will be done in warp 0)
        if (warp_idx == 0 && elect_one_sync()) {
            // Wait for all dQ data to be ready in SMEM
            __syncthreads();
            
            // Output dQ using TMA
            Tensor gdQ = flat_divide(
                tma_params.tma_dQ.get_tma_tensor(tma_params.shape_dQ)(_, _, s_q_idx),
                Tile<Int<B_H/2>>{}
            )(_, cta_idx, _);
            ku::launch_tma_copy(tma_params.tma_dQ, sdQ, gdQ, plan.bar_prologue_q, TMA::CacheHintSm90::EVICT_FIRST);
        }
        
        if (warp_idx == 0) {
            cute::TMEM::Allocator2Sm().free(0, 512);
        }

    // ========================================
    // Warpgroup 1: KV Loading and dKV Output (WG1) - 2CTA mode
    // Responsibility: Loop load KV, output dKV using atomic operations
    // ========================================
    } else if (warpgroup_idx == 1) {
        cutlass::arch::warpgroup_reg_alloc<144>();
        int local_warp_idx = cutlass::canonical_warp_idx_sync();
        constexpr int NUM_WARPS = 4;
        constexpr int NUM_LOCAL_ROWS_PER_WARP = (B_TOPK/2) / 4 / NUM_WARPS;

        if (elect_one_sync()) {
            bf16* sKV_base = plan.u.q_kv.kv.data() + local_warp_idx*4*64;

            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks; ++k) {
                constexpr int cur_buf = 0;
                
                // Step 1: Wait for WG2 to compute dQ (for k > 0, wait for previous iteration's dQ)
                if (k > 0) {
                    plan.bar_dq_ready[cur_buf].wait(((k-1) & 1) ? 1 : 0);
                }
                
                // Step 2: Load TopK indices and KV data
                int4 indices[NUM_LOCAL_ROWS_PER_WARP];
                int max_indices = -1, min_indices = params.s_kv;
                CUTE_UNROLL
                for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                    indices[local_row] = __ldg((int4*)(gIndices + k*B_TOPK + cta_idx*(B_TOPK/2)) + local_row*NUM_WARPS + local_warp_idx);
                    max_indices = max(max_indices, int4_max(indices[local_row]));
                    min_indices = min(min_indices, int4_min(indices[local_row]));
                }
                bool is_all_rows_invalid = min_indices == params.s_kv || max_indices == -1;
                bool should_skip_tma = is_all_rows_invalid && k >= 1;

                // Load KV data using TMA gather
                auto load_kv = [&](transac_bar_t &bar) {
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                        CUTE_UNROLL
                        for (int local_col = 0; local_col < D_Q/64; ++local_col) {
                            ku::tma_gather4_cta_group_2<true>(
                                &(tma_params.tensor_map_kv),
                                bar,
                                sKV_base + local_row*(4*NUM_WARPS)*64 + local_col*((B_TOPK/2)*64),
                                local_col*64,
                                indices[local_row],
                                (int64_t)TMA::CacheHintSm90::EVICT_LAST
                            );
                        }
                    }
                };

                if (!should_skip_tma) {
                    load_kv(plan.bar_kv_ready[cur_buf]);
                } else {
                    plan.bar_kv_ready[cur_buf].complete_transaction(0u, NUM_LOCAL_ROWS_PER_WARP*4*D_Q*sizeof(bf16), 1u);
                }
                
                // Step 3: Wait for WG2 to compute dKV, then output using atomic operations
                if (k > 0) {
                    plan.bar_dkv_ready[cur_buf].wait(((k-1) & 1) ? 1 : 0);
                    ku::tcgen05_after_thread_sync();
                    
                    // Load dKV from TMEM and output using atom.add.v4
                    // dKV shape: [B_TOPK/2, D_Q] = [32, 576]
                    // Process in chunks of 4 floats (v4)
                    constexpr int CHUNK_SIZE = 4;
                    constexpr int NUM_CHUNKS_PER_ROW = D_Q / CHUNK_SIZE;
                    
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                        const int global_row_idx_in_block = local_row * NUM_WARPS + local_warp_idx;
                        if (global_row_idx_in_block >= B_TOPK/2) continue;
                        
                        const int kv_idx_in_topk = (k-1) * B_TOPK + cta_idx * (B_TOPK/2) + global_row_idx_in_block;
                        const int actual_kv_idx = __ldg(gIndices + s_q_idx * params.stride_indices_s_q + kv_idx_in_topk);
                        
                        if (actual_kv_idx < 0 || actual_kv_idx >= params.s_kv) continue;
                        
                        // Load dKV from TMEM (NoPE part: D_V = 512)
                        CUTE_UNROLL
                        for (int chunk_idx = 0; chunk_idx < D_V / CHUNK_SIZE; ++chunk_idx) {
                            float4 dkv_chunk;
                            ku::tmem_ld_32dp32bNx<CHUNK_SIZE>(
                                tmem_cols::dKV + chunk_idx * CHUNK_SIZE,
                                &dkv_chunk
                            );
                            cutlass::arch::fence_view_async_tmem_load();
                            
                            // Atomic add to global memory using atom.add.v4
                            float* gdKV_ptr = (float*)(params.dKV) + actual_kv_idx * params.stride_dKV_s_kv + chunk_idx * CHUNK_SIZE;
                            asm volatile(
                                "red.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
                                :
                                : "l"(gdKV_ptr), "f"(dkv_chunk.x), "f"(dkv_chunk.y), "f"(dkv_chunk.z), "f"(dkv_chunk.w)
                                : "memory"
                            );
                        }
                        
                        // Load and output dKV RoPE part (D_ROPE = 64)
                        CUTE_UNROLL
                        for (int chunk_idx = 0; chunk_idx < D_ROPE / CHUNK_SIZE; ++chunk_idx) {
                            float4 dkv_rope_chunk;
                            ku::tmem_ld_32dp32bNx<CHUNK_SIZE>(
                                tmem_cols::dKV_RoPE + chunk_idx * CHUNK_SIZE,
                                &dkv_rope_chunk
                            );
                            cutlass::arch::fence_view_async_tmem_load();
                            
                            float* gdKV_rope_ptr = (float*)(params.dKV) + actual_kv_idx * params.stride_dKV_s_kv + D_V + chunk_idx * CHUNK_SIZE;
                            asm volatile(
                                "red.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
                                :
                                : "l"(gdKV_rope_ptr), "f"(dkv_rope_chunk.x), "f"(dkv_rope_chunk.y), "f"(dkv_rope_chunk.z), "f"(dkv_rope_chunk.w)
                                : "memory"
                            );
                        }
                    }
                    ku::tcgen05_before_thread_sync();
                }
            }
        }

    // ========================================
    // Warpgroup 2: Matrix Multiplication (WG2) - 2CTA mode
    // Responsibility: Compute QK^T, dp, dKV, dQ using utcmma_ss
    // ========================================
    } else if (warpgroup_idx == 2) {
        cutlass::arch::warpgroup_reg_alloc<168>();
        
        // Wait for Q and dO to be loaded
        if (cta_idx == 0 && warp_idx == 8 && elect_one_sync()) {
            plan.bar_prologue_q.arrive_and_expect_tx((B_H)*D_Q*sizeof(bf16));
            plan.bar_prologue_q.wait(0);
            plan.bar_prologue_dO.arrive_and_expect_tx((B_H)*D_V*sizeof(bf16));
            plan.bar_prologue_dO.wait(0);
            ku::tcgen05_after_thread_sync();
        }
        
        // Allocate TMEM tensors
        TiledMMA tiled_mma_P = TiledMMA_P{};
        TiledMMA tiled_mma_dP = TiledMMA_dP{};
        TiledMMA tiled_mma_dKV_part1 = TiledMMA_dKV_part1{};
        TiledMMA tiled_mma_dQ = TiledMMA_dQ{};
        TiledMMA tiled_mma_dKV_part2 = TiledMMA_dKV_part2{};
        
        Tensor tP = partition_fragment_C(tiled_mma_P, Shape<Int<B_H/2>, Int<B_TOPK>>{});
        Tensor tdP = partition_fragment_C(tiled_mma_dP, Shape<Int<B_H/2>, Int<B_TOPK>>{});
        Tensor tdKV = partition_fragment_C(tiled_mma_dKV_part1, Shape<Int<B_TOPK/2>, Int<D_V>>{});
        Tensor tdQ = partition_fragment_C(tiled_mma_dQ, Shape<Int<B_H/2>, Int<D_Q>>{});
        Tensor tdKV_part2 = partition_fragment_C(tiled_mma_dKV_part2, Shape<Int<B_TOPK/2>, Int<D_Q>>{});
        
        tP.data().get() = tmem_cols::P;
        tdP.data().get() = tmem_cols::dP;
        tdKV.data().get() = tmem_cols::dKV;
        tdQ.data().get() = tmem_cols::dQ;
        tdKV_part2.data().get() = tmem_cols::dKV;
        
        // Initialize dQ accumulator to zero
        CUTE_UNROLL
        for (int i = 0; i < size(tdQ); ++i) {
            tdQ(i) = 0.0f;
        }
        
        // Main computation loop
        if (cta_idx == 0 && warp_idx == 8 && elect_one_sync()) {
            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks; ++k) {
                constexpr int cur_buf = 0;
                
                // Build SMEM tensor views
                Tensor sQ = make_tensor(make_smem_ptr(plan.u.q_kv.q.data()), SmemLayoutQ{});
                Tensor sKV = make_tensor(make_smem_ptr(plan.u.q_kv.kv.data()), SmemLayoutKV{});
                Tensor sdO = make_tensor(make_smem_ptr(plan.dO.data()), SmemLayoutdO{});
                Tensor sS = make_tensor(make_smem_ptr(plan.s_ds.s.data()), SmemLayoutS{});
                Tensor sDS = make_tensor(make_smem_ptr(plan.s_ds.ds.data()), SmemLayoutS{});
                
                // Extract V from KV (V is the NoPE part: [B_TOPK/2, D_V])
                Tensor sV = make_tensor(
                    make_smem_ptr(plan.u.q_kv.kv.data()),
                    SmemLayoutKVNoPE_TiledMMA{}
                );
                
                // Wait for KV to be ready
                plan.bar_kv_ready[cur_buf].arrive_and_expect_tx((B_TOPK)*D_Q*sizeof(bf16));
                plan.bar_kv_ready[cur_buf].wait(k & 1);
                ku::tcgen05_after_thread_sync();
                
                // Step 1: Compute P = Q @ K^T
                ku::utcmma_ss(tiled_mma_P, sQ, sKV, tP, true);  // clear_accum = true
                ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_p_ready[cur_buf], 1|2);
                
                // Step 2: Wait for WG0 to compute s, then compute dp = dO @ V^T
                plan.bar_s_ready[cur_buf].wait((k & 1) ? 1 : 0);
                ku::tcgen05_after_thread_sync();
                
                ku::utcmma_ss(tiled_mma_dP, sdO, sV, tdP, true);  // clear_accum = true
                ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_dp_ready[cur_buf], 1|2);
                
                // Step 3: Compute dKV part1 = s^T @ dO
                ku::utcmma_ss(tiled_mma_dKV_part1, sS, sdO, tdKV, true);  // clear_accum = true
                
                // Step 4: Wait for WG0 to compute ds, then compute dQ = ds @ KV^T (accumulate)
                plan.bar_ds_ready[cur_buf].wait((k & 1) ? 1 : 0);
                ku::tcgen05_after_thread_sync();
                
                ku::utcmma_ss(tiled_mma_dQ, sDS, sKV, tdQ, false);  // clear_accum = false (accumulate)
                ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_dq_ready[cur_buf], 1|2);
                
                // Step 5: Compute dKV part2 = ds^T @ Q (accumulate)
                ku::utcmma_ss(tiled_mma_dKV_part2, sDS, sQ, tdKV_part2, false);  // clear_accum = false (accumulate)
                ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_dkv_ready[cur_buf], 1|2);
            }
        }
    }
 
#else
    // Error handling for non-SM100 architectures
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm100");
    }
#endif
}

/**
 * @brief Global Kernel Wrapper Function (consistent structure with forward propagation)
 * @tparam Kernel KernelTemplate instantiation type
 * @tparam TmaParams TMA parameter type
 * @param params Attention computation parameters
 * @param tma_params TMA descriptor parameters
 */
template<typename Kernel, typename TmaParams>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, 2)
sparse_attn_bwd_kernel(__grid_constant__ const SparseAttnBwdParams params, 
                              __grid_constant__ const TmaParams tma_params) {
    Kernel::sparse_attn_bwd_kernel_devfunc(params, tma_params);
}

/**
 * @brief Host wrapper function to launch backward Phase1 kernel (2CTA Mode)
 * @tparam D_QK Query/Key dimension (576)
 * @param params Attention computation parameter struct
 * 
 * Functionality:
 * 1. Parameter validation
 * 2. Create TMA descriptors (Q, dO, O, KV) with 2SM support
 * 3. Configure and launch CUDA kernel with cluster
 * 
 * 2CTA Mode:
 * - Grid: [2*s_q, 1, 1] - 2 CTAs per query token
 * - Cluster: [2, 1, 1] - 2 CTAs form a cluster for cooperative computation
 */
template<int D_QK>
void run_bwd_phase1_kernel(const SparseAttnBwdParams& params) {
    static_assert(D_QK == 576);  // Only support D_QK == 576 for backward kernel
    using Kernel = KernelTemplate<D_QK>;

    // === Parameter validation ===
    KU_ASSERT(params.h_kv == 1);                    // KV head count must be 1 (MLA feature)
    KU_ASSERT(params.topk % Kernel::B_TOPK == 0);   // TopK must be multiple of B_TOPK (skip boundary check)
    KU_ASSERT(params.h_q == Kernel::B_H);           // Query head count must equal B_H
    KU_ASSERT(params.d_qk == D_QK);

    // === Launch preprocessing kernel to compute delta ===
    // Delta must be computed before main kernel launch
    run_bwd_preprocess_delta_kernel<D_QK>(params);

    // === Create TMA descriptor for Q (2SM mode) ===
    // Q shape: [h_q, d_qk, s_q] = [h_q, 576, s_q] (including NoPE 512 + RoPE 64)
    auto shape_Q = make_shape(params.h_q, Kernel::D_Q, params.s_q);
    auto tma_Q = cute::make_tma_copy(
        SM100_TMA_2SM_LOAD_NOSPLIT{},  // 2SM TMA for cluster
        make_tensor(
            make_gmem_ptr((bf16*)params.q),
            make_layout(
                shape_Q,
                make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q)
            )
        ),
        (typename Kernel::SmemLayoutQ){}
    );

    // === Create TMA descriptor for dO (2SM mode) ===
    // dO shape: [h_q, d_v, s_q]
    auto shape_dO = make_shape(params.h_q, params.d_v, params.s_q);
    auto tma_dO = cute::make_tma_copy(
        SM100_TMA_2SM_LOAD_NOSPLIT{},  // 2SM TMA for cluster
        make_tensor(
            make_gmem_ptr((bf16*)params.dO),
            make_layout(
                shape_dO,
                make_stride(params.stride_dO_h_q, _1{}, params.stride_dO_s_q)
            )
        ),
        (typename Kernel::SmemLayoutdO){}
    );

    // === Create TMA descriptor for dQ output (2SM mode) ===
    // dQ shape: [h_q, d_qk, s_q] = [h_q, 576, s_q]
    auto shape_dQ = make_shape(params.h_q, Kernel::D_Q, params.s_q);
    auto tma_dQ = cute::make_tma_copy(
        SM100_TMA_2SM_STORE_NOSPLIT{},  // 2SM TMA for cluster output
        make_tensor(
            make_gmem_ptr((bf16*)params.dQ),
            make_layout(
                shape_dQ,
                make_stride(params.stride_dQ_h_q, _1{}, params.stride_dQ_s_q)
            )
        ),
        (typename Kernel::SmemLayoutdQ){}
    );

    // === Create TensorMap for KV (for TMA gather) ===
    // KV shape: [d_qk, s_kv] = [576, s_kv] (including NoPE 512 + RoPE 64)
    CUtensorMap tensor_map_kv;
    {
        uint64_t size[2] = {Kernel::D_Q, (unsigned long)params.s_kv};  // [d_qk, s_kv]
        uint64_t stride[1] = {params.stride_kv_s_kv*sizeof(bf16)};  // Row stride
        uint32_t box_size[2] = {64, 1};    // Box size per load: 64 cols x 1 row
        uint32_t elem_stride[2] = {1, 1};  // Element stride
        CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &tensor_map_kv,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,                                                      // 2D tensor
            params.kv,                                              // Global memory pointer
            size,
            stride,
            box_size,
            elem_stride,
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,   // No interleave
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,         // 128-byte swizzle
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,  // L2 cache promotion
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE   // No OOB fill
        );
        KU_ASSERT(res == CUresult::CUDA_SUCCESS);
    }

    // === Assemble TMA parameter struct ===
    TmaParams<
        decltype(shape_Q), decltype(tma_Q),
        decltype(shape_dO), decltype(tma_dO),
        decltype(shape_dQ), decltype(tma_dQ)
    > tma_params = {
        shape_Q, tma_Q,
        shape_dO, tma_dO,
        shape_dQ, tma_dQ,
        tensor_map_kv
    };
    
    // Use kernel instantiated from KernelTemplate (consistent with forward propagation)
    auto kernel = &sparse_attn_bwd_kernel<Kernel, decltype(tma_params)>;

    // === Configure and launch kernel with cluster (2CTA mode) ===
    constexpr size_t smem_size = sizeof(typename Kernel::SharedMemoryPlan);  // Dynamic shared memory size
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    // 2CTA Cluster Launch:
    // - Grid: 2*s_q blocks (2 CTAs per query token)
    // - Block: NUM_THREADS (384) threads
    // - Cluster: (2, 1, 1) - 2 CTAs form a cluster
    cutlass::ClusterLaunchParams launch_params = {
        dim3(2*params.s_q, 1, 1),       // Grid: 2 CTAs per query
        dim3(Kernel::NUM_THREADS, 1, 1),// Block: 384 threads
        dim3(2, 1, 1),                  // Cluster: 2 CTAs
        smem_size,                      // Dynamic shared memory
        params.stream                   // CUDA stream
    };
    KU_CUTLASS_CHECK(cutlass::launch_kernel_on_cluster(
        launch_params, (void*)kernel, params, tma_params
    ));
}

}  // namespace sm100::bwd::head128