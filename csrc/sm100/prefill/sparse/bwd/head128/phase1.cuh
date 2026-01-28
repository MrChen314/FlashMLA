/**
 * @file phase1.cuh
 * @brief SM100 Sparse Attention Backward Kernel Implementation (head_dim=64) - Part0
 * 
 * This file implements sparse MLA attention backward computation optimized for
 * NVIDIA Blackwell (SM100) architecture with 2CTA (cluster) support.
 * Part0 contains the following computation flow:
 *   - WG0: Data Loading (TMA prefetch Q/dO, loop load KV)
 *   - WG1: QK^T Matrix Multiplication (NoPE + RoPE parts)
 *   - WG2: Delta Computation (delta = sum(O * dO))
 * 
 * 2CTA Mode:
 *   - 2 CTAs form a cluster, sharing TMEM for cooperative MMA
 *   - Each CTA processes B_H/2 rows of Q and B_TOPK/2 rows of KV
 *   - Uses SM100_TMA_2SM_LOAD_NOSPLIT for TMA loading
 *   - Uses 2x1SM MMA instructions for cooperative computation
 */

 #pragma once

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
   1. First matmul (NoPE part): P = Q_nope @ K_nope^T
      - Uses Implicit Dual GEMM optimization
      - Instruction: utcmma_ss (SMEM-SMEM)
      - Output to tmem_cols::P
      - First part requires clear_accum
   2. Second matmul (RoPE part): P += Q_rope @ K_rope^T
      - Instruction: utcmma_ss
      - Accumulation mode
 
 WG2: Delta Computation
   1. Read O from global memory
   2. Read dO from SMEM
   3. Compute delta = sum(O * dO, dim=-1)
 */
 
/**
 * @brief Sparse Attention Backward Kernel - Part0 (2CTA Mode)
 * @tparam HAVE_ROPE Whether RoPE positional encoding is included
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
 *   - Warpgroup 2: Delta computation
 * 
 * 2CTA Cooperation:
 * - CTA0 processes Q[0:B_H/2, :] and KV[0:B_TOPK/2, :]
 * - CTA1 processes Q[B_H/2:B_H, :] and KV[B_TOPK/2:B_TOPK, :]
 * - Both CTAs share TMEM through 2x1SM MMA
 */
template<bool HAVE_ROPE, typename TmaParams>
__global__ void __launch_bounds__(NUM_THREADS, 1, 2)
sparse_attn_bwd_kernel_part0(__grid_constant__ const SparseAttnBwdParams params, 
                              __grid_constant__ const TmaParams tma_params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
    
    // === 2CTA Index Computation ===
    const int cta_idx = blockIdx.x % 2;                                 // CTA index (0 or 1)
    const int s_q_idx = blockIdx.x / 2;                                 // Current query index
    const int warp_idx = cutlass::canonical_warp_idx_sync();            // Warp index (0-11)
    const int lane_idx = threadIdx.x % 32;                              // Lane index (0-31)
    const int warpgroup_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);  // Warpgroup index (0-2)
    const int idx_in_warpgroup = threadIdx.x % 128;                     // Thread index within warpgroup
    const int topk_length = params.topk_length != nullptr ? __ldg(params.topk_length + s_q_idx) : params.topk;
    const int num_k_blocks = max(cute::ceil_div(topk_length, (int)B_TOPK), 1);  // In 2CTA mode B_TOPK=64
 
     // === Shared memory setup ===
     extern __shared__ char wksp_buf[];
     SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);
 
     int* gIndices = params.indices + s_q_idx * params.stride_indices_s_q;
 
    // === TMEM tensor allocation (2CTA mode, B_H=128) ===
    TiledMMA tiled_mma_P = TiledMMA_P{};
    
    // P matrix fragment (attention scores): [B_H/2, B_TOPK] = [64, 64]
    // In 2CTA mode each CTA processes B_H/2=64 rows, full B_TOPK=64 columns
    Tensor tP = partition_fragment_C(tiled_mma_P, Shape<Int<B_H/2>, Int<B_TOPK>>{});
    tP.data().get() = tmem_cols::P;
 
    // === Warp 0: Prologue - Initialize barriers and load Q/dO (2CTA mode) ===
    if (warp_idx == 0) {
        if (elect_one_sync()) {
            // --- Prefetch TMA descriptors ---
            if constexpr (HAVE_ROPE) {
                cute::prefetch_tma_descriptor(tma_params.tma_Q_rope.get_tma_descriptor());
            }
            cute::prefetch_tma_descriptor(tma_params.tma_Q_nope.get_tma_descriptor());
            cute::prefetch_tma_descriptor(tma_params.tma_dO.get_tma_descriptor());

            // --- Initialize barriers (double buffering support) ---
            plan.bar_prologue_q_nope.init(1);
            plan.bar_prologue_q_rope.init(1);
            plan.bar_prologue_dO.init(1);
            plan.bar_prologue_utccp.init(1);
            
            // 单缓冲模式初始化 (调试模式：禁用双缓冲)
            plan.bar_qk_done[0].init(1);
            plan.bar_sv_part_done[0].init(1);
            plan.bar_sv_done[0].init(1);
            plan.bar_kv_nope_ready[0].init(1);
            plan.bar_kv_rope_ready[0].init(1);
            plan.bar_p_free[0].init(128*2);         // 2CTA sync
            plan.bar_so_ready[0].init(128*2);       // 2CTA sync
            plan.bar_kv_valid_ready[0].init(B_TOPK/8);
            plan.bar_kv_valid_free[0].init(128);
            
            fence_barrier_init();
        }
    }

    // 2CTA sync: ensure all barriers are initialized
    cute::cluster_sync();

    if (warp_idx == 0) {
        if (elect_one_sync()) {
            // --- Load Q matrix using 2SM TMA ---
            // Q NoPE: In 2CTA mode use flat_divide to select the corresponding part by cta_idx
            Tensor gQ_nope = flat_divide(
                tma_params.tma_Q_nope.get_tma_tensor(tma_params.shape_Q_nope)(_, _, s_q_idx),
                Tile<Int<B_H/2>>{}
            )(_, cta_idx, _);
            Tensor sQ_nope = make_tensor(make_smem_ptr(plan.u.q_kv.q_nope.data()), SmemLayoutQNoPE{});
            ku::launch_tma_copy(tma_params.tma_Q_nope, gQ_nope, sQ_nope, plan.bar_prologue_q_nope, TMA::CacheHintSm90::EVICT_FIRST);

            // Q RoPE
            if constexpr (HAVE_ROPE) {
                Tensor gQ_rope = flat_divide(
                    tma_params.tma_Q_rope.get_tma_tensor(tma_params.shape_Q_rope)(_, _, s_q_idx),
                    Tile<Int<B_H/2>>{}
                )(_, cta_idx, _);
                Tensor sQ_rope = make_tensor(make_smem_ptr(plan.u.q_kv.q_rope.data()), SmemLayoutQRoPE{});
                ku::launch_tma_copy(tma_params.tma_Q_rope, gQ_rope, sQ_rope, plan.bar_prologue_q_rope, TMA::CacheHintSm90::EVICT_FIRST);
            }

            // --- Load dO ---
            Tensor gdO = flat_divide(
                tma_params.tma_dO.get_tma_tensor(tma_params.shape_dO)(_, _, s_q_idx),
                Tile<Int<B_H/2>>{}
            )(_, cta_idx, _);
            Tensor sdO = make_tensor(make_smem_ptr(plan.dO.data()), SmemLayoutdO{});
            ku::launch_tma_copy(tma_params.tma_dO, gdO, sdO, plan.bar_prologue_dO, TMA::CacheHintSm90::EVICT_FIRST);

            // Prefetch KV TMA descriptor
            cute::prefetch_tma_descriptor(&(tma_params.tensor_map_kv_nope));
        }

        // --- Allocate TMEM (using 2SM Allocator) ---
        cute::TMEM::Allocator2Sm().allocate(512, plan.tmem_start_addr.data());
        TRAP_ONLY_DEVICE_ASSERT(plan.tmem_start_addr.data()[0] == 0);
        cute::TMEM::Allocator2Sm().release_allocation_lock();
    }

    __syncthreads();
 
    // ========================================
    // Warpgroup 0: Data Loading (WG0) - 2CTA mode
    // Responsibility: TMA prefetch Q/dO, loop load KV (单缓冲模式，方便调试)
    // Each CTA loads B_TOPK/2 rows of KV data
    // ========================================
    if (warpgroup_idx == 0) {
        int local_warp_idx = cutlass::canonical_warp_idx_sync();
        constexpr int NUM_WARPS = 4;
        // 2CTA mode: each CTA loads B_TOPK/2 rows, 4 elements per row, 4 warps total
        constexpr int NUM_LOCAL_ROWS_PER_WARP = (B_TOPK/2) / 4 / NUM_WARPS;

        if (elect_one_sync()) {
            bf16* sKV_nope_base = plan.u.q_kv.kv_nope.data() + local_warp_idx*4*64;

            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks; ++k) {
                // --- Load TopK indices (2CTA mode: each CTA loads corresponding B_TOPK/2 rows) ---
                int4 indices[NUM_LOCAL_ROWS_PER_WARP];
                int max_indices = -1, min_indices = params.s_kv;
                CUTE_UNROLL
                for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                    // cta_idx * (B_TOPK/2) offset to corresponding CTA's data
                    indices[local_row] = __ldg((int4*)(gIndices + k*B_TOPK + cta_idx*(B_TOPK/2)) + local_row*NUM_WARPS + local_warp_idx);
                    max_indices = max(max_indices, int4_max(indices[local_row]));
                    min_indices = min(min_indices, int4_min(indices[local_row]));
                }
                bool is_all_rows_invalid = min_indices == params.s_kv || max_indices == -1;
                // 单缓冲模式：k >= 1 时可以跳过无效行
                bool should_skip_tma = is_all_rows_invalid && k >= 1;

                // 单缓冲模式：始终使用 buffer 0
                constexpr int cur_buf = 0;

                // --- Wait for previous QK^T to complete (单缓冲模式) ---
                if (k > 0) {
                    plan.bar_qk_done[0].wait((k-1) & 1);
                }

                // --- Load full KV NoPE data using TMA gather (2CTA) ---
                // Not using Dual GEMM, load full D_V=512 columns at once
                auto load_kv_nope = [&](transac_bar_t &bar) {
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                        CUTE_UNROLL
                        for (int local_col = 0; local_col < D_V/64; ++local_col) {
                            ku::tma_gather4_cta_group_2<true>(
                                &(tma_params.tensor_map_kv_nope),
                                bar,
                                sKV_nope_base + local_row*(4*NUM_WARPS)*64 + local_col*((B_TOPK/2)*64),
                                local_col*64,
                                indices[local_row],
                                (int64_t)TMA::CacheHintSm90::EVICT_LAST
                            );
                        }
                    }
                };

                // 单缓冲模式：始终使用 bar_kv_nope_ready[0]
                if (!should_skip_tma) {
                    load_kv_nope(plan.bar_kv_nope_ready[0]);
                } else {
                    plan.bar_kv_nope_ready[0].complete_transaction(0u, NUM_LOCAL_ROWS_PER_WARP*4*D_V*sizeof(bf16), 1u);
                }
            }
        }

    // ========================================
    // Warpgroup 1: QK^T Computation (WG1) - 2CTA mode
    // Responsibility: Execute P = Q @ K^T (NoPE + RoPE) with 2x1SM MMA
    // ========================================
    } else if (warpgroup_idx == 1) {
        // === CTA0 Warp 4: MMA control warp (only CTA0 initiates MMA in 2CTA mode) ===
        if (cta_idx == 0 && warp_idx == 4 && elect_one_sync()) {
            // --- Build SMEM tensor views (2CTA mode, B_H=128) ---
            // Q NoPE: [B_H/2, D_V] = [64, 512]
            Tensor sQ_nope = make_tensor(
                make_smem_ptr(plan.u.q_kv.q_nope.data()),
                SmemLayoutQNoPE_TiledMMA{}
            );
            
            // Q RoPE: [B_H, D_ROPE/2] = [128, 32] (2CTA merged)
            Tensor sQ_rope = make_tensor(
                make_smem_ptr(plan.u.q_kv.q_rope.data()),
                SmemLayoutQRoPE_TiledMMA{}
            );

            // KV NoPE: [B_TOPK/2, D_V] = [32, 512]
            Tensor sKV_nope = make_tensor(
                make_smem_ptr(plan.u.q_kv.kv_nope.data()),
                SmemLayoutKVNoPE_TiledMMA{}
            );
            
            // KV RoPE: [B_TOPK, D_ROPE/2] = [64, 32]
            Tensor sKV_rope = make_tensor(
                make_smem_ptr(plan.u.q_kv.kv_rope.data()),
                SmemLayoutKVRoPE_TiledMMA{}
            );

            // Wait for Q load to complete (2CTA: each CTA loads B_H/2 rows)
            plan.bar_prologue_q_nope.arrive_and_expect_tx((B_H/2)*D_V*sizeof(bf16));
            plan.bar_prologue_q_nope.wait(0);
            if constexpr (HAVE_ROPE) {
                plan.bar_prologue_q_rope.arrive_and_expect_tx((B_H/2)*D_ROPE*sizeof(bf16));
                plan.bar_prologue_q_rope.wait(0);
            }
            ku::tcgen05_after_thread_sync();

            // --- Main computation loop (单缓冲模式，方便调试) ---
            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks; ++k) {
                // 单缓冲模式：始终使用 buffer 0
                constexpr int cur_buf = 0;

                // Wait for P matrix buffer to be available (2CTA sync，单缓冲模式)
                if (k > 0) {
                    plan.bar_p_free[0].wait((k-1) & 1);
                }
                ku::tcgen05_after_thread_sync();

                // ================================================================
                // P = Q_nope @ K_nope^T (2CTA, not using Dual GEMM)
                // Q_nope: [64, 512], K_nope: [32, 512]
                // 2CTA cooperation: P: [B_H/2, B_TOPK] = [64, 64] (both CTAs complete together)
                // ================================================================
                
                // Wait for KV NoPE data to be ready (单缓冲模式)
                plan.bar_kv_nope_ready[0].arrive_and_expect_tx((B_TOPK/2)*D_V*sizeof(bf16));
                plan.bar_kv_nope_ready[0].wait(k & 1);
                ku::tcgen05_after_thread_sync();

                // P = Q_nope @ K_nope^T
                ku::utcmma_ss(
                    tiled_mma_P,
                    sQ_nope,
                    sKV_nope,
                    tP,
                    true  // Clear accumulator
                );
                // Notify QK^T NoPE completion (2CTA multicast，单缓冲模式)
                ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_qk_done[0], 1|2);

                // ================================================================
                // P += Q_rope @ K_rope^T (RoPE part)
                // ================================================================
                if constexpr (HAVE_ROPE) {
                    // Wait for KV RoPE data to be ready (单缓冲模式)
                    plan.bar_kv_rope_ready[0].arrive_and_expect_tx((B_TOPK/2)*D_ROPE*sizeof(bf16));
                    plan.bar_kv_rope_ready[0].wait(k & 1);
                    ku::tcgen05_after_thread_sync();

                    // P += Q_rope @ KV_rope^T
                    ku::utcmma_ss(
                        tiled_mma_P,
                        sQ_rope,
                        sKV_rope,
                        tP,
                        false  // Accumulate
                    );
                }
            }
        }

    // ========================================
    // Warpgroup 2: Delta Computation (WG2) - 2CTA mode
    // Compute: delta = sum(O * dO, dim=-1)
    // Each CTA computes delta for B_H/2=64 rows
    // ========================================
    } else if (warpgroup_idx == 2) {
        // Wait for dO load to complete (2CTA: each CTA loads B_H/2=64 rows)
        plan.bar_prologue_dO.arrive_and_expect_tx((B_H/2)*D_V*sizeof(bf16));
        plan.bar_prologue_dO.wait(0);

        // Delta computation: delta[i] = sum_j(O[i,j] * dO[i,j])
        // 2CTA mode: each thread processes one row, each CTA processes B_H/2=64 rows
        const int local_row_idx = idx_in_warpgroup;
        if (local_row_idx < B_H/2) {
            // Compute global row index (considering cta_idx offset)
            const int global_row_idx = cta_idx * (B_H/2) + local_row_idx;
            
            // Read O from global memory (forward output)
            const bf16* gO = params.o + s_q_idx * params.stride_o_s_q + global_row_idx * params.d_v;
            
            // Read dO from SMEM (each CTA's SMEM only has B_H/2 rows)
            Tensor sdO = make_tensor(make_smem_ptr(plan.dO.data()), SmemLayoutdO{});
            
            float delta = 0.0f;
            
            // Accumulate O * dO using vectorized loads
            CUTE_UNROLL
            for (int col = 0; col < D_V; col += 8) {
                // Vectorized load of O (8 bf16 values = 128 bits)
                uint4 o_raw = __ldg((const uint4*)(gO + col));
                bf16x8 o_vec;
                *(uint4*)&o_vec = o_raw;
                
                // Read dO from shared memory (using local_row_idx)
                bf16x8 do_vec;
                *(uint4*)&do_vec = *(uint4*)(&sdO(local_row_idx, col));
                
                // Accumulate dot product
                delta += __bfloat162float(o_vec.a01.x) * __bfloat162float(do_vec.a01.x);
                delta += __bfloat162float(o_vec.a01.y) * __bfloat162float(do_vec.a01.y);
                delta += __bfloat162float(o_vec.a23.x) * __bfloat162float(do_vec.a23.x);
                delta += __bfloat162float(o_vec.a23.y) * __bfloat162float(do_vec.a23.y);
                delta += __bfloat162float(o_vec.a45.x) * __bfloat162float(do_vec.a45.x);
                delta += __bfloat162float(o_vec.a45.y) * __bfloat162float(do_vec.a45.y);
                delta += __bfloat162float(o_vec.a67.x) * __bfloat162float(do_vec.a67.x);
                delta += __bfloat162float(o_vec.a67.y) * __bfloat162float(do_vec.a67.y);
            }
            
            // Write delta to shared memory buffer (using local_row_idx)
            plan.rowwise_delta_buf[local_row_idx] = delta;
        }
        
        // Sync to ensure all delta computations are done
        NamedBarrier::arrive_and_wait(128, NamedBarriers::wg2_sync);
    }
 
 #else
     // Error handling for non-SM100 architectures
     if (cute::thread0()) {
         CUTE_INVALID_CONTROL_PATH("This kernel only supports sm100");
     }
 #endif
 }
 
/**
 * @brief Host wrapper function to launch backward Phase1 kernel (2CTA Mode)
 * @tparam D_QK Query/Key dimension (576 or 512)
 * @param params Attention computation parameter struct
 * 
 * Functionality:
 * 1. Parameter validation
 * 2. Create TMA descriptors (Q_nope, Q_rope, dO, O, KV) with 2SM support
 * 3. Configure and launch CUDA kernel with cluster
 * 
 * 2CTA Mode:
 * - Grid: [2*s_q, 1, 1] - 2 CTAs per query token
 * - Cluster: [2, 1, 1] - 2 CTAs form a cluster for cooperative computation
 */
template<int D_QK>
void run_bwd_phase1_kernel(const SparseAttnBwdParams& params) {
    // === Parameter validation ===
    KU_ASSERT(params.h_kv == 1);            // KV head count must be 1 (MLA feature)
    KU_ASSERT(params.topk % B_TOPK == 0);   // TopK must be multiple of B_TOPK (skip boundary check)
    KU_ASSERT(params.h_q == B_H);           // Query head count must equal B_H
    KU_ASSERT(params.d_qk == D_QK);
    static_assert(D_QK == 576 || D_QK == 512);  // Only support these two QK dimensions

    // === Create TMA descriptor for Q NoPE (2SM mode) ===
    // Q NoPE shape: [h_q, d_v, s_q]
    auto shape_Q_nope = make_shape(params.h_q, D_V, params.s_q);
    auto tma_Q_nope = cute::make_tma_copy(
        SM100_TMA_2SM_LOAD_NOSPLIT{},  // 2SM TMA for cluster
        make_tensor(
            make_gmem_ptr((bf16*)params.q),
            make_layout(
                shape_Q_nope,
                make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q)
            )
        ),
        SmemLayoutQNoPE{}
    );

    // === Create TMA descriptor for Q RoPE (2SM mode) ===
    // Q RoPE shape: [h_q, d_qk-d_v, s_q], starting from Q offset D_V
    auto shape_Q_rope = make_shape(params.h_q, D_Q-D_V, params.s_q);
    auto tma_Q_rope = cute::make_tma_copy(
        SM100_TMA_2SM_LOAD_NOSPLIT{},  // 2SM TMA for cluster
        make_tensor(
            make_gmem_ptr((bf16*)params.q + D_V),  // RoPE part starts from D_V offset
            make_layout(
                shape_Q_rope,
                make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q)
            )
        ),
        SmemLayoutQRoPE{}
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
        SmemLayoutdO{}
    );

    // === Create TMA descriptor for O (to read forward output for Delta computation) ===
    // O shape: [h_q, d_v, s_q]
    auto shape_O = make_shape(params.h_q, params.d_v, params.s_q);
    auto tma_O = cute::make_tma_copy(
        SM100_TMA_2SM_LOAD_NOSPLIT{},  // 2SM TMA for cluster
        make_tensor(
            make_gmem_ptr((bf16*)params.o),
            make_layout(
                shape_O,
                make_stride(params.stride_o_h_q, _1{}, params.stride_o_s_q)
            )
        ),
        SmemLayoutdO{}  // Reuse dO layout
    );

    // === Create TensorMap for KV NoPE (for TMA gather) ===
    CUtensorMap tensor_map_kv_nope;
    {
        uint64_t size[2] = {D_V, (unsigned long)params.s_kv};  // [d_v, s_kv]
        uint64_t stride[1] = {params.stride_kv_s_kv*sizeof(bf16)};  // Row stride
        uint32_t box_size[2] = {64, 1};    // Box size per load: 64 cols x 1 row
        uint32_t elem_stride[2] = {1, 1};  // Element stride
        CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &tensor_map_kv_nope,
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
        decltype(shape_Q_nope), decltype(tma_Q_nope),
        decltype(shape_Q_rope), decltype(tma_Q_rope),
        decltype(shape_dO), decltype(tma_dO),
        decltype(shape_O), decltype(tma_O)
    > tma_params = {
        shape_Q_nope, tma_Q_nope,
        shape_Q_rope, tma_Q_rope,
        shape_dO, tma_dO,
        shape_O, tma_O,
        tensor_map_kv_nope
    };
    
    // Enable RoPE when D_QK==576 (576 = 512 + 64)
    auto kernel = &sparse_attn_bwd_kernel_part0<D_QK == 576, decltype(tma_params)>;

    // === Configure and launch kernel with cluster (2CTA mode) ===
    constexpr size_t smem_size = sizeof(SharedMemoryPlan);  // Dynamic shared memory size
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    // 2CTA Cluster Launch:
    // - Grid: 2*s_q blocks (2 CTAs per query token)
    // - Block: NUM_THREADS (384) threads
    // - Cluster: (2, 1, 1) - 2 CTAs form a cluster
    cutlass::ClusterLaunchParams launch_params = {
        dim3(2*params.s_q, 1, 1),   // Grid: 2 CTAs per query
        dim3(NUM_THREADS, 1, 1),    // Block: 384 threads
        dim3(2, 1, 1),              // Cluster: 2 CTAs
        smem_size,                  // Dynamic shared memory
        params.stream               // CUDA stream
    };
    KU_CUTLASS_CHECK(cutlass::launch_kernel_on_cluster(
        launch_params, (void*)kernel, params, tma_params
    ));
}
 
 }  // namespace sm100::bwd::head128