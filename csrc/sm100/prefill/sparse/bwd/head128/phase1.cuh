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
            plan.bar_sv_part_done[0].init(1);
            plan.bar_sv_done[0].init(1);
            plan.bar_kv_ready[0].init(1);
            plan.bar_p_free[0].init(128*2);         // 2CTA sync
            plan.bar_so_ready[0].init(128*2);       // 2CTA sync
            plan.bar_kv_valid_ready[0].init(B_TOPK/8);
            plan.bar_kv_valid_free[0].init(128);
            
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

    if (warpgroup_idx == 0) {
        cutlass::arch::warpgroup_reg_alloc<144>();
        int local_warp_idx = cutlass::canonical_warp_idx_sync();
        constexpr int NUM_WARPS = 4;
        // 2CTA mode: each CTA loads B_TOPK/2 rows, 4 elements per row, 4 warps total
        constexpr int NUM_LOCAL_ROWS_PER_WARP = (B_TOPK/2) / 4 / NUM_WARPS;

        if (elect_one_sync()) {
            bf16* sKV_base = plan.u.q_kv.kv.data() + local_warp_idx*4*64;

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
                // Single buffer mode: can skip invalid rows when k >= 1
                bool should_skip_tma = is_all_rows_invalid && k >= 1;

                // Single buffer mode: always use buffer 0
                constexpr int cur_buf = 0;

                // --- Wait for previous QK^T to complete (single buffer mode) ---
                if (k > 0) {
                    plan.bar_qk_done[0].wait((k-1) & 1);
                }

                // --- Load full KV data using TMA gather (2CTA) ---
                // Load complete KV data (D_Q=576 columns, including NoPE 512 + RoPE 64)
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

                // Single buffer mode: always use bar_kv_ready[0]
                if (!should_skip_tma) {
                    load_kv(plan.bar_kv_ready[0]);
                } else {
                    plan.bar_kv_ready[0].complete_transaction(0u, NUM_LOCAL_ROWS_PER_WARP*4*D_Q*sizeof(bf16), 1u);
                }
            }
        }
        if (warp_idx == 0) {
            cute::TMEM::Allocator2Sm().free(0, 512);
        }

    // ========================================
    // Warpgroup 1: QK^T Computation (WG1) - 2CTA mode
    // Responsibility: Execute P = Q @ K^T (NoPE + RoPE) with 2x1SM MMA
    // ========================================
    } else if (warpgroup_idx == 1) {
        // === CTA0 Warp 4: MMA control warp (only CTA0 initiates MMA in 2CTA mode) ===
        if (cta_idx == 0 && warp_idx == 4 && elect_one_sync()) {
            // Wait for Q load to complete (2CTA: each CTA loads B_H/2 rows)
            plan.bar_prologue_q.arrive_and_expect_tx((B_H)*D_Q*sizeof(bf16));
            plan.bar_prologue_q.wait(0);
            ku::tcgen05_after_thread_sync();

            // --- Main computation loop (single buffer mode, convenient for debugging) ---
            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks; ++k) {
                // Single buffer mode: always use buffer 0
                constexpr int cur_buf = 0;

                // --- Build SMEM tensor views for Q and KV (reload each iteration) ---
                // Use the complete sQ defined at function start: [B_H/2, D_Q] = [64, 576]
                // sQ is already defined at line 94: Tensor sQ = make_tensor(make_smem_ptr(plan.u.q_kv.q.data()), SmemLayoutQ{});
                
                // KV complete tensor: [B_TOPK/2, D_Q] = [32, 576]
                Tensor sKV = make_tensor(
                    make_smem_ptr(plan.u.q_kv.kv.data()),
                    SmemLayoutKV{}
                );

                // Wait for P matrix buffer to be available (2CTA sync, single buffer mode)
                // if (k > 0) {
                //     plan.bar_p_free[0].wait((k-1) & 1);
                // }
                ku::tcgen05_after_thread_sync();

                // ================================================================
                // P = Q @ K^T (2CTA, compute complete D_Q=576 dimensions in one pass)
                // Q: [64, 576], K: [32, 576]
                // 2CTA cooperation: P: [B_H/2, B_TOPK] = [64, 64] (both CTAs complete together)
                // ================================================================
                
                // Wait for KV data to be ready (single buffer mode)
                plan.bar_kv_ready[0].arrive_and_expect_tx((B_TOPK)*D_Q*sizeof(bf16));
                plan.bar_kv_ready[0].wait(k & 1);
                ku::tcgen05_after_thread_sync();

                // P = Q @ K^T (compute complete NoPE + RoPE in one pass)
                ku::utcmma_ss(
                    tiled_mma_P,
                    sQ,
                    sKV,
                    tP,
                    true  // Clear accumulator
                );
                // Notify QK^T completion (2CTA multicast, single buffer mode)
                ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_qk_done[0], 1|2);
            }
        }

    // ========================================
    // Warpgroup 2: Delta Loading (WG2) - 2CTA mode
    // Load precomputed delta from global memory to shared memory
    // Delta is computed by preprocess kernel before main kernel launch
    // Each CTA loads delta for B_H/2=64 rows
    // ========================================
    } else if (warpgroup_idx == 2) {
        if (cta_idx == 0 && warp_idx == 8 && elect_one_sync()) {
            plan.bar_prologue_dO.arrive_and_expect_tx((B_H)*D_V*sizeof(bf16));
            plan.bar_prologue_dO.wait(0);
            ku::tcgen05_after_thread_sync();
        }
        __syncthreads();

        // Load precomputed delta from global memory to shared memory
        // 2CTA mode: each thread processes one row, each CTA processes B_H/2=64 rows
        const int local_row_idx = idx_in_warpgroup;
        if (local_row_idx < B_H/2) {
            // Compute global row index (considering cta_idx offset)
            const int global_row_idx = cta_idx * (B_H/2) + local_row_idx;
            
            // Read precomputed delta from global memory
            const float* gDelta = params.delta + s_q_idx * params.stride_delta_s_q + global_row_idx * params.stride_delta_h_q;
            float delta = __ldg(gDelta);
            
            // Write delta to shared memory buffer (using local_row_idx)
            plan.rowwise_delta_buf[local_row_idx] = delta;
        }
        
        // Sync to ensure all delta loads are done
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

    // === Create TMA descriptor for O (kept for compatibility, not used in main kernel) ===
    // O shape: [h_q, d_v, s_q]
    // Note: O is no longer used in main kernel since delta is precomputed
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
        (typename Kernel::SmemLayoutdO){}  // Reuse dO layout
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
        decltype(shape_O), decltype(tma_O)
    > tma_params = {
        shape_Q, tma_Q,
        shape_dO, tma_dO,
        shape_O, tma_O,  // Kept for compatibility, not used in main kernel
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