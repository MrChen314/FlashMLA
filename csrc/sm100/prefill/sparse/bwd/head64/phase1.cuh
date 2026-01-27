/**
 * @file phase1.cuh
 * @brief SM100 Sparse Attention Backward Kernel Implementation (head_dim=64) - Part0
 * 
 * This file implements sparse MLA attention backward computation optimized for
 * NVIDIA Blackwell (SM100) architecture.
 * Part0 contains the following computation flow:
 *   - WG0: Data Loading (TMA prefetch Q/dO, loop load KV)
 *   - WG1: QK^T Matrix Multiplication (NoPE + RoPE parts)
 *   - WG2: Delta Computation (delta = sum(O * dO))
 */

 #pragma once

 #include <math_constants.h>
 #include <cute/tensor.hpp>
 #include <cutlass/arch/reg_reconfig.h>
 #include <cutlass/arch/arch.h>
 #include <cutlass/cuda_host_adapter.hpp>
 
 #include <kerutils/kerutils.cuh>
 
 #include "params.h"
 #include "utils.h"
 #include "sm100/helpers.h"
 #include "config.h"
 
 namespace sm100::bwd::head64 {
 
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
  * @brief Sparse Attention Backward Kernel - Part0
  * @tparam HAVE_ROPE Whether RoPE positional encoding is included
  * @tparam TmaParams TMA parameter type
  * @param params Attention computation parameters
  * @param tma_params TMA descriptor parameters
  * 
  * Thread Organization:
  * - Grid: [s_q, 1, 1] - Each block processes one query token
  * - Block: 384 threads = 3 warpgroups (128 threads each)
  *   - Warpgroup 0: Data loading (TMA)
  *   - Warpgroup 1: QK^T computation (MMA)
  *   - Warpgroup 2: Delta computation
  */
 template<bool HAVE_ROPE, typename TmaParams>
 __global__ void __launch_bounds__(NUM_THREADS, 1, 1)
 sparse_attn_bwd_kernel_part0(__grid_constant__ const SparseAttnBwdParams params, 
                               __grid_constant__ const TmaParams tma_params) {
 #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
     
     // === Thread index computation ===
     const int s_q_idx = blockIdx.x;                                     // Current query index
     const int warp_idx = cutlass::canonical_warp_idx_sync();            // Warp index (0-11)
     const int lane_idx = threadIdx.x % 32;                              // Lane index (0-31)
     const int warpgroup_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);  // Warpgroup index (0-2)
     const int idx_in_warpgroup = threadIdx.x % 128;                     // Thread index within warpgroup
     const int topk_length = params.topk_length != nullptr ? __ldg(params.topk_length + s_q_idx) : params.topk;
     const int num_k_blocks = max(cute::ceil_div(topk_length, (int)B_TOPK), 1);
 
     // === Shared memory setup ===
     extern __shared__ char wksp_buf[];
     SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);
 
     int* gIndices = params.indices + s_q_idx * params.stride_indices_s_q;
 
     // === TMEM tensor allocation ===
     TiledMMA tiled_mma_P = TiledMMA_P{};
     
     // P matrix fragment (attention scores): [B_H, B_TOPK*2] = [64, 64] (Dual GEMM)
     Tensor tP = partition_fragment_C(tiled_mma_P, Shape<Int<B_H>, Int<B_TOPK*2>>{});
     tP.data().get() = tmem_cols::P;
 
     // === Warp 0: Prologue - Initialize barriers and load Q/dO ===
     if (warp_idx == 0) {
         if (elect_one_sync()) {
             // --- Prefetch TMA descriptors ---
             if constexpr (HAVE_ROPE) {
                 cute::prefetch_tma_descriptor(tma_params.tma_Q_rope.get_tma_descriptor());
             }
             cute::prefetch_tma_descriptor(tma_params.tma_Q_nope.get_tma_descriptor());
             cute::prefetch_tma_descriptor(tma_params.tma_dO.get_tma_descriptor());
 
             // --- Initialize barriers ---
             plan.bar_prologue_q_nope.init(1);
             plan.bar_prologue_q_rope.init(1);
             plan.bar_prologue_dO.init(1);
             plan.bar_prologue_utccp.init(1);
             
             // QK^T related barriers
             plan.bar_qk_nope_done.init(1);
             plan.bar_qk_rope_done.init(1);
             
             // KV ready barriers
             plan.bar_kv_nope_ready.init(1);
             plan.bar_kv_rope_ready.init(1);
             
             // P/dP free barriers
             plan.bar_p_free.init(128);
             plan.bar_dp_free.init(128);
             
             // KV validity barriers
             plan.bar_kv_valid_ready.init(B_TOPK/8);
             plan.bar_kv_valid_free.init(128);
             
             // S/dO ready barrier
             plan.bar_s_do_ready.init(128);
             
             fence_barrier_init();
 
             // --- Load Q matrix using TMA ---
             if constexpr (HAVE_ROPE) {
                 Tensor gQ_rope = tma_params.tma_Q_rope.get_tma_tensor(tma_params.shape_Q_rope)(_, _, s_q_idx);
                 Tensor sQ_rope = make_tensor(make_smem_ptr(plan.u.q_kv.q_rope.data()), SmemLayoutQRoPE{});
                 ku::launch_tma_copy(tma_params.tma_Q_rope, gQ_rope, sQ_rope, plan.bar_prologue_q_rope, TMA::CacheHintSm90::EVICT_FIRST);
             }
 
             Tensor gQ_nope = tma_params.tma_Q_nope.get_tma_tensor(tma_params.shape_Q_nope)(_, _, s_q_idx);
             Tensor sQ_nope = make_tensor(make_smem_ptr(plan.u.q_kv.q_nope.data()), SmemLayoutQNoPE{});
             ku::launch_tma_copy(tma_params.tma_Q_nope, gQ_nope, sQ_nope, plan.bar_prologue_q_nope, TMA::CacheHintSm90::EVICT_FIRST);
 
             // --- Load dO ---
             Tensor gdO = tma_params.tma_dO.get_tma_tensor(tma_params.shape_dO)(_, _, s_q_idx);
             Tensor sdO = make_tensor(make_smem_ptr(plan.dO.data()), SmemLayoutdO{});
             ku::launch_tma_copy(tma_params.tma_dO, gdO, sdO, plan.bar_prologue_dO, TMA::CacheHintSm90::EVICT_FIRST);
 
             // Prefetch KV TMA descriptor
             cute::prefetch_tma_descriptor(&(tma_params.tensor_map_kv_nope));
         }
 
         // --- Allocate TMEM ---
         cute::TMEM::Allocator1Sm().allocate(512, plan.tmem_start_addr.data());
         TRAP_ONLY_DEVICE_ASSERT(plan.tmem_start_addr.data()[0] == 0);
         cute::TMEM::Allocator1Sm().release_allocation_lock();
     }
 
     __syncthreads();
 
     // ========================================
     // Warpgroup 0: Data Loading (WG0)
     // Responsibility: TMA prefetch Q/dO, loop load KV
     // ========================================
     if (warpgroup_idx == 0) {
         int local_warp_idx = cutlass::canonical_warp_idx_sync();
         constexpr int NUM_WARPS = 4;
         constexpr int NUM_LOCAL_ROWS_PER_WARP = (B_TOPK/4) / NUM_WARPS;
 
         if (elect_one_sync()) {
             CUTE_NO_UNROLL
             for (int k = 0; k < num_k_blocks; ++k) {
                 // --- Load TopK indices ---
                 int4 indices[NUM_LOCAL_ROWS_PER_WARP];
                 int max_indices = -1, min_indices = params.s_kv;
                 CUTE_UNROLL
                 for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                     indices[local_row] = __ldg((int4*)(gIndices + k*B_TOPK) + local_row*NUM_WARPS + local_warp_idx);
                     max_indices = max(max_indices, int4_max(indices[local_row]));
                     min_indices = min(min_indices, int4_min(indices[local_row]));
                 }
                 bool is_all_rows_invalid = min_indices == params.s_kv || max_indices == -1;
                 bool should_skip_tma = is_all_rows_invalid && k >= 1;
 
                 // --- Load KV NoPE using TMA gather ---
                 plan.bar_p_free.wait(k&1^1);  // Wait for P matrix to be free
                 bf16* sKV_nope_base = plan.u.q_kv.kv_nope.data() + local_warp_idx*4*64;
 
                 // Lambda: Load KV NoPE
                 auto load_kv_nope = [&]() {
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
                 };
 
                 if (!should_skip_tma) {
                     load_kv_nope();
                 } else {
                     plan.bar_kv_nope_ready.complete_transaction(NUM_LOCAL_ROWS_PER_WARP*4*D_V*sizeof(bf16));
                 }
             }
         }
 
     // ========================================
     // Warpgroup 1: QK^T Computation (WG1)
     // Responsibility: Execute P = Q @ K^T (NoPE + RoPE)
     // ========================================
     } else if (warpgroup_idx == 1) {
         // === Warp 4 (Warp 0 in WG1): MMA control warp ===
         if (warp_idx == 4 && elect_one_sync()) {
             // --- Build SMEM tensor views ---
             // Q NoPE: [B_H, D_V] = [64, 512]
             Tensor sQ_nope = make_tensor(
                 make_smem_ptr(plan.u.q_kv.q_nope.data()),
                 SmemLayoutQNoPE_TiledMMA{}
             );
             
             // Q RoPE: [B_H*2, D_ROPE/2] = [128, 32] for Dual GEMM
             Tensor sQ_rope = make_tensor(
                 make_smem_ptr(plan.u.q_kv.q_rope.data()),
                 SmemLayoutQRoPE_TiledMMA{}
             );
 
             // KV NoPE: reshape to [B_TOPK*2, D_V/2] = [64, 256] for Dual GEMM
             Tensor sKV_nope = make_tensor(
                 make_smem_ptr(plan.u.q_kv.kv_nope.data()),
                 SmemLayoutKVNoPE_TiledMMA{}
             );
             
             // KV RoPE: [B_TOPK*2, D_ROPE/2] = [64, 32]
             Tensor sKV_rope = make_tensor(
                 make_smem_ptr(plan.u.q_kv.kv_rope.data()),
                 SmemLayoutKVRoPE_TiledMMA{}
             );
 
             // Wait for Q load to complete
             plan.bar_prologue_q_nope.arrive_and_expect_tx(B_H*D_V*sizeof(bf16));
             plan.bar_prologue_q_nope.wait(0);
             if constexpr (HAVE_ROPE) {
                 plan.bar_prologue_q_rope.arrive_and_expect_tx(B_H*D_ROPE*sizeof(bf16));
                 plan.bar_prologue_q_rope.wait(0);
             }
             ku::tcgen05_after_thread_sync();
 
             // --- Split Q NoPE and KV NoPE for Dual GEMM ---
             // Dual GEMM: P = Q @ K^T = Q0 @ K0^T + Q1 @ K1^T
             // where Q = [Q0 | Q1] (column concat), K = [K0 | K1] (column concat)
             //
             // Q NoPE: [64, 512] split by COLUMN to [64, 256] x 2
             // Q0 = Q[:, 0:256], Q1 = Q[:, 256:512]
             Tensor sQ_nope_divided = flat_divide(sQ_nope, Tile<Int<B_H>, Int<D_V/2>>{})(_, _, _0{}, _);
             
             // KV NoPE: [64, 256] is already reshaped from K[32, 512]:
             //   KV_reshape[0:32, :] = K[:, 0:256]   (K0)
             //   KV_reshape[32:64, :] = K[:, 256:512] (K1)
             // Split by ROW to [32, 256] x 2
             // See docs/dualgemm.md for the mapping mechanism
             Tensor sKV_nope_divided = flat_divide(sKV_nope, Tile<Int<B_TOPK>, Int<D_V/2>>{})(_, _, _, _0{});
 
             // --- Main computation loop ---
             CUTE_NO_UNROLL
             for (int k = 0; k < num_k_blocks; ++k) {
                 // Wait for P matrix buffer to be available
                 plan.bar_p_free.wait(k&1^1);
                 ku::tcgen05_after_thread_sync();
 
                 // ================================================================
                 // First matmul: P = Q_nope @ K_nope^T (Dual GEMM)
                 // 
                 // Dual GEMM decomposition:
                 //   P = Q @ K^T = Q0 @ K0^T + Q1 @ K1^T
                 // where:
                 //   Q0 = Q[:, 0:256],   Q1 = Q[:, 256:512]
                 //   K0 = K[:, 0:256],   K1 = K[:, 256:512]
                 // ================================================================
                 
                 // Wait for KV NoPE data to be ready
                 plan.bar_kv_nope_ready.arrive_and_expect_tx(B_TOPK*D_V*sizeof(bf16));
                 plan.bar_kv_nope_ready.wait(0);
                 ku::tcgen05_after_thread_sync();
 
                 CUTE_UNROLL
                 for (int nope_part_idx = 0; nope_part_idx < 2; ++nope_part_idx) {
                     // NoPE first part needs clear_accum (clear on each K block iteration)
                     // NoPE second part accumulates
                     bool clear_accum = (nope_part_idx == 0);
 
                     // Dual GEMM: P = Q @ K^T = Q0 @ K0^T + Q1 @ K1^T
                     // Part 0: Q0[64, 256] @ K0[32, 256]^T -> TMEM P[0:31, :]
                     // Part 1: Q1[64, 256] @ K1[32, 256]^T -> TMEM P[32:63, :] (accumulated)
                     // K's row index determines P's output row position (see dualgemm.md)
                     ku::utcmma_ss(
                         tiled_mma_P,
                         sQ_nope_divided(_, _, nope_part_idx),       // Q[:, part*256:(part+1)*256] = [64, 256]
                         sKV_nope_divided(_, _, nope_part_idx, _),   // KV[part*32:(part+1)*32, :] = [32, 256]
                         tP,
                         clear_accum
                     );
                 }
                 
                 // Notify NoPE part computation done
                 ku::umma_arrive_noelect(plan.bar_qk_nope_done);
 
                 // ================================================================
                 // Second matmul: P += Q_rope @ K_rope^T (RoPE part)
                 // Accumulation mode, no clear
                 // ================================================================
                 if constexpr (HAVE_ROPE) {
                     // Wait for KV RoPE data to be ready
                     plan.bar_kv_rope_ready.arrive_and_expect_tx(B_TOPK*D_ROPE*sizeof(bf16));
                     plan.bar_kv_rope_ready.wait(0);
                     ku::tcgen05_after_thread_sync();
 
                     // P += Q_rope @ KV_rope^T
                     ku::utcmma_ss(
                         tiled_mma_P,
                         sQ_rope,
                         sKV_rope,
                         tP,
                         false  // Accumulate, no clear
                     );
                     
                     ku::umma_arrive_noelect(plan.bar_qk_rope_done);
                 }
             }
         }
 
     // ========================================
     // Warpgroup 2: Delta Computation (WG2)
     // Compute: delta = sum(O * dO, dim=-1)
     // ========================================
     } else if (warpgroup_idx == 2) {
         // Wait for dO load to complete
         plan.bar_prologue_dO.arrive_and_expect_tx(B_H*D_V*sizeof(bf16));
         plan.bar_prologue_dO.wait(0);
 
         // Delta computation: delta[i] = sum_j(O[i,j] * dO[i,j])
         // Each thread processes one row
         const int row_idx = idx_in_warpgroup;
         if (row_idx < B_H) {
             // Read O from global memory (forward output)
             const bf16* gO = params.o + s_q_idx * params.stride_o_s_q + row_idx * params.d_v;
             
             // Read dO from SMEM
             Tensor sdO = make_tensor(make_smem_ptr(plan.dO.data()), SmemLayoutdO{});
             
             float delta = 0.0f;
             
             // Accumulate O * dO using vectorized loads
             CUTE_UNROLL
             for (int col = 0; col < D_V; col += 8) {
                 // Vectorized load of O (8 bf16 values = 128 bits)
                 // Use uint4 (4x32bit) for __ldg since uint128_t is not supported
                 uint4 o_raw = __ldg((const uint4*)(gO + col));
                 bf16x8 o_vec;
                 *(uint4*)&o_vec = o_raw;
                 
                 // Read dO from shared memory
                 bf16x8 do_vec;
                 *(uint4*)&do_vec = *(uint4*)(&sdO(row_idx, col));
                 
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
             
             // Write delta to shared memory buffer
             plan.rowwise_delta_buf[row_idx] = delta;
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
  * @brief Host wrapper function to launch backward Phase1 kernel
  * @tparam D_QK Query/Key dimension (576 or 512)
  * @param params Attention computation parameter struct
  * 
  * Functionality:
  * 1. Parameter validation
  * 2. Create TMA descriptors (Q_nope, Q_rope, dO, O, KV)
  * 3. Configure and launch CUDA kernel
  */
 template<int D_QK>
 void run_bwd_phase1_kernel(const SparseAttnBwdParams& params) {
     // === Parameter validation ===
     KU_ASSERT(params.h_kv == 1);            // KV head count must be 1 (MLA feature)
     KU_ASSERT(params.topk % B_TOPK == 0);   // TopK must be multiple of B_TOPK (skip boundary check)
     KU_ASSERT(params.h_q == B_H);           // Query head count must equal B_H
     KU_ASSERT(params.d_qk == D_QK);
     static_assert(D_QK == 576 || D_QK == 512);  // Only support these two QK dimensions
 
     // === Create TMA descriptor for Q NoPE ===
     // Q NoPE shape: [h_q, d_v, s_q]
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
 
     // === Create TMA descriptor for Q RoPE ===
     // Q RoPE shape: [h_q, d_qk-d_v, s_q], starting from Q offset D_V
     auto shape_Q_rope = make_shape(params.h_q, D_Q-D_V, params.s_q);
     auto tma_Q_rope = cute::make_tma_copy(
         SM90_TMA_LOAD{},
         make_tensor(
             make_gmem_ptr((bf16*)params.q + D_V),  // RoPE part starts from D_V offset
             make_layout(
                 shape_Q_rope,
                 make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q)
             )
         ),
         SmemLayoutQRoPE{}
     );
 
     // === Create TMA descriptor for dO ===
     // dO shape: [h_q, d_v, s_q]
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
 
     // === Create TMA descriptor for O (to read forward output for Delta computation) ===
     // O shape: [h_q, d_v, s_q]
     auto shape_O = make_shape(params.h_q, params.d_v, params.s_q);
     auto tma_O = cute::make_tma_copy(
         SM90_TMA_LOAD{},
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
 
     // === Configure and launch kernel ===
     constexpr size_t smem_size = sizeof(SharedMemoryPlan);  // Dynamic shared memory size
     KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
 
     // Grid: s_q blocks, each block processes one query token
     // Block: NUM_THREADS (384) threads
     kernel<<<params.s_q, NUM_THREADS, smem_size, params.stream>>>(params, tma_params);
     KU_CHECK_KERNEL_LAUNCH();
 }
 
 }  // namespace sm100::bwd::head64