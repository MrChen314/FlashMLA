#pragma once

#include "dkv_config.h"

#include <cstring>
#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cuda_host_adapter.hpp>

#include "params.h"
#include "utils.h"
#include "sm100/helpers.h"

namespace sm100::bwd::head128_2kernels::dkv {

using namespace cute;

CUTE_DEVICE
void atomic_add_32floats_unrolled(float* dst, const float* src) {
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
        :: "l"(dst + 0), "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3]) : "memory");
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
        :: "l"(dst + 4), "f"(src[4]), "f"(src[5]), "f"(src[6]), "f"(src[7]) : "memory");
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
        :: "l"(dst + 8), "f"(src[8]), "f"(src[9]), "f"(src[10]), "f"(src[11]) : "memory");
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
        :: "l"(dst + 12), "f"(src[12]), "f"(src[13]), "f"(src[14]), "f"(src[15]) : "memory");
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
        :: "l"(dst + 16), "f"(src[16]), "f"(src[17]), "f"(src[18]), "f"(src[19]) : "memory");
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
        :: "l"(dst + 20), "f"(src[20]), "f"(src[21]), "f"(src[22]), "f"(src[23]) : "memory");
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
        :: "l"(dst + 24), "f"(src[24]), "f"(src[25]), "f"(src[26]), "f"(src[27]) : "memory");
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
        :: "l"(dst + 28), "f"(src[28]), "f"(src[29]), "f"(src[30]), "f"(src[31]) : "memory");
}

enum class WarpRole {
    SoftmaxAndDQTransfer = 0x1,
    KvTileTransfer = 0x2,
    DkvTransfer = 0x3,
    Mma = 0x4,
    KvValidLoad = 0x5,
};

static constexpr int kNumSoftmaxAndDQTransferWarps = 4;
static constexpr int kNumKvTileTransferWarps = 2;
static constexpr int kNumDkvTransferWarps = 8;
static constexpr int kNumMmaWarps = 1;
static constexpr int kNumKvValidLoadWarps = 1;
static constexpr int kThreadsPerWarp = 32;

static constexpr int kSoftmaxAndDQTransferFirstWarp = 0;
static constexpr int kKvTileTransferFirstWarp = kSoftmaxAndDQTransferFirstWarp + kNumSoftmaxAndDQTransferWarps;
static constexpr int kDkvTransferFirstWarp = kKvTileTransferFirstWarp + kNumKvTileTransferWarps;
static constexpr int kMmaFirstWarp = kDkvTransferFirstWarp + kNumDkvTransferWarps;
static constexpr int kKvValidLoadFirstWarp = kMmaFirstWarp + kNumMmaWarps;
static constexpr int kNumAssignedWarps =
    kNumSoftmaxAndDQTransferWarps + kNumKvTileTransferWarps +
    kNumDkvTransferWarps + kNumMmaWarps + kNumKvValidLoadWarps;
static constexpr unsigned long long kWarpAssignment = 0x5433'3333'3322'1111ull;

static_assert(kNumAssignedWarps == 16, "Warp assignment must cover exactly 16 warps");
static_assert(kKvValidLoadFirstWarp + kNumKvValidLoadWarps == kNumAssignedWarps, "Warp role ranges must be contiguous");
static_assert(NUM_THREADS == kNumAssignedWarps * kThreadsPerWarp, "NUM_THREADS must match warp assignment");

CUTE_DEVICE
WarpRole warp_idx_to_role(int warp_idx) {
    return static_cast<WarpRole>((kWarpAssignment >> (4 * warp_idx)) & 0xF);
}

template<typename TmaParamsType>
__global__ __launch_bounds__(NUM_THREADS, 1) void dkv_phase_kernel(
    __grid_constant__ const SparseAttnBwdParams params,
    __grid_constant__ const TmaParamsType tma_params
) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200))
    extern __shared__ char smem_raw[];
    SharedMemoryPlan& plan = *reinterpret_cast<SharedMemoryPlan*>(smem_raw);

    const int cta_idx = blockIdx.x % 2;
    const int s_q_idx = blockIdx.x / 2;
    const int max_kv_i = params.q_start_index_s + s_q_idx;
    const int tid = threadIdx.x;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = tid % 32;
    const WarpRole warp_role = warp_idx_to_role(warp_idx);
    if (s_q_idx >= params.s_q) {
        return;
    }

    const int topk_length = params.topk_length == nullptr ?
        params.topk :
        min(max(__ldg(params.topk_length + s_q_idx), 0), params.topk);
    const int32_t* gIndices_s = params.indices + (int64_t)s_q_idx * params.stride_indices_s_q;
    const int num_k_blocks = max(cute::ceil_div(topk_length, (int)B_TOPK), 1);

    if (tid == 0) {
        cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_Q_rope.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dO.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_S.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dS.get_tma_descriptor());
    }

    if (warp_idx == 0 && elect_one_sync()) {
        plan.bar_q_nope_ready.init(1);
        plan.bar_q_rope_ready.init(1);
        plan.bar_dO_ready.init(1);
        plan.bar_s_tile_ready.init(1);
        plan.bar_ds_tile_ready.init(1);
        plan.bar_dkv_nope_ready.init(1);
        plan.bar_dkv_rope_ready.init(1);
        fence_barrier_init();
    }

    cluster_sync();

    Tensor sQ = make_tensor(make_smem_ptr(plan.q.data()), SmemLayoutQ{});
    Tensor sQRoPE = make_tensor(make_smem_ptr(plan.q_rope.data()), SmemLayoutQRoPE{});
    Tensor sdO = make_tensor(make_smem_ptr(plan.dO.data()), SmemLayoutdO{});
    Tensor sS = make_tensor(make_smem_ptr(plan.s_ds.s.data()), SmemLayoutS{});
    Tensor sDS = make_tensor(make_smem_ptr(plan.s_ds.ds.data()), SmemLayoutdS{});

    if (warp_idx == 0) {
        if (elect_one_sync()) {
            Tensor gQ = flat_divide(
                tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx),
                Tile<Int<D_V / 2>>{}
            )(_, cta_idx, _);
            ku::launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_q_nope_ready, TMA::CacheHintSm90::EVICT_FIRST);

            Tensor gQRoPE = flat_divide(
                tma_params.tma_Q_rope.get_tma_tensor(tma_params.shape_Q_rope)(_, _, s_q_idx),
                Tile<Int<D_ROPE / 2>>{}
            )(_, cta_idx, _);
            ku::launch_tma_copy(
                tma_params.tma_Q_rope, gQRoPE, sQRoPE, plan.bar_q_rope_ready, TMA::CacheHintSm90::EVICT_FIRST);

            Tensor gdO = flat_divide(
                tma_params.tma_dO.get_tma_tensor(tma_params.shape_dO)(_, _, s_q_idx),
                Tile<Int<D_V / 2>>{}
            )(_, cta_idx, _);
            ku::launch_tma_copy(tma_params.tma_dO, gdO, sdO, plan.bar_dO_ready, TMA::CacheHintSm90::EVICT_FIRST);
        }

        plan.bar_q_nope_ready.arrive_and_expect_tx((D_V / 2) * B_H * sizeof(bf16));
        plan.bar_q_rope_ready.arrive_and_expect_tx((D_ROPE / 2) * B_H * sizeof(bf16));
        plan.bar_dO_ready.arrive_and_expect_tx((D_V / 2) * B_H * sizeof(bf16));
        plan.bar_q_nope_ready.wait(0);
        plan.bar_q_rope_ready.wait(0);
        plan.bar_dO_ready.wait(0);
        ku::tcgen05_after_thread_sync();

        TMEM::Allocator2Sm().allocate(512, plan.tmem_start_addr.data());
        TMEM::Allocator2Sm().release_allocation_lock();
    }

    __syncthreads();
    const uint32_t tmem_base = plan.tmem_start_addr.data()[0];
    cluster_sync();

    TiledMMA_dKV tiled_mma_dKV{};
    TiledMMA_dKV_RoPE tiled_mma_dKV_RoPE{};
    Tensor tdKV = partition_fragment_C(tiled_mma_dKV, Shape<Int<B_TOPK / 2>, Int<D_V>>{});
    Tensor tdKV_RoPE = partition_fragment_C(tiled_mma_dKV_RoPE, Shape<Int<B_TOPK / 2>, Int<D_ROPE>>{});
    tdKV.data().get() = tmem_cols::dKV;
    tdKV_RoPE.data().get() = tmem_cols::dKV_RoPE;

    CUTE_NO_UNROLL
    for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
        const int phase = k_block & 1;

        if (warp_idx == 0) {
            if (elect_one_sync()) {
                const int topk_tile_idx = k_block * 2 + cta_idx;
                Tensor gS = flat_divide(
                    tma_params.tma_S.get_tma_tensor(tma_params.shape_S)(_, _, s_q_idx),
                    Tile<Int<B_TOPK / 2>>{}
                )(_, topk_tile_idx, _);
                ku::launch_tma_copy(tma_params.tma_S, gS, sS, plan.bar_s_tile_ready, TMA::CacheHintSm90::EVICT_FIRST);

                Tensor gdS = flat_divide(
                    tma_params.tma_dS.get_tma_tensor(tma_params.shape_dS)(_, _, s_q_idx),
                    Tile<Int<B_TOPK / 2>>{}
                )(_, topk_tile_idx, _);
                ku::launch_tma_copy(tma_params.tma_dS, gdS, sDS, plan.bar_ds_tile_ready, TMA::CacheHintSm90::EVICT_FIRST);
            }

            plan.bar_s_tile_ready.arrive_and_expect_tx((B_TOPK / 2) * B_H * sizeof(bf16));
            plan.bar_ds_tile_ready.arrive_and_expect_tx((B_TOPK / 2) * B_H * sizeof(bf16));
            plan.bar_s_tile_ready.wait(phase);
            plan.bar_ds_tile_ready.wait(phase);
            ku::tcgen05_after_thread_sync();
        }

        __syncthreads();
        cluster_sync();

        if (warp_role == WarpRole::Mma && cta_idx == 0 && elect_one_sync()) {
            ku::utcmma_ss(tiled_mma_dKV, sS, sdO, tdKV, true);
            ku::utcmma_ss(tiled_mma_dKV, sDS, sQ, tdKV, false);
            ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_dkv_nope_ready, 1 | 2);
            ku::tcgen05_after_thread_sync();

            ku::utcmma_ss(tiled_mma_dKV_RoPE, sDS, sQRoPE, tdKV_RoPE, true);
            ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_dkv_rope_ready, 1 | 2);
            ku::tcgen05_after_thread_sync();
        }

        if (warp_role == WarpRole::DkvTransfer) {
            const int tmem_lane_128 = (warp_idx & 0x3) * kThreadsPerWarp + lane_idx;
            const int row = tmem_lane_128 % (B_TOPK / 2);
            const int half = (tmem_lane_128 / (B_TOPK / 2)) & 1;
            const int chunk_group = (warp_idx - kDkvTransferFirstWarp) / 4;
            const int row_global = k_block * B_TOPK + cta_idx * (B_TOPK / 2) + row;

            int kv_idx = -1;
            if (row_global < topk_length) {
                kv_idx = __ldg(gIndices_s + row_global);
            }
            const bool row_valid = kv_idx >= 0 && kv_idx < params.s_kv && kv_idx <= max_kv_i;

            plan.bar_dkv_nope_ready.wait(phase);
            ku::tcgen05_after_thread_sync();

            constexpr int COLS_PER_HALF = D_V / 2;
            constexpr int CHUNK_SIZE = 32;
            constexpr int NUM_CHUNKS = COLS_PER_HALF / CHUNK_SIZE;
            constexpr int NUM_CHUNK_GROUPS = kNumDkvTransferWarps / 4;
            constexpr int CHUNKS_PER_GROUP = NUM_CHUNKS / NUM_CHUNK_GROUPS;
            static_assert(NUM_CHUNKS % NUM_CHUNK_GROUPS == 0);

            CUTE_UNROLL
            for (int local_chunk = 0; local_chunk < CHUNKS_PER_GROUP; ++local_chunk) {
                const int chunk = chunk_group * CHUNKS_PER_GROUP + local_chunk;
                float2 dkv_data[CHUNK_SIZE / 2];
                ku::tmem_ld_32dp32bNx<CHUNK_SIZE>(tmem_base + tmem_cols::dKV + chunk * CHUNK_SIZE, dkv_data);
                cutlass::arch::fence_view_async_tmem_load();
                ku::tcgen05_before_thread_sync();
                if (row_valid) {
                    float* dst = params.dKV + kv_idx * params.stride_dKV_s_kv + half * COLS_PER_HALF + chunk * CHUNK_SIZE;
                    atomic_add_32floats_unrolled(dst, reinterpret_cast<float*>(dkv_data));
                }
            }
            plan.bar_dkv_rope_ready.wait(phase);
            ku::tcgen05_after_thread_sync();
            constexpr int ROPE_COLS_PER_HALF = D_ROPE / 2;
            float2 dkv_rope_data[ROPE_COLS_PER_HALF / 2];
            ku::tmem_ld_32dp32bNx<ROPE_COLS_PER_HALF>(tmem_base + tmem_cols::dKV_RoPE, dkv_rope_data);
            cutlass::arch::fence_view_async_tmem_load();
            ku::tcgen05_before_thread_sync();
            if (row_valid && chunk_group == 0) {
                float* dst = params.dKV + kv_idx * params.stride_dKV_s_kv + D_V + half * ROPE_COLS_PER_HALF;
                atomic_add_32floats_unrolled(dst, reinterpret_cast<float*>(dkv_rope_data));
            }
        }

        cluster_sync();
    }

    cluster_sync();

    if (warp_idx == 0 && elect_one_sync()) {
        TMEM::Allocator2Sm().free(tmem_base, 512);
    }
#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm100");
    }
#endif
}

static void launch_dkv_phase(const SparseAttnBwdParams& params) {
    auto shape_Q = cute::make_shape(D_V, B_H, params.s_q);
    auto tma_Q = cute::make_tma_copy(
        cute::SM100_TMA_2SM_LOAD_NOSPLIT{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)params.q),
            cute::make_layout(
                shape_Q,
                cute::make_stride(cute::_1{}, params.stride_q_h_q, params.stride_q_s_q)
            )
        ),
        SmemLayoutQ{}
    );

    auto shape_Q_rope = cute::make_shape(D_ROPE, B_H, params.s_q);
    auto tma_Q_rope = cute::make_tma_copy(
        cute::SM100_TMA_2SM_LOAD_NOSPLIT{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)params.q + D_V),
            cute::make_layout(
                shape_Q_rope,
                cute::make_stride(cute::_1{}, params.stride_q_h_q, params.stride_q_s_q)
            )
        ),
        SmemLayoutQRoPE{}
    );

    auto shape_dO = cute::make_shape(D_V, B_H, params.s_q);
    auto tma_dO = cute::make_tma_copy(
        cute::SM100_TMA_2SM_LOAD_NOSPLIT{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)params.dO),
            cute::make_layout(
                shape_dO,
                cute::make_stride(cute::_1{}, params.stride_dO_h_q, params.stride_dO_s_q)
            )
        ),
        SmemLayoutdO{}
    );

    auto shape_S = cute::make_shape(params.topk, B_H, params.s_q);
    auto tma_S = cute::make_tma_copy(
        cute::SM100_TMA_2SM_LOAD_NOSPLIT{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)params.s),
            cute::make_layout(
                shape_S,
                cute::make_stride(cute::_1{}, params.stride_s_h_q, params.stride_s_s_q)
            )
        ),
        SmemLayoutS{}
    );

    auto shape_dS = cute::make_shape(params.topk, B_H, params.s_q);
    auto tma_dS = cute::make_tma_copy(
        cute::SM100_TMA_2SM_LOAD_NOSPLIT{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)params.ds),
            cute::make_layout(
                shape_dS,
                cute::make_stride(cute::_1{}, params.stride_ds_h_q, params.stride_ds_s_q)
            )
        ),
        SmemLayoutdS{}
    );

    using KernelTmaParams = TmaParams<
        decltype(shape_Q), decltype(tma_Q),
        decltype(shape_Q_rope), decltype(tma_Q_rope),
        decltype(shape_dO), decltype(tma_dO),
        decltype(shape_S), decltype(tma_S),
        decltype(shape_dS), decltype(tma_dS)
    >;

    KernelTmaParams tma_params = {
        shape_Q, tma_Q,
        shape_Q_rope, tma_Q_rope,
        shape_dO, tma_dO,
        shape_S, tma_S,
        shape_dS, tma_dS
    };

    auto kernel = &dkv_phase_kernel<KernelTmaParams>;
    dim3 grid(2 * params.s_q, 1, 1);
    dim3 block(NUM_THREADS, 1, 1);

    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

    cudaLaunchConfig_t config;
    memset(&config, 0, sizeof(config));
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = SMEM_SIZE;
    config.stream = params.stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    KU_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, params, tma_params));
}

template<int DQK>
void run_bwd_dkv_phase_kernel(const SparseAttnBwdParams& params) {
    static_assert(DQK == D_QK);

    KU_ASSERT(params.d_qk == DQK);
    KU_ASSERT(params.d_v == D_V);
    KU_ASSERT(params.h_q == B_H);
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.topk > 0 && params.topk % B_TOPK == 0);
    KU_ASSERT(params.s != nullptr);
    KU_ASSERT(params.ds != nullptr);
    KU_ASSERT(params.dKV != nullptr);

    launch_dkv_phase(params);
}

}  // namespace sm100::bwd::head128_2kernels::dkv
