#pragma once

#include "../head128/preprocess_delta.cuh"

#include "dq_config.h"

#include <cstring>
#include <math_constants.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cuda_host_adapter.hpp>

#include "params.h"
#include "utils.h"
#include "sm100/helpers.h"

namespace sm100::bwd::head128_2kernels::dq {

using namespace cute;

CUTE_DEVICE
int32x8_t ldg_256_indices(void* src_ptr) {
    int32x8_t val;
    asm volatile("ld.global.nc.L1::evict_normal.L2::evict_normal.L2::256B.v8.s32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=r"(val.a0), "=r"(val.a1), "=r"(val.a2), "=r"(val.a3),
          "=r"(val.a4), "=r"(val.a5), "=r"(val.a6), "=r"(val.a7)
        : "l"(src_ptr)
    );
    return val;
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
    kNumDkvTransferWarps +
    kNumMmaWarps + kNumKvValidLoadWarps;
static constexpr unsigned long long kWarpAssignment = 0x5433'3333'3322'1111ull;

static_assert(kNumAssignedWarps == 16, "Warp assignment must cover exactly 16 warps");
static_assert(kKvValidLoadFirstWarp + kNumKvValidLoadWarps == kNumAssignedWarps, "Warp role ranges must be contiguous");
static_assert(NUM_THREADS == kNumAssignedWarps * kThreadsPerWarp, "NUM_THREADS must match warp assignment");

CUTE_DEVICE
WarpRole warp_idx_to_role(int warp_idx) {
    return static_cast<WarpRole>((kWarpAssignment >> (4 * warp_idx)) & 0xF);
}

template<typename TmaParamsType>
__global__ __launch_bounds__(NUM_THREADS, 1) void dq_phase_kernel(
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
    const int lane_idx = threadIdx.x % 32;
    const WarpRole warp_role = warp_idx_to_role(warp_idx);
    if (s_q_idx >= params.s_q) {
        return;
    }
    const int topk_length = params.topk_length == nullptr ?
        params.topk :
        min(max(__ldg(params.topk_length + s_q_idx), 0), params.topk);
    const int32_t* gIndices_s = params.indices + (int64_t)s_q_idx * params.stride_indices_s_q;
    const float* lse_s = params.lse + (int64_t)s_q_idx * params.h_q;
    const float* delta_s = params.delta + (int64_t)s_q_idx * params.stride_delta_s_q;
    const int num_k_blocks = max(cute::ceil_div(topk_length, (int)B_TOPK), 1);

    if (tid == 0) {
        cute::prefetch_tma_descriptor(tma_params.tma_Q_nope.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_Q_rope.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dO.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dQ.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_S.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dS.get_tma_descriptor());
        cute::prefetch_tma_descriptor(&(tma_params.tensor_map_kv));
    }

    if (warp_idx == 0 && elect_one_sync()) {
        plan.bar_prologue_q_nope.init(1);
        plan.bar_prologue_q_rope.init(1);
        plan.bar_prologue_kv.init(1);
        plan.bar_prologue_dO.init(1);
        plan.bar_p_ready.init(1);
        plan.bar_dp_ready.init(1);
        plan.bar_s_ready.init(kNumSoftmaxAndDQTransferWarps * kThreadsPerWarp);
        plan.bar_ds_ready.init(kNumSoftmaxAndDQTransferWarps * kThreadsPerWarp);
        plan.bar_k_valid_ready.init(B_TOPK / 8);
        plan.bar_k_valid_free.init(kNumSoftmaxAndDQTransferWarps * kThreadsPerWarp);
        plan.bar_kv_peer_cp_async.init(1);
        plan.bar_kv_peer_ready.init(1);
        plan.bar_dq_ready.init(1);
        fence_barrier_init();
    }

    cluster_sync();

    Tensor sQNoPE = make_tensor(make_smem_ptr(plan.u.q_kv.q_nope.data()), SmemLayoutQNoPE{});
    Tensor sQRoPE = make_tensor(make_smem_ptr(plan.u.q_kv.q_rope.data()), SmemLayoutQRoPE{});
    Tensor sKNoPE = make_tensor(make_smem_ptr(plan.u.q_kv.k_nope.data()), SmemLayoutKNoPE{});
    Tensor sKRoPE = make_tensor(make_smem_ptr(plan.u.q_kv.k_rope.data()), SmemLayoutKRoPE{});
    Tensor sdO = make_tensor(make_smem_ptr(plan.dO.data()), SmemLayoutdO{});

    if (warp_idx == 0) {
        if (elect_one_sync()) {
            Tensor gQNoPE = flat_divide(
                tma_params.tma_Q_nope.get_tma_tensor(tma_params.shape_Q_nope)(_, _, s_q_idx),
                Tile<Int<B_H / 2>>{}
            )(_, cta_idx, _);
            ku::launch_tma_copy(tma_params.tma_Q_nope, gQNoPE, sQNoPE, plan.bar_prologue_q_nope, TMA::CacheHintSm90::EVICT_FIRST);

            Tensor gQRoPE = flat_divide(
                tma_params.tma_Q_rope.get_tma_tensor(tma_params.shape_Q_rope)(_, _, s_q_idx),
                Tile<Int<B_H / 2>>{}
            )(_, cta_idx, _);
            ku::launch_tma_copy(tma_params.tma_Q_rope, gQRoPE, sQRoPE, plan.bar_prologue_q_rope, TMA::CacheHintSm90::EVICT_FIRST);

            Tensor gdO = flat_divide(
                tma_params.tma_dO.get_tma_tensor(tma_params.shape_dO)(_, _, s_q_idx),
                Tile<Int<B_H / 2>>{}
            )(_, cta_idx, _);
            ku::launch_tma_copy(tma_params.tma_dO, gdO, sdO, plan.bar_prologue_dO, TMA::CacheHintSm90::EVICT_FIRST);
        }

        TMEM::Allocator2Sm().allocate(512, plan.tmem_start_addr.data());
        TMEM::Allocator2Sm().release_allocation_lock();
    }
    __syncthreads();
    const uint32_t tmem_base = plan.tmem_start_addr.data()[0];

    if (warp_role == WarpRole::SoftmaxAndDQTransfer) {
        Tensor sS = make_tensor(make_smem_ptr(plan.s_ds.s.data()), SmemLayoutS{});
        Tensor sDS = make_tensor(make_smem_ptr(plan.s_ds.ds.data()), SmemLayoutdS{});
        const int idx_in_softmax = (warp_idx - kSoftmaxAndDQTransferFirstWarp) * kThreadsPerWarp + lane_idx;
        const int global_row_idx = cta_idx * (B_H / 2) + idx_in_softmax % (B_H / 2);
        const float row_lse = __ldg(lse_s + global_row_idx) * 1.44269504f;
        const float neg_delta_val = __ldg(delta_s + global_row_idx);
        const float sm_scale = params.sm_scale;
        const float scale = params.sm_scale_div_log2;

        const float2 neg_lse_f2 = make_float2(-row_lse, -row_lse);
        const float2 scale_f2 = make_float2(scale, scale);
        const float2 neg_delta_f2 = make_float2(neg_delta_val, neg_delta_val);
        const float2 sm_scale_f2 = make_float2(sm_scale, sm_scale);

        const uint32_t tmem_lane = idx_in_softmax % S_DS_ROWS_PER_CTA;
        const uint32_t tmem_p_addr = tmem_base + (tmem_lane << 16) + tmem_cols::P;
        const uint32_t tmem_dp_addr = tmem_base + (tmem_lane << 16) + tmem_cols::dP;
        const int row_in_tile = idx_in_softmax % S_DS_ROWS_PER_CTA;
        const int col_half = idx_in_softmax / S_DS_ROWS_PER_CTA;
        bf16* sS_base = plan.s_ds.s.data() +
            row_in_tile * S_DS_VEC_ELEMS + col_half * S_DS_ROWS_PER_CTA * S_DS_COLS_PER_THREAD;
        bf16* sDS_base = plan.s_ds.ds.data() +
            row_in_tile * S_DS_VEC_ELEMS + col_half * S_DS_ROWS_PER_CTA * S_DS_COLS_PER_THREAD;

        constexpr int SMEM_VEC_F2 = S_DS_VEC_ELEMS / 2;
        constexpr int NUM_SMEM_VEC_STORES = S_DS_COLS_PER_THREAD / S_DS_VEC_ELEMS;
        constexpr int SMEM_VEC_STRIDE = S_DS_ROWS_PER_CTA * S_DS_VEC_ELEMS;
        static_assert(SMEM_VEC_F2 == 4, "Softmax vectorized write path expects 4 float2 per 128-bit store.");

        CUTE_NO_UNROLL
        for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
            const int phase = k_block & 1;

            plan.bar_p_ready.wait(phase);
            ku::tcgen05_after_thread_sync();

            float2 p[(B_TOPK / 2) / 2];
            ku::tmem_ld_32dp32bNx<B_TOPK / 2>(tmem_p_addr, p);
            cutlass::arch::fence_view_async_tmem_load();
            ku::tcgen05_before_thread_sync();

            plan.bar_k_valid_ready.wait(phase);
            const uint32_t is_k_valid_lo =
                *(uint32_t*)(plan.is_k_valid + (idx_in_softmax >= S_DS_ROWS_PER_CTA ? B_TOPK / 8 / 2 : 0));
            float* p_float = (float*)p;
            CUTE_UNROLL
            for (int i = 0; i < B_TOPK / 2; ++i) {
                if (!(is_k_valid_lo >> i & 1)) {
                    p_float[i] = -CUDART_INF_F;
                }
            }
            plan.bar_k_valid_free.arrive();

            CUTE_UNROLL
            for (int vec = 0; vec < NUM_SMEM_VEC_STORES; ++vec) {
                const int base_idx = vec * SMEM_VEC_F2;
                bf16x8 s_pack;
                {
                    float2 p_vec = ku::float2_fma(p[base_idx + 0], scale_f2, neg_lse_f2);
                    p_vec = make_float2(exp2f(p_vec.x), exp2f(p_vec.y));
                    p[base_idx + 0] = p_vec;
                    s_pack.a01 = __float22bfloat162_rn(p_vec);
                }
                {
                    float2 p_vec = ku::float2_fma(p[base_idx + 1], scale_f2, neg_lse_f2);
                    p_vec = make_float2(exp2f(p_vec.x), exp2f(p_vec.y));
                    p[base_idx + 1] = p_vec;
                    s_pack.a23 = __float22bfloat162_rn(p_vec);
                }
                {
                    float2 p_vec = ku::float2_fma(p[base_idx + 2], scale_f2, neg_lse_f2);
                    p_vec = make_float2(exp2f(p_vec.x), exp2f(p_vec.y));
                    p[base_idx + 2] = p_vec;
                    s_pack.a45 = __float22bfloat162_rn(p_vec);
                }
                {
                    float2 p_vec = ku::float2_fma(p[base_idx + 3], scale_f2, neg_lse_f2);
                    p_vec = make_float2(exp2f(p_vec.x), exp2f(p_vec.y));
                    p[base_idx + 3] = p_vec;
                    s_pack.a67 = __float22bfloat162_rn(p_vec);
                }
                *reinterpret_cast<bf16x8*>(sS_base + vec * SMEM_VEC_STRIDE) = s_pack;
            }
            fence_view_async_shared();
            __threadfence_block();

            plan.bar_s_ready.arrive(static_cast<uint32_t>(cta_idx));
            NamedBarrier::arrive_and_wait(128, 2);
            if (warp_idx == 0 && elect_one_sync()) {
                Tensor gS = flat_divide(
                    tma_params.tma_S.get_tma_tensor(tma_params.shape_S)(_, _, s_q_idx),
                    Shape<Int<B_H / 2>, Int<B_TOPK>>{}
                )(_, _, cta_idx, k_block);
                auto thr_tma_s = tma_params.tma_S.get_slice(_0{});
                cute::copy(
                    tma_params.tma_S,
                    thr_tma_s.partition_S(sS),
                    thr_tma_s.partition_D(gS)
                );
                cute::tma_store_arrive();
                cute::tma_store_wait<0>();
            }
            NamedBarrier::arrive_and_wait(128, 2);

            plan.bar_dp_ready.wait(phase);
            ku::tcgen05_after_thread_sync();

            constexpr int DP_CHUNK_F2 = SMEM_VEC_F2;
            constexpr int NUM_DP_CHUNKS = (B_TOPK / 2) / 2 / DP_CHUNK_F2;
            CUTE_UNROLL
            for (int ch = 0; ch < NUM_DP_CHUNKS; ++ch) {
                float2 dp[DP_CHUNK_F2];
                ku::tmem_ld_32dp32bNx<DP_CHUNK_F2 * 2>(tmem_dp_addr + ch * DP_CHUNK_F2 * 2, dp);
                cutlass::arch::fence_view_async_tmem_load();
                ku::tcgen05_before_thread_sync();

                const int base_idx = ch * DP_CHUNK_F2;
                bf16x8 ds_pack;
                {
                    float2 ds_vec = ku::float2_mul(
                        ku::float2_mul(p[base_idx + 0], ku::float2_add(dp[0], neg_delta_f2)),
                        sm_scale_f2
                    );
                    ds_pack.a01 = __float22bfloat162_rn(ds_vec);
                }
                {
                    float2 ds_vec = ku::float2_mul(
                        ku::float2_mul(p[base_idx + 1], ku::float2_add(dp[1], neg_delta_f2)),
                        sm_scale_f2
                    );
                    ds_pack.a23 = __float22bfloat162_rn(ds_vec);
                }
                {
                    float2 ds_vec = ku::float2_mul(
                        ku::float2_mul(p[base_idx + 2], ku::float2_add(dp[2], neg_delta_f2)),
                        sm_scale_f2
                    );
                    ds_pack.a45 = __float22bfloat162_rn(ds_vec);
                }
                {
                    float2 ds_vec = ku::float2_mul(
                        ku::float2_mul(p[base_idx + 3], ku::float2_add(dp[3], neg_delta_f2)),
                        sm_scale_f2
                    );
                    ds_pack.a67 = __float22bfloat162_rn(ds_vec);
                }
                *reinterpret_cast<bf16x8*>(sDS_base + ch * SMEM_VEC_STRIDE) = ds_pack;
            }
            fence_view_async_shared();
            __threadfence_block();

            plan.bar_ds_ready.arrive(static_cast<uint32_t>(cta_idx));
            NamedBarrier::arrive_and_wait(128, 3);
            if (warp_idx == 0 && elect_one_sync()) {
                Tensor gdS = flat_divide(
                    tma_params.tma_dS.get_tma_tensor(tma_params.shape_dS)(_, _, s_q_idx),
                    Shape<Int<B_H / 2>, Int<B_TOPK>>{}
                )(_, _, cta_idx, k_block);
                auto thr_tma_ds = tma_params.tma_dS.get_slice(_0{});
                cute::copy(
                    tma_params.tma_dS,
                    thr_tma_ds.partition_S(sDS),
                    thr_tma_ds.partition_D(gdS)
                );
                cute::tma_store_arrive();
                cute::tma_store_wait<0>();
            }
            NamedBarrier::arrive_and_wait(128, 3);
        }

        const int final_phase = (num_k_blocks - 1) & 1;
        plan.bar_dq_ready.wait(final_phase);
        ku::tcgen05_after_thread_sync();

        {
            constexpr int dQ_ROWS = B_H / 2;
            constexpr int NOPE_FLOATS_PER_HALF = 256 / 2;
            constexpr int NOPE_CHUNKS = 8;
            constexpr int NOPE_CHUNK_FLOATS = NOPE_FLOATS_PER_HALF / NOPE_CHUNKS;
            constexpr int NOPE_CHUNK_FLOAT2 = NOPE_CHUNK_FLOATS / 2;
            constexpr int ROPE_FLOAT2_PER_ROW = D_ROPE / 2 / 2;

            Tensor sdQ = make_tensor(make_smem_ptr(plan.u.dq.data()), SmemLayoutQ{});

            const int row_in_cta = idx_in_softmax % dQ_ROWS;
            const int col_half = idx_in_softmax / dQ_ROWS;

            const uint32_t tmem_addr_dq0 = tmem_base + (row_in_cta << 16) + tmem_cols::dQ;
            const uint32_t tmem_addr_dq1 = tmem_base + (row_in_cta << 16) + (tmem_cols::dQ + 128);

            CUTE_UNROLL
            for (int chunk = 0; chunk < NOPE_CHUNKS; ++chunk) {
                const int chunk_col_base = chunk * NOPE_CHUNK_FLOATS;
                float2 dq_chunk[NOPE_CHUNK_FLOAT2];
                ku::tmem_ld_32dp32bNx<NOPE_CHUNK_FLOATS>(tmem_addr_dq0 + chunk_col_base, dq_chunk);
                cutlass::arch::fence_view_async_tmem_load();
                ku::tcgen05_before_thread_sync();

                CUTE_UNROLL
                for (int i = 0; i < NOPE_CHUNK_FLOAT2; ++i) {
                    int col = col_half * NOPE_FLOATS_PER_HALF + chunk_col_base + i * 2;
                    sdQ(row_in_cta, col) = bf16(dq_chunk[i].x);
                    sdQ(row_in_cta, col + 1) = bf16(dq_chunk[i].y);
                }
            }

            CUTE_UNROLL
            for (int chunk = 0; chunk < NOPE_CHUNKS; ++chunk) {
                const int chunk_col_base = chunk * NOPE_CHUNK_FLOATS;
                float2 dq_chunk[NOPE_CHUNK_FLOAT2];
                ku::tmem_ld_32dp32bNx<NOPE_CHUNK_FLOATS>(tmem_addr_dq1 + chunk_col_base, dq_chunk);
                cutlass::arch::fence_view_async_tmem_load();
                ku::tcgen05_before_thread_sync();

                CUTE_UNROLL
                for (int i = 0; i < NOPE_CHUNK_FLOAT2; ++i) {
                    int col = 256 + col_half * NOPE_FLOATS_PER_HALF + chunk_col_base + i * 2;
                    sdQ(row_in_cta, col) = bf16(dq_chunk[i].x);
                    sdQ(row_in_cta, col + 1) = bf16(dq_chunk[i].y);
                }
            }

            constexpr int ROPE_F2_CHUNK = 8;
            constexpr int ROPE_NUM_CHUNKS = ROPE_FLOAT2_PER_ROW / ROPE_F2_CHUNK;
            const uint32_t tmem_addr_dq_rope = tmem_base + (row_in_cta << 16) + tmem_cols::dQ_RoPE;
            CUTE_UNROLL
            for (int rch = 0; rch < ROPE_NUM_CHUNKS; ++rch) {
                float2 dq_rope_chunk[ROPE_F2_CHUNK];
                ku::tmem_ld_32dp32bNx<ROPE_F2_CHUNK * 2>(
                    tmem_addr_dq_rope + rch * ROPE_F2_CHUNK * 2, dq_rope_chunk);
                cutlass::arch::fence_view_async_tmem_load();
                ku::tcgen05_before_thread_sync();

                CUTE_UNROLL
                for (int i = 0; i < ROPE_F2_CHUNK; ++i) {
                    const int fi = rch * ROPE_F2_CHUNK + i;
                    int col = D_V + col_half * (D_ROPE / 2) + fi * 2;
                    sdQ(row_in_cta, col) = bf16(dq_rope_chunk[i].x);
                    sdQ(row_in_cta, col + 1) = bf16(dq_rope_chunk[i].y);
                }
            }

            fence_view_async_shared();
            NamedBarrier::arrive_and_wait(128, 0);

            if (warp_idx == 0 && elect_one_sync()) {
                Tensor gdQ = flat_divide(
                    tma_params.tma_dQ.get_tma_tensor(tma_params.shape_dQ)(_, _, s_q_idx),
                    Tile<Int<B_H / 2>>{}
                )(_, cta_idx, _);
                auto thr_tma_dq = tma_params.tma_dQ.get_slice(_0{});
                cute::copy(
                    tma_params.tma_dQ,
                    thr_tma_dq.partition_S(sdQ),
                    thr_tma_dq.partition_D(gdQ)
                );
                cute::tma_store_arrive();
                cute::tma_store_wait<0>();
            }
        }
    }

    if (warp_role == WarpRole::KvTileTransfer) {
        constexpr int NUM_WARPS = kNumKvTileTransferWarps;
        static_assert((B_TOPK / 2) % (4 * NUM_WARPS) == 0);
        constexpr int NUM_LOCAL_ROWS_PER_WARP = (B_TOPK / 2) / 4 / NUM_WARPS;
        constexpr int KV_PEER_ELEMENTS = (B_TOPK / 2) * D_K;
        const int local_warp_idx = warp_idx - kKvTileTransferFirstWarp;

        CUTE_NO_UNROLL
        for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
            const int phase = k_block & 1;

            if (k_block > 0) {
                plan.bar_dq_ready.wait((k_block - 1) & 1);
            }

            if (elect_one_sync()) {
                bf16* sKV_base = plan.u.q_kv.k_nope.data() + local_warp_idx * 4 * 64;
                int4 indices4[NUM_LOCAL_ROWS_PER_WARP];
                CUTE_UNROLL
                for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                    indices4[local_row] = __ldg(
                        (const int4*)(gIndices_s + k_block * B_TOPK + cta_idx * (B_TOPK / 2)) +
                        local_row * NUM_WARPS + local_warp_idx
                    );
                }

                CUTE_UNROLL
                for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                    CUTE_UNROLL
                    for (int local_col = 0; local_col < D_K / 64; ++local_col) {
                        ku::tma_gather4_cta_group_2<true>(
                            &(tma_params.tensor_map_kv),
                            plan.bar_prologue_kv,
                            sKV_base + local_row * (4 * NUM_WARPS) * 64 + local_col * ((B_TOPK / 2) * 64),
                            local_col * 64,
                            indices4[local_row],
                            (int64_t)TMA::CacheHintSm90::EVICT_LAST
                        );
                    }
                }
            }

            plan.bar_p_ready.wait(phase);
            NamedBarrier::arrive_and_wait(NUM_WARPS * kThreadsPerWarp, 1);

            if (local_warp_idx == 0 && elect_one_sync()) {
                plan.bar_kv_peer_cp_async.arrive_and_expect_tx(sizeof(bf16) * KV_PEER_ELEMENTS);
                bf16* peer_kv_peer_ptr = kerutils::get_peer_addr(plan.u.q_kv.kv_peer.data());
                transac_bar_t* peer_bar_ptr = kerutils::get_peer_addr(&plan.bar_kv_peer_cp_async);
                kerutils::cp_async_bulk_shared_cta_to_shared_cluster(
                    peer_kv_peer_ptr,
                    plan.u.q_kv.k_nope.data(),
                    sizeof(bf16) * KV_PEER_ELEMENTS,
                    *peer_bar_ptr
                );
                fence_view_async_shared();
            }

            NamedBarrier::arrive_and_wait(NUM_WARPS * kThreadsPerWarp, 1);
            plan.bar_kv_peer_cp_async.wait(phase);

            if (local_warp_idx == 0 && elect_one_sync()) {
                plan.bar_kv_peer_ready.arrive(static_cast<uint32_t>(cta_idx));
            }
        }
    }

    if (warp_role == WarpRole::DkvTransfer) {
        // Keep the baseline warp partition for now, but leave these warps idle
        // until the dq-only path is functionally stable.
    }

    if (warp_role == WarpRole::Mma) {
        TiledMMA_P tiled_mma_P{};
        TiledMMA_dP tiled_mma_dP{};
        Tensor tP = partition_fragment_C(tiled_mma_P, Shape<Int<B_H / 2>, Int<B_TOPK>>{});
        Tensor tdP = partition_fragment_C(tiled_mma_dP, Shape<Int<B_H / 2>, Int<B_TOPK>>{});
        tP.data().get() = tmem_cols::P;
        tdP.data().get() = tmem_cols::dP;

        Tensor sV = make_tensor(make_smem_ptr(plan.u.q_kv.k_nope.data()), SmemLayoutV{});

        if (elect_one_sync()) {
            if (cta_idx == 0) {
                plan.bar_prologue_q_nope.arrive_and_expect_tx(B_H * D_V * sizeof(bf16));
                plan.bar_prologue_q_rope.arrive_and_expect_tx(B_H * D_ROPE * sizeof(bf16));
                plan.bar_prologue_dO.arrive_and_expect_tx(B_H * D_V * sizeof(bf16));
                plan.bar_prologue_q_nope.wait(0);
                plan.bar_prologue_q_rope.wait(0);
                plan.bar_prologue_dO.wait(0);
                ku::tcgen05_after_thread_sync();
            }

            TiledMMA_dQ tiled_mma_dQ{};
            TiledMMA_dQ_RoPE tiled_mma_dQ_RoPE{};
            Tensor tdQ_part0 = partition_fragment_C(tiled_mma_dQ, Shape<Int<B_H / 2>, Int<256>>{});
            tdQ_part0.data().get() = tmem_cols::dQ;
            Tensor tdQ_part1 = partition_fragment_C(tiled_mma_dQ, Shape<Int<B_H / 2>, Int<256>>{});
            tdQ_part1.data().get() = tmem_cols::dQ + 128;
            Tensor tdQ_RoPE = partition_fragment_C(tiled_mma_dQ_RoPE, Shape<Int<B_H / 2>, Int<D_ROPE>>{});
            tdQ_RoPE.data().get() = tmem_cols::dQ_RoPE;

            Tensor sDS_t = make_tensor(make_smem_ptr(plan.s_ds.ds.data()), SmemLayoutdSTransposed{});
            auto sDS_t_div = flat_divide(sDS_t, Shape<Int<B_H / 2>, Int<B_TOPK / 2>>{});

            Tensor sK_nope_t_full = make_tensor(make_smem_ptr(plan.u.q_kv.k_nope.data()), SmemLayoutKVTilesTransposed<D_V / 64>{});
            auto sK_nope_t_div = flat_divide(sK_nope_t_full, Shape<Int<256>, Int<B_TOPK / 2>>{});
            Tensor sK_rope_t = make_tensor(make_smem_ptr(plan.u.q_kv.k_rope.data()), SmemLayoutKRoPETransposed{});

            Tensor sK_peer_nope_t_full = make_tensor(make_smem_ptr(plan.u.q_kv.kv_peer.data()), SmemLayoutKVTilesTransposed<D_V / 64>{});
            auto sK_peer_nope_t_div = flat_divide(sK_peer_nope_t_full, Shape<Int<256>, Int<B_TOPK / 2>>{});
            Tensor sK_peer_rope_t = make_tensor(
                make_smem_ptr(plan.u.q_kv.kv_peer.data() + (B_TOPK / 2) * D_V),
                SmemLayoutKRoPETransposed{}
            );

            CUTE_NO_UNROLL
            for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
                const int phase = k_block & 1;
                const bool dq_clear = (k_block == 0);

                if (cta_idx == 0) {
                    plan.bar_prologue_kv.arrive_and_expect_tx(B_TOPK * D_K * sizeof(bf16));
                    plan.bar_prologue_kv.wait(phase);
                    ku::tcgen05_after_thread_sync();
                    ku::utcmma_ss(tiled_mma_P, sQNoPE, sKNoPE, tP, true);
                    ku::utcmma_ss(tiled_mma_P, sQRoPE, sKRoPE, tP, false);
                    ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_p_ready, 1 | 2);
                    ku::tcgen05_after_thread_sync();

                    ku::utcmma_ss(tiled_mma_dP, sdO, sV, tdP, true);
                    ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_dp_ready, 1 | 2);
                    ku::tcgen05_after_thread_sync();
                }

                // dQ consumes dS produced by the softmax warps, so keep the
                // same producer/consumer ordering as the fused baseline.
                plan.bar_ds_ready.wait(phase);
                ku::tcgen05_after_thread_sync();

                if (cta_idx == 0) {
                    ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _0{}), sK_nope_t_div(_, _, _0{}, _0{}), tdQ_part0, dq_clear);
                    plan.bar_kv_peer_ready.wait(phase);
                    ku::tcgen05_after_thread_sync();
                    ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _1{}), sK_peer_nope_t_div(_, _, _0{}, _0{}), tdQ_part0, false);

                    ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _0{}), sK_nope_t_div(_, _, _1{}, _0{}), tdQ_part1, dq_clear);
                    ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _1{}), sK_peer_nope_t_div(_, _, _1{}, _0{}), tdQ_part1, false);

                    ku::utcmma_ss(tiled_mma_dQ_RoPE, sDS_t_div(_, _, _0{}, _0{}), sK_rope_t, tdQ_RoPE, dq_clear);
                    ku::utcmma_ss(tiled_mma_dQ_RoPE, sDS_t_div(_, _, _0{}, _1{}), sK_peer_rope_t, tdQ_RoPE, false);
                } else {
                    ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _1{}), sK_nope_t_div(_, _, _0{}, _0{}), tdQ_part0, dq_clear);
                    plan.bar_kv_peer_ready.wait(phase);
                    ku::tcgen05_after_thread_sync();
                    ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _0{}), sK_peer_nope_t_div(_, _, _0{}, _0{}), tdQ_part0, false);

                    ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _1{}), sK_nope_t_div(_, _, _1{}, _0{}), tdQ_part1, dq_clear);
                    ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _0{}), sK_peer_nope_t_div(_, _, _1{}, _0{}), tdQ_part1, false);

                    ku::utcmma_ss(tiled_mma_dQ_RoPE, sDS_t_div(_, _, _0{}, _1{}), sK_rope_t, tdQ_RoPE, dq_clear);
                    ku::utcmma_ss(tiled_mma_dQ_RoPE, sDS_t_div(_, _, _0{}, _0{}), sK_peer_rope_t, tdQ_RoPE, false);
                }
                ku::umma_arrive_noelect(plan.bar_dq_ready);
                ku::tcgen05_after_thread_sync();
            }
        }
    }

    if (warp_role == WarpRole::KvValidLoad) {
        if (lane_idx < B_TOPK / 8) {
            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks; ++k) {
                int32x8_t indices = ldg_256_indices((void*)(gIndices_s + k * B_TOPK + lane_idx * 8));
                auto is_valid = [&](int rel_idx, int index) -> char {
                    const int topk_idx = k * B_TOPK + lane_idx * 8 + rel_idx;
                    return index >= 0 && index < params.s_kv && index <= max_kv_i && topk_idx < topk_length;
                };
                char is_ks_valid_mask =
                    is_valid(7, indices.a7) << 7 |
                    is_valid(6, indices.a6) << 6 |
                    is_valid(5, indices.a5) << 5 |
                    is_valid(4, indices.a4) << 4 |
                    is_valid(3, indices.a3) << 3 |
                    is_valid(2, indices.a2) << 2 |
                    is_valid(1, indices.a1) << 1 |
                    is_valid(0, indices.a0) << 0;

                plan.bar_k_valid_free.wait((k & 1) ^ 1);
                plan.is_k_valid[lane_idx] = is_ks_valid_mask;
                plan.bar_k_valid_ready.arrive();
            }
        }
    }

    cluster_sync();

    if (warp_idx == 0 && elect_one_sync()) {
        TMEM::Allocator2Sm().free(tmem_base, 512);
    }
#endif
}

static void launch_dq_phase(const SparseAttnBwdParams& params) {
    auto shape_Q_nope = cute::make_shape(B_H, D_V, params.s_q);
    auto tma_Q_nope = cute::make_tma_copy(
        cute::SM100_TMA_2SM_LOAD_NOSPLIT{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)params.q),
            cute::make_layout(
                shape_Q_nope,
                cute::make_stride(params.stride_q_h_q, cute::_1{}, params.stride_q_s_q)
            )
        ),
        SmemLayoutQNoPE{}
    );

    auto shape_Q_rope = cute::make_shape(B_H, D_ROPE, params.s_q);
    auto tma_Q_rope = cute::make_tma_copy(
        cute::SM100_TMA_2SM_LOAD_NOSPLIT{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)params.q + D_V),
            cute::make_layout(
                shape_Q_rope,
                cute::make_stride(params.stride_q_h_q, cute::_1{}, params.stride_q_s_q)
            )
        ),
        SmemLayoutQRoPE{}
    );

    auto shape_dO = cute::make_shape(B_H, D_V, params.s_q);
    auto tma_dO = cute::make_tma_copy(
        cute::SM100_TMA_2SM_LOAD_NOSPLIT{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)params.dO),
            cute::make_layout(
                shape_dO,
                cute::make_stride(params.stride_dO_h_q, cute::_1{}, params.stride_dO_s_q)
            )
        ),
        SmemLayoutdO{}
    );

    auto shape_dQ = cute::make_shape(B_H, D_Q, params.s_q);
    auto tma_dQ = cute::make_tma_copy(
        cute::SM90_TMA_STORE{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)params.dQ),
            cute::make_layout(
                shape_dQ,
                cute::make_stride(params.stride_dQ_h_q, cute::_1{}, params.stride_dQ_s_q)
            )
        ),
        SmemLayoutQ{}
    );

    auto shape_S = cute::make_shape(B_H, params.topk, params.s_q);
    auto tma_S = cute::make_tma_copy(
        cute::SM90_TMA_STORE{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)params.s),
            cute::make_layout(
                shape_S,
                cute::make_stride(params.stride_s_h_q, cute::_1{}, params.stride_s_s_q)
            )
        ),
        SmemLayoutS{}
    );

    auto shape_dS = cute::make_shape(B_H, params.topk, params.s_q);
    auto tma_dS = cute::make_tma_copy(
        cute::SM90_TMA_STORE{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)params.ds),
            cute::make_layout(
                shape_dS,
                cute::make_stride(params.stride_ds_h_q, cute::_1{}, params.stride_ds_s_q)
            )
        ),
        SmemLayoutdS{}
    );

    CUtensorMap tensor_map_kv;
    {
        uint64_t size[2] = {(uint64_t)D_K, (unsigned long)params.s_kv};
        uint64_t stride[1] = {(uint64_t)params.stride_kv_s_kv * sizeof(bf16)};
        uint32_t box_size[2] = {64, 1};
        uint32_t elem_stride[2] = {1, 1};
        CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &tensor_map_kv,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            const_cast<bf16*>(params.kv),
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

    using KernelTmaParams = TmaParams<
        decltype(shape_Q_nope), decltype(tma_Q_nope),
        decltype(shape_Q_rope), decltype(tma_Q_rope),
        decltype(shape_dO), decltype(tma_dO),
        decltype(shape_dQ), decltype(tma_dQ),
        decltype(shape_S), decltype(tma_S),
        decltype(shape_dS), decltype(tma_dS)
    >;

    KernelTmaParams tma_params = {
        shape_Q_nope, tma_Q_nope,
        shape_Q_rope, tma_Q_rope,
        shape_dO, tma_dO,
        shape_dQ, tma_dQ,
        shape_S, tma_S,
        shape_dS, tma_dS,
        tensor_map_kv
    };

    auto kernel = &dq_phase_kernel<KernelTmaParams>;
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
void run_bwd_dq_phase_kernel(const SparseAttnBwdParams& params) {
    static_assert(DQK == D_QK);

    KU_ASSERT(params.d_qk == DQK);
    KU_ASSERT(params.d_v == D_V);
    KU_ASSERT(params.h_q == B_H);
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.topk > 0 && params.topk % B_TOPK == 0);

    sm100::bwd::head128::run_bwd_preprocess_delta_kernel<DQK>(params);
    launch_dq_phase(params);
}

}  // namespace sm100::bwd::head128_2kernels::dq
