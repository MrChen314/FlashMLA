#pragma once

#include "dkv_config.h"

#include <cstring>
#include <cstdio>
#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
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

CUTE_DEVICE
void atomic_add_64floats_unrolled(float* dst, const float* src) {
    atomic_add_32floats_unrolled(dst, src);
    atomic_add_32floats_unrolled(dst + 32, src + 32);
}

static constexpr int kThreadsPerWarp = 32;
static constexpr int kWarpsPerWarpgroup = 4;
static constexpr int kThreadsPerWarpgroup = kWarpsPerWarpgroup * kThreadsPerWarp;
static constexpr int kNumWarpgroups = 3;
static constexpr uint16_t kClusterMask2Cta = 0x3;

static_assert(NUM_THREADS == kNumWarpgroups * kThreadsPerWarpgroup, "NUM_THREADS must match the dKV warpgroup layout.");
// WG0/WG1 drain dKV to global memory; WG2 uses warp_idx 8/9/10/11 as MMA, S-TMA, dS-TMA, idle.

#ifndef FLASHMLA_DKV_PHASE_DEBUG
#define FLASHMLA_DKV_PHASE_DEBUG 1
#endif

#define DKV_DBG_PRINT(enabled, fmt, ...)                                                                                   \
    do {                                                                                                                   \
        if (FLASHMLA_DKV_PHASE_DEBUG && (enabled)) {                                                                       \
            printf(                                                                                                        \
                "[DKVDBG][B%d SQ%d CTA%d W%d WG%d L%d] " fmt "\n",                                                        \
                static_cast<int>(blockIdx.x),                                                                              \
                s_q_idx,                                                                                                   \
                cta_idx,                                                                                                   \
                warp_idx,                                                                                                  \
                warpgroup_idx,                                                                                             \
                lane_idx,                                                                                                  \
                ##__VA_ARGS__                                                                                              \
            );                                                                                                             \
        }                                                                                                                  \
    } while (0)

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
    const int tid = threadIdx.x;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = tid % kThreadsPerWarp;
    const int warpgroup_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
    const int local_warp_idx = warp_idx % kWarpsPerWarpgroup;
    if (s_q_idx >= params.s_q) {
        return;
    }

    const int max_kv_i = params.q_start_index_s + s_q_idx;
    const int topk_length = params.topk_length == nullptr ?
        params.topk :
        min(max(__ldg(params.topk_length + s_q_idx), 0), params.topk);
    const int num_k_blocks = max(params.topk / DKV_TILE_M, 1);
    const int* gIndices_s = params.indices + (int64_t)s_q_idx * params.stride_indices_s_q;
    const bool dbg_block = blockIdx.x < 2;
    const bool dbg_tid0 = dbg_block && tid == 0;
    const bool dbg_warp0 = dbg_block && warp_idx == 0;
    const bool dbg_s_tma = dbg_block && warp_idx == 9;
    const bool dbg_ds_tma = dbg_block && warp_idx == 10;
    const bool dbg_drain = dbg_block && warpgroup_idx < 2 && local_warp_idx == 0 && lane_idx == 0;
    const bool dbg_mma = dbg_block && cta_idx == 0 && warp_idx == 8;

    DKV_DBG_PRINT(
        dbg_tid0,
        "enter topk=%d topk_length=%d num_k_blocks=%d max_kv_i=%d",
        params.topk,
        topk_length,
        num_k_blocks,
        max_kv_i
    );

    if (tid == 0) {
        cute::prefetch_tma_descriptor(tma_params.tma_Q_nope.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_Q_rope.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dO.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_S.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dS.get_tma_descriptor());
        DKV_DBG_PRINT(dbg_tid0, "prefetch_tma_descriptors_done");
    }

    if (warp_idx == 0 && elect_one_sync()) {
        DKV_DBG_PRINT(dbg_warp0, "barrier_init_begin");
        plan.bar_q_nope_ready.init(1);
        plan.bar_q_rope_ready.init(1);
        plan.bar_dO_ready.init(1);
        CUTE_UNROLL
        for (int buf = 0; buf < NUM_S_DS_BUFS; ++buf) {
            plan.bar_s_ready[buf].init(1);
            plan.bar_ds_ready[buf].init(1);
            plan.bar_dkv_part0_ready[buf].init(1);
            plan.bar_dkv_rope_ready[buf].init(1);
            plan.bar_dkv_part1_ready[buf].init(1);
            plan.bar_dkv_part2_ready[buf].init(1);
        }
        plan.bar_dkv_part0_done.init(2 * kThreadsPerWarpgroup);
        plan.bar_dkv_rope_done.init(2 * kThreadsPerWarpgroup);
        plan.bar_dkv_part1_done.init(2 * kThreadsPerWarpgroup);
        plan.bar_dkv_part2_done.init(2 * kThreadsPerWarpgroup);
        fence_barrier_init();
        DKV_DBG_PRINT(dbg_warp0, "barrier_init_done");
    }

    DKV_DBG_PRINT(dbg_tid0, "cluster_sync_init_before");
    cluster_sync();
    DKV_DBG_PRINT(dbg_tid0, "cluster_sync_init_after");

    Tensor sQNoPE = make_tensor(make_smem_ptr(plan.q_nope.data()), SmemLayoutQNoPE{});
    Tensor sQRoPE = make_tensor(make_smem_ptr(plan.q_rope.data()), SmemLayoutQRoPE{});
    Tensor sdO = make_tensor(make_smem_ptr(plan.dO.data()), SmemLayoutdO{});
    if (warp_idx == 0) {
        if (elect_one_sync()) {
            DKV_DBG_PRINT(dbg_warp0, "launch_qdo_tma_begin");
            Tensor gQNoPE = tma_params.tma_Q_nope.get_tma_tensor(tma_params.shape_Q_nope)(_, _, cta_idx, s_q_idx);
            ku::launch_tma_copy(
                tma_params.tma_Q_nope,
                gQNoPE,
                sQNoPE,
                plan.bar_q_nope_ready,
                TMA::CacheHintSm90::EVICT_FIRST
            );

            Tensor gQRoPE = tma_params.tma_Q_rope.get_tma_tensor(tma_params.shape_Q_rope)(_, _, cta_idx, s_q_idx);
            ku::launch_tma_copy(
                tma_params.tma_Q_rope,
                gQRoPE,
                sQRoPE,
                plan.bar_q_rope_ready,
                TMA::CacheHintSm90::EVICT_FIRST
            );

            Tensor gdO = tma_params.tma_dO.get_tma_tensor(tma_params.shape_dO)(_, _, cta_idx, s_q_idx);
            ku::launch_tma_copy(
                tma_params.tma_dO,
                gdO,
                sdO,
                plan.bar_dO_ready,
                TMA::CacheHintSm90::EVICT_FIRST
            );
            DKV_DBG_PRINT(dbg_warp0, "launch_qdo_tma_done");
        }

        TMEM::Allocator2Sm().allocate(512, plan.tmem_start_addr.data());
        KU_TRAP_ONLY_DEVICE_ASSERT(plan.tmem_start_addr.data()[0] == 0);
        TMEM::Allocator2Sm().release_allocation_lock();
        DKV_DBG_PRINT(dbg_tid0, "tmem_allocate_done");
    }
    DKV_DBG_PRINT(dbg_tid0, "__syncthreads_before");
    __syncthreads();

    const uint32_t tmem_base = plan.tmem_start_addr.data()[0];
    DKV_DBG_PRINT(dbg_tid0, "__syncthreads_after tmem_base=%u", tmem_base);

    if (warp_idx == 9) {
        const bool issue_s_tma = elect_one_sync();
        DKV_DBG_PRINT(dbg_s_tma && issue_s_tma, "S-TMA enter");
        CUTE_NO_UNROLL
        for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
            if (issue_s_tma) {
                const int buf = k_block % NUM_S_DS_BUFS;
                const int phase = (k_block / NUM_S_DS_BUFS) & 1;
                if (k_block >= NUM_S_DS_BUFS) {
                    DKV_DBG_PRINT(
                        dbg_s_tma,
                        "S-TMA k=%d wait part2_ready buf=%d phase=%d wait_phase=%d",
                        k_block,
                        buf,
                        phase,
                        phase ^ 1
                    );
                    plan.bar_dkv_part2_ready[buf].wait(phase ^ 1);
                    ku::tcgen05_after_thread_sync();
                    DKV_DBG_PRINT(dbg_s_tma, "S-TMA k=%d wait_done part2_ready", k_block);
                }

                Tensor sS = make_tensor(make_smem_ptr(plan.s_ds.s[buf].data()), SmemLayoutS{});
                Tensor gS = tma_params.tma_S.get_tma_tensor(tma_params.shape_S)(_, _, cta_idx, k_block, s_q_idx);
                DKV_DBG_PRINT(dbg_s_tma, "S-TMA k=%d launch buf=%d phase=%d", k_block, buf, phase);
                ku::launch_tma_copy(
                    tma_params.tma_S,
                    gS,
                    sS,
                    plan.bar_s_ready[buf],
                    TMA::CacheHintSm90::EVICT_FIRST
                );
                DKV_DBG_PRINT(dbg_s_tma, "S-TMA k=%d launch_done", k_block);
            }
        }
        DKV_DBG_PRINT(dbg_s_tma && issue_s_tma, "S-TMA exit");
    } else if (warp_idx == 10) {
        const bool issue_ds_tma = elect_one_sync();
        DKV_DBG_PRINT(dbg_ds_tma && issue_ds_tma, "dS-TMA enter");
        CUTE_NO_UNROLL
        for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
            if (issue_ds_tma) {
                const int buf = k_block % NUM_S_DS_BUFS;
                const int phase = (k_block / NUM_S_DS_BUFS) & 1;
                if (k_block >= NUM_S_DS_BUFS) {
                    DKV_DBG_PRINT(
                        dbg_ds_tma,
                        "dS-TMA k=%d wait part2_ready buf=%d phase=%d wait_phase=%d",
                        k_block,
                        buf,
                        phase,
                        phase ^ 1
                    );
                    plan.bar_dkv_part2_ready[buf].wait(phase ^ 1);
                    ku::tcgen05_after_thread_sync();
                    DKV_DBG_PRINT(dbg_ds_tma, "dS-TMA k=%d wait_done part2_ready", k_block);
                }

                Tensor sDS = make_tensor(make_smem_ptr(plan.s_ds.ds[buf].data()), SmemLayoutdS{});
                Tensor gdS = tma_params.tma_dS.get_tma_tensor(tma_params.shape_dS)(_, _, cta_idx, k_block, s_q_idx);
                DKV_DBG_PRINT(dbg_ds_tma, "dS-TMA k=%d launch buf=%d phase=%d", k_block, buf, phase);
                ku::launch_tma_copy(
                    tma_params.tma_dS,
                    gdS,
                    sDS,
                    plan.bar_ds_ready[buf],
                    TMA::CacheHintSm90::EVICT_FIRST
                );
                DKV_DBG_PRINT(dbg_ds_tma, "dS-TMA k=%d launch_done", k_block);
            }
        }
        DKV_DBG_PRINT(dbg_ds_tma && issue_ds_tma, "dS-TMA exit");
    } else {
        if (warpgroup_idx < 2) {
            const int row = local_warp_idx * kThreadsPerWarp + lane_idx;
            const int chunk_group = warpgroup_idx;
            constexpr int kNumTransferWarpgroups = 2;
            constexpr int PART0_CHUNK_SIZE = 64;
            constexpr int PART0_NUM_CHUNKS = 256 / PART0_CHUNK_SIZE;
            constexpr int PART0_CHUNKS_PER_GROUP = PART0_NUM_CHUNKS / kNumTransferWarpgroups;
            constexpr int PART12_CHUNK_SIZE = 64;
            constexpr int PART12_NUM_CHUNKS = 128 / PART12_CHUNK_SIZE;
            constexpr int ROPE_CHUNK_SIZE = 32;
            constexpr int ROPE_NUM_CHUNKS = D_ROPE / ROPE_CHUNK_SIZE;
            static_assert(DKV_ROWS_PER_CTA == kThreadsPerWarpgroup);
            static_assert(PART0_NUM_CHUNKS == 4);
            static_assert(PART0_CHUNKS_PER_GROUP == 2);
            static_assert(PART12_NUM_CHUNKS == 2);
            static_assert(ROPE_NUM_CHUNKS == 2);
            DKV_DBG_PRINT(dbg_drain, "drain enter row=%d chunk_group=%d", row, chunk_group);

            CUTE_NO_UNROLL
            for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
                const int buf = k_block % NUM_S_DS_BUFS;
                const int phase = (k_block / NUM_S_DS_BUFS) & 1;
                const int row_global = k_block * DKV_TILE_M + cta_idx * DKV_ROWS_PER_CTA + row;
                int kv_idx = -1;
                if (row_global < topk_length) {
                    kv_idx = __ldg(gIndices_s + row_global);
                }
                const bool row_valid = kv_idx >= 0 && kv_idx < params.s_kv && kv_idx <= max_kv_i;

                DKV_DBG_PRINT(
                    dbg_drain,
                    "drain k=%d wait part0_ready buf=%d phase=%d row_global=%d row_valid=%d kv_idx=%d",
                    k_block,
                    buf,
                    phase,
                    row_global,
                    static_cast<int>(row_valid),
                    kv_idx
                );
                plan.bar_dkv_part0_ready[buf].wait(phase);
                ku::tcgen05_after_thread_sync();
                DKV_DBG_PRINT(dbg_drain, "drain k=%d wait_done part0_ready", k_block);
                CUTE_UNROLL
                for (int local_chunk = 0; local_chunk < PART0_CHUNKS_PER_GROUP; ++local_chunk) {
                    const int chunk = chunk_group * PART0_CHUNKS_PER_GROUP + local_chunk;
                    float2 dkv_data[PART0_CHUNK_SIZE / 2];
                    ku::tmem_ld_32dp32bNx<PART0_CHUNK_SIZE>(tmem_cols::dKV_part0 + chunk * PART0_CHUNK_SIZE, dkv_data);
                    cutlass::arch::fence_view_async_tmem_load();
                    ku::tcgen05_before_thread_sync();

                    if (row_valid) {
                        float* dst = params.dKV + (int64_t)kv_idx * params.stride_dKV_s_kv +
                            chunk * PART0_CHUNK_SIZE;
                        atomic_add_64floats_unrolled(dst, reinterpret_cast<float*>(dkv_data));
                    }
                }
                plan.bar_dkv_part0_done.arrive(static_cast<uint32_t>(0));
                DKV_DBG_PRINT(dbg_drain, "drain k=%d arrive part0_done", k_block);

                DKV_DBG_PRINT(dbg_drain, "drain k=%d wait rope_ready buf=%d phase=%d", k_block, buf, phase);
                plan.bar_dkv_rope_ready[buf].wait(phase);
                ku::tcgen05_after_thread_sync();
                DKV_DBG_PRINT(dbg_drain, "drain k=%d wait_done rope_ready", k_block);
                {
                    float2 dkv_rope_data[ROPE_CHUNK_SIZE / 2];
                    ku::tmem_ld_32dp32bNx<ROPE_CHUNK_SIZE>(tmem_cols::dKV_RoPE + chunk_group * ROPE_CHUNK_SIZE, dkv_rope_data);
                    cutlass::arch::fence_view_async_tmem_load();
                    ku::tcgen05_before_thread_sync();

                    if (row_valid) {
                        float* dst = params.dKV + (int64_t)kv_idx * params.stride_dKV_s_kv +
                            D_V + chunk_group * ROPE_CHUNK_SIZE;
                        atomic_add_32floats_unrolled(dst, reinterpret_cast<float*>(dkv_rope_data));
                    }
                }
                plan.bar_dkv_rope_done.arrive(static_cast<uint32_t>(0));
                DKV_DBG_PRINT(dbg_drain, "drain k=%d arrive rope_done", k_block);

                DKV_DBG_PRINT(dbg_drain, "drain k=%d wait part1_ready buf=%d phase=%d", k_block, buf, phase);
                plan.bar_dkv_part1_ready[buf].wait(phase);
                ku::tcgen05_after_thread_sync();
                DKV_DBG_PRINT(dbg_drain, "drain k=%d wait_done part1_ready", k_block);
                {
                    float2 dkv_part1_data[PART12_CHUNK_SIZE / 2];
                    ku::tmem_ld_32dp32bNx<PART12_CHUNK_SIZE>(tmem_cols::dKV_part1 + chunk_group * PART12_CHUNK_SIZE, dkv_part1_data);
                    cutlass::arch::fence_view_async_tmem_load();
                    ku::tcgen05_before_thread_sync();

                    if (row_valid) {
                        float* dst = params.dKV + (int64_t)kv_idx * params.stride_dKV_s_kv +
                            256 + chunk_group * PART12_CHUNK_SIZE;
                        atomic_add_64floats_unrolled(dst, reinterpret_cast<float*>(dkv_part1_data));
                    }
                }
                plan.bar_dkv_part1_done.arrive(static_cast<uint32_t>(0));
                DKV_DBG_PRINT(dbg_drain, "drain k=%d arrive part1_done", k_block);

                DKV_DBG_PRINT(dbg_drain, "drain k=%d wait part2_ready buf=%d phase=%d", k_block, buf, phase);
                plan.bar_dkv_part2_ready[buf].wait(phase);
                ku::tcgen05_after_thread_sync();
                DKV_DBG_PRINT(dbg_drain, "drain k=%d wait_done part2_ready", k_block);
                {
                    float2 dkv_part2_data[PART12_CHUNK_SIZE / 2];
                    ku::tmem_ld_32dp32bNx<PART12_CHUNK_SIZE>(tmem_cols::dKV_part2 + chunk_group * PART12_CHUNK_SIZE, dkv_part2_data);
                    cutlass::arch::fence_view_async_tmem_load();
                    ku::tcgen05_before_thread_sync();

                    if (row_valid) {
                        float* dst = params.dKV + (int64_t)kv_idx * params.stride_dKV_s_kv +
                            384 + chunk_group * PART12_CHUNK_SIZE;
                        atomic_add_64floats_unrolled(dst, reinterpret_cast<float*>(dkv_part2_data));
                    }
                }
                plan.bar_dkv_part2_done.arrive(static_cast<uint32_t>(0));
                DKV_DBG_PRINT(dbg_drain, "drain k=%d arrive part2_done", k_block);
            }
            DKV_DBG_PRINT(dbg_drain, "drain exit");
        }

        if (cta_idx == 0 && warp_idx == 8 && elect_one_sync()) {
            DKV_DBG_PRINT(dbg_mma, "MMA enter");
            plan.bar_q_nope_ready.arrive_and_expect_tx(B_H * D_V * sizeof(bf16));
            plan.bar_q_rope_ready.arrive_and_expect_tx(B_H * D_ROPE * sizeof(bf16));
            plan.bar_dO_ready.arrive_and_expect_tx(B_H * D_V * sizeof(bf16));
            DKV_DBG_PRINT(dbg_mma, "MMA q/do arrive_and_expect_tx done");

            DKV_DBG_PRINT(dbg_mma, "MMA wait q_nope_ready phase=0");
            plan.bar_q_nope_ready.wait(0);
            DKV_DBG_PRINT(dbg_mma, "MMA wait_done q_nope_ready");
            DKV_DBG_PRINT(dbg_mma, "MMA wait q_rope_ready phase=0");
            plan.bar_q_rope_ready.wait(0);
            DKV_DBG_PRINT(dbg_mma, "MMA wait_done q_rope_ready");
            DKV_DBG_PRINT(dbg_mma, "MMA wait dO_ready phase=0");
            plan.bar_dO_ready.wait(0);
            ku::tcgen05_after_thread_sync();
            DKV_DBG_PRINT(dbg_mma, "MMA wait_done dO_ready");

            TiledMMA_dKV_Part0 tiled_mma_dKV_part0{};
            TiledMMA_dKV_Part1_2 tiled_mma_dKV_part1_2{};
            TiledMMA_dKV_RoPE tiled_mma_dKV_RoPE{};
            Tensor tdKV_part0 = partition_fragment_C(tiled_mma_dKV_part0, Shape<Int<DKV_ROWS_PER_CTA>, Int<256>>{});
            Tensor tdKV_part1 = partition_fragment_C(tiled_mma_dKV_part1_2, Shape<Int<DKV_ROWS_PER_CTA>, Int<128>>{});
            Tensor tdKV_part2 = partition_fragment_C(tiled_mma_dKV_part1_2, Shape<Int<DKV_ROWS_PER_CTA>, Int<128>>{});
            Tensor tdKV_RoPE = partition_fragment_C(tiled_mma_dKV_RoPE, Shape<Int<DKV_ROWS_PER_CTA>, Int<D_ROPE>>{});
            tdKV_part0.data().get() = tmem_cols::dKV_part0;
            tdKV_part1.data().get() = tmem_cols::dKV_part1;
            tdKV_part2.data().get() = tmem_cols::dKV_part2;
            tdKV_RoPE.data().get() = tmem_cols::dKV_RoPE;

            Tensor sdO_mma_full = make_tensor(make_smem_ptr(plan.dO.data()), SmemLayoutdO_MMA{});
            Tensor sQNoPE_mma_full = make_tensor(make_smem_ptr(plan.q_nope.data()), SmemLayoutQNoPE_MMA{});
            Tensor sQRoPE_mma_full = make_tensor(make_smem_ptr(plan.q_rope.data()), SmemLayoutQRoPE_MMA{});
            auto sdO_mma_halves = flat_divide(sdO_mma_full, Shape<Int<128>, Int<B_H>>{});
            auto sQNoPE_mma_halves = flat_divide(sQNoPE_mma_full, Shape<Int<128>, Int<B_H>>{});

            CUTE_NO_UNROLL
            for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
                const int buf = k_block % NUM_S_DS_BUFS;
                const int phase = (k_block / NUM_S_DS_BUFS) & 1;
                const int round_phase = k_block & 1;

                Tensor sS_mma = make_tensor(make_smem_ptr(plan.s_ds.s[buf].data()), SmemLayoutS_MMA{});
                Tensor sDS_mma = make_tensor(make_smem_ptr(plan.s_ds.ds[buf].data()), SmemLayoutdS_MMA{});
                plan.bar_s_ready[buf].arrive_and_expect_tx(B_H * DKV_TILE_M * sizeof(bf16));
                DKV_DBG_PRINT(dbg_mma, "MMA k=%d arrive_and_expect_tx s_ready buf=%d phase=%d", k_block, buf, phase);

                if (k_block > 0) {
                    DKV_DBG_PRINT(dbg_mma, "MMA k=%d wait part0_done round_phase=%d", k_block, round_phase ^ 1);
                    plan.bar_dkv_part0_done.wait(round_phase ^ 1);
                    ku::tcgen05_after_thread_sync();
                    DKV_DBG_PRINT(dbg_mma, "MMA k=%d wait_done part0_done", k_block);
                }

                DKV_DBG_PRINT(dbg_mma, "MMA k=%d wait s_ready phase=%d", k_block, phase);
                plan.bar_s_ready[buf].wait(phase);
                ku::tcgen05_after_thread_sync();
                DKV_DBG_PRINT(dbg_mma, "MMA k=%d wait_done s_ready", k_block);
                ku::utcmma_ss(tiled_mma_dKV_part0, sS_mma, sdO_mma_full, tdKV_part0, true);

                plan.bar_ds_ready[buf].arrive_and_expect_tx(B_H * DKV_TILE_M * sizeof(bf16));
                DKV_DBG_PRINT(dbg_mma, "MMA k=%d arrive_and_expect_tx ds_ready", k_block);
                DKV_DBG_PRINT(dbg_mma, "MMA k=%d wait ds_ready phase=%d", k_block, phase);
                plan.bar_ds_ready[buf].wait(phase);
                ku::tcgen05_after_thread_sync();
                DKV_DBG_PRINT(dbg_mma, "MMA k=%d wait_done ds_ready", k_block);
                ku::utcmma_ss(tiled_mma_dKV_part0, sDS_mma, sQNoPE_mma_full, tdKV_part0, false);
                ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_dkv_part0_ready[buf], kClusterMask2Cta);
                ku::tcgen05_after_thread_sync();
                DKV_DBG_PRINT(dbg_mma, "MMA k=%d arrive part0_ready", k_block);

                if (k_block > 0) {
                    DKV_DBG_PRINT(dbg_mma, "MMA k=%d wait part2_done round_phase=%d", k_block, round_phase ^ 1);
                    plan.bar_dkv_part2_done.wait(round_phase ^ 1);
                    ku::tcgen05_after_thread_sync();
                    DKV_DBG_PRINT(dbg_mma, "MMA k=%d wait_done part2_done", k_block);
                }

                ku::utcmma_ss(tiled_mma_dKV_RoPE, sDS_mma, sQRoPE_mma_full, tdKV_RoPE, true);
                ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_dkv_rope_ready[buf], kClusterMask2Cta);
                ku::tcgen05_after_thread_sync();
                DKV_DBG_PRINT(dbg_mma, "MMA k=%d arrive rope_ready", k_block);

                if (k_block > 0) {
                    DKV_DBG_PRINT(dbg_mma, "MMA k=%d wait part1_done round_phase=%d", k_block, round_phase ^ 1);
                    plan.bar_dkv_part1_done.wait(round_phase ^ 1);
                    ku::tcgen05_after_thread_sync();
                    DKV_DBG_PRINT(dbg_mma, "MMA k=%d wait_done part1_done", k_block);
                }

                ku::utcmma_ss(
                    tiled_mma_dKV_part1_2,
                    sS_mma,
                    sdO_mma_halves(_, _, _0{}, _0{}),
                    tdKV_part1,
                    true
                );
                ku::utcmma_ss(
                    tiled_mma_dKV_part1_2,
                    sDS_mma,
                    sQNoPE_mma_halves(_, _, _0{}, _0{}),
                    tdKV_part1,
                    false
                );
                ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_dkv_part1_ready[buf], kClusterMask2Cta);
                ku::tcgen05_after_thread_sync();
                DKV_DBG_PRINT(dbg_mma, "MMA k=%d arrive part1_ready", k_block);

                DKV_DBG_PRINT(dbg_mma, "MMA k=%d wait rope_done round_phase=%d", k_block, round_phase);
                plan.bar_dkv_rope_done.wait(round_phase);
                ku::tcgen05_after_thread_sync();
                DKV_DBG_PRINT(dbg_mma, "MMA k=%d wait_done rope_done", k_block);

                ku::utcmma_ss(
                    tiled_mma_dKV_part1_2,
                    sS_mma,
                    sdO_mma_halves(_, _, _1{}, _0{}),
                    tdKV_part2,
                    true
                );
                ku::utcmma_ss(
                    tiled_mma_dKV_part1_2,
                    sDS_mma,
                    sQNoPE_mma_halves(_, _, _1{}, _0{}),
                    tdKV_part2,
                    false
                );
                ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_dkv_part2_ready[buf], kClusterMask2Cta);
                ku::tcgen05_after_thread_sync();
                DKV_DBG_PRINT(dbg_mma, "MMA k=%d arrive part2_ready", k_block);
            }

            if (num_k_blocks > 0) {
                const int final_phase = (num_k_blocks - 1) & 1;
                DKV_DBG_PRINT(dbg_mma, "MMA final wait part0_done phase=%d", final_phase);
                plan.bar_dkv_part0_done.wait(final_phase);
                ku::tcgen05_after_thread_sync();
                DKV_DBG_PRINT(dbg_mma, "MMA final wait_done part0_done");
                DKV_DBG_PRINT(dbg_mma, "MMA final wait part1_done phase=%d", final_phase);
                plan.bar_dkv_part1_done.wait(final_phase);
                ku::tcgen05_after_thread_sync();
                DKV_DBG_PRINT(dbg_mma, "MMA final wait_done part1_done");
                DKV_DBG_PRINT(dbg_mma, "MMA final wait part2_done phase=%d", final_phase);
                plan.bar_dkv_part2_done.wait(final_phase);
                ku::tcgen05_after_thread_sync();
                DKV_DBG_PRINT(dbg_mma, "MMA final wait_done part2_done");
            }
            DKV_DBG_PRINT(dbg_mma, "MMA exit");
        }
    }

    DKV_DBG_PRINT(dbg_tid0, "cluster_sync_exit_before");
    cluster_sync();
    DKV_DBG_PRINT(dbg_tid0, "cluster_sync_exit_after");
    if (warp_idx == 0) {
        TMEM::Allocator2Sm().free(tmem_base, 512);
        DKV_DBG_PRINT(dbg_tid0, "tmem_free_done");
    }
#endif
}

static void launch_dkv_phase(const SparseAttnBwdParams& params) {
    const int num_k_blocks = max(params.topk / DKV_TILE_M, 1);

    auto shape_Q_nope = cute::make_shape(B_H, NOPE_COLS_PER_CTA, 2, params.s_q);
    auto tma_Q_nope = cute::make_tma_copy(
        cute::SM100_TMA_2SM_LOAD_NOSPLIT{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)params.q),
            cute::make_layout(
                shape_Q_nope,
                cute::make_stride(
                    params.stride_q_h_q,
                    cute::_1{},
                    Int<NOPE_COLS_PER_CTA>{},
                    params.stride_q_s_q
                )
            )
        ),
        SmemLayoutQNoPE{}
    );

    auto shape_Q_rope = cute::make_shape(B_H, ROPE_COLS_PER_CTA, 2, params.s_q);
    auto tma_Q_rope = cute::make_tma_copy(
        cute::SM100_TMA_2SM_LOAD_NOSPLIT{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)params.q + D_V),
            cute::make_layout(
                shape_Q_rope,
                cute::make_stride(
                    params.stride_q_h_q,
                    cute::_1{},
                    Int<ROPE_COLS_PER_CTA>{},
                    params.stride_q_s_q
                )
            )
        ),
        SmemLayoutQRoPE{}
    );

    auto shape_dO = cute::make_shape(B_H, NOPE_COLS_PER_CTA, 2, params.s_q);
    auto tma_dO = cute::make_tma_copy(
        cute::SM100_TMA_2SM_LOAD_NOSPLIT{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)params.dO),
            cute::make_layout(
                shape_dO,
                cute::make_stride(
                    params.stride_dO_h_q,
                    cute::_1{},
                    Int<NOPE_COLS_PER_CTA>{},
                    params.stride_dO_s_q
                )
            )
        ),
        SmemLayoutdO{}
    );

    auto shape_S = cute::make_shape(B_H, DKV_ROWS_PER_CTA, 2, num_k_blocks, params.s_q);
    auto tma_S = cute::make_tma_copy(
        cute::SM100_TMA_2SM_LOAD_NOSPLIT{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)params.s),
            cute::make_layout(
                shape_S,
                cute::make_stride(
                    params.stride_s_h_q,
                    cute::_1{},
                    Int<DKV_ROWS_PER_CTA>{},
                    Int<DKV_TILE_M>{},
                    params.stride_s_s_q
                )
            )
        ),
        SmemLayoutS{}
    );

    auto shape_dS = cute::make_shape(B_H, DKV_ROWS_PER_CTA, 2, num_k_blocks, params.s_q);
    auto tma_dS = cute::make_tma_copy(
        cute::SM100_TMA_2SM_LOAD_NOSPLIT{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)params.ds),
            cute::make_layout(
                shape_dS,
                cute::make_stride(
                    params.stride_ds_h_q,
                    cute::_1{},
                    Int<DKV_ROWS_PER_CTA>{},
                    Int<DKV_TILE_M>{},
                    params.stride_ds_s_q
                )
            )
        ),
        SmemLayoutdS{}
    );

    using KernelTmaParams = TmaParams<
        decltype(shape_Q_nope), decltype(tma_Q_nope),
        decltype(shape_Q_rope), decltype(tma_Q_rope),
        decltype(shape_dO), decltype(tma_dO),
        decltype(shape_S), decltype(tma_S),
        decltype(shape_dS), decltype(tma_dS)
    >;

    KernelTmaParams tma_params = {
        shape_Q_nope, tma_Q_nope,
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
    KU_ASSERT(params.topk >= DKV_TILE_M, "dKV two-kernel path requires topk >= %d, got %d", DKV_TILE_M, params.topk);
    KU_ASSERT(params.topk % DKV_TILE_M == 0, "dKV two-kernel path requires topk to be a multiple of %d, got %d", DKV_TILE_M, params.topk);
    KU_ASSERT(params.q != nullptr && params.dO != nullptr);
    KU_ASSERT(params.s != nullptr && params.ds != nullptr);
    KU_ASSERT(params.dKV != nullptr);

    launch_dkv_phase(params);
}

}  // namespace sm100::bwd::head128_2kernels::dkv

#undef DKV_DBG_PRINT
