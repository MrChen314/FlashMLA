#pragma once

#include <cuda.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/cluster_launch.hpp>
#include <kerutils/kerutils.cuh>

#include "params.h"
#include "defines.h"

namespace sm100::bwd::head128_2kernels::dq {

using namespace cute;

template<
    typename Shape_QNoPE, typename TMA_QNoPE,
    typename Shape_QRoPE, typename TMA_QRoPE,
    typename Shape_dO, typename TMA_dO,
    typename Shape_dQ, typename TMA_dQ,
    typename Shape_S, typename TMA_S,
    typename Shape_dS, typename TMA_dS
>
struct TmaParams {
    Shape_QNoPE shape_Q_nope;
    TMA_QNoPE tma_Q_nope;
    Shape_QRoPE shape_Q_rope;
    TMA_QRoPE tma_Q_rope;
    Shape_dO shape_dO;
    TMA_dO tma_dO;
    Shape_dQ shape_dQ;
    TMA_dQ tma_dQ;
    Shape_S shape_S;
    TMA_S tma_S;
    Shape_dS shape_dS;
    TMA_dS tma_dS;
    CUtensorMap tensor_map_kv;
    CUtensorMap tensor_map_kv_nope;
    CUtensorMap tensor_map_kv_rope;
};

static constexpr int D_QK = 576;
static constexpr int D_Q = D_QK;
static constexpr int D_K = D_QK;
static constexpr int D_V = 512;
static constexpr int D_ROPE = D_Q - D_V;
static constexpr int B_H = 128;
static constexpr int B_TOPK = 64;
static constexpr int NUM_THREADS = 16 * 32;
static constexpr int S_DS_VEC_ELEMS = 8;
static constexpr int S_DS_ROWS_PER_CTA = B_H / 2;
static constexpr int S_DS_COLS_PER_THREAD = B_TOPK / 2;

static_assert(S_DS_ROWS_PER_CTA == B_TOPK, "S/dS writer mapping assumes a 64x64 softmax tile per CTA.");
static_assert(S_DS_COLS_PER_THREAD % S_DS_VEC_ELEMS == 0, "S/dS vectorized stores require B_TOPK/2 to be a multiple of 8.");

template<int NUM_TILES>
using SmemLayoutQTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H / 2>, Int<64 * NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutQNoPE = SmemLayoutQTiles<8>;
using SmemLayoutQRoPE = SmemLayoutQTiles<1>;
using SmemLayoutQ = SmemLayoutQTiles<9>;
using SmemLayoutdO = SmemLayoutQTiles<D_V / 64>;

template<int NUM_TILES>
using SmemLayoutQTilesTransposed = decltype(coalesce(tile_to_shape(
    UMMA::Layout_MN_SW128_Atom<bf16>{},
    Shape<Int<64 * NUM_TILES>, Int<B_H / 2>>{},
    Step<_2, _1>{}
), Shape<_1, _1>{}));

using SmemLayoutQNoPETransposed = SmemLayoutQTilesTransposed<4>;
using SmemLayoutQRoPETransposed = SmemLayoutQTilesTransposed<1>;

template<int NUM_TILES>
using SmemLayoutKVTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_TOPK / 2>, Int<64 * NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutKNoPE = SmemLayoutKVTiles<8>;
using SmemLayoutKRoPE = SmemLayoutKVTiles<1>;
using SmemLayoutKV = SmemLayoutKVTiles<9>;
using SmemLayoutV = SmemLayoutKNoPE;

template<int NUM_TILES>
using SmemLayoutKVTilesTransposed = decltype(coalesce(tile_to_shape(
    UMMA::Layout_MN_SW128_Atom<bf16>{},
    Shape<Int<64 * NUM_TILES>, Int<B_TOPK / 2>>{},
    Step<_2, _1>{}
), Shape<_1, _1>{}));

using SmemLayoutKNoPETransposed = SmemLayoutKVTilesTransposed<4>;
using SmemLayoutKRoPETransposed = SmemLayoutKVTilesTransposed<1>;

using SmemLayoutKPeerNoPE = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_TOPK>, Int<D_V / 2>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutKPeerRoPE = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_TOPK>, Int<D_ROPE / 2>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutKPeerNoPE_MMA = decltype(coalesce(tile_to_shape(
    UMMA::Layout_MN_SW128_Atom<bf16>{},
    Shape<Int<D_V / 2>, Int<B_TOPK>>{},
    Step<_2, _1>{}
), Shape<_1, _1>{}));

using SmemLayoutKPeerRoPE_MMA = decltype(coalesce(tile_to_shape(
    UMMA::Layout_MN_SW64_Atom<bf16>{},
    Shape<Int<D_ROPE / 2>, Int<B_TOPK>>{},
    Step<_2, _1>{}
), Shape<_1, _1>{}));

using SmemLayoutS = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_INTER_Atom<bf16>{},
    Shape<Int<B_H / 2>, Int<B_TOPK>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutdS = SmemLayoutS;

using SmemLayoutdSTransposed = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_INTER_Atom<bf16>{},
    Shape<Int<B_H / 2>, Int<B_TOPK>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using TiledMMA_P = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS_NOELECT<bf16, bf16, float, B_H, B_TOPK, UMMA::Major::K, UMMA::Major::K>{}
));

using TiledMMA_dP = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS_NOELECT<bf16, bf16, float, B_H, B_TOPK, UMMA::Major::K, UMMA::Major::K>{}
));

using TiledMMA_dQ = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS_NOELECT<bf16, bf16, float, B_H, 256, UMMA::Major::K, UMMA::Major::MN>{},
    Layout<Shape<_1, _1, _1>>{},
    Tile<Int<128>, Layout<Shape<_128, _2, _2>, Stride<_1, _256, _128>>, _16>{}
));

using TiledMMA_dQ_RoPE = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS_NOELECT<bf16, bf16, float, B_H, D_ROPE, UMMA::Major::K, UMMA::Major::MN>{}
));

static_assert(cosize_v<SmemLayoutKPeerNoPE> == cosize_v<SmemLayoutKPeerNoPE_MMA>);
static_assert(cosize_v<SmemLayoutKPeerRoPE> == cosize_v<SmemLayoutKPeerRoPE_MMA>);
static_assert(cosize_v<SmemLayoutKV> == cosize_v<SmemLayoutKPeerNoPE> + cosize_v<SmemLayoutKPeerRoPE>);

struct tmem_cols {
    static constexpr int dQ = 0;
    static constexpr int dQ_RoPE = 256;
    static constexpr int P = 288;
    static constexpr int dP = 320;
    static constexpr int kNumUsedCols = 352;
};

static_assert(tmem_cols::dQ_RoPE == tmem_cols::dQ + D_V / 2);
static_assert(tmem_cols::P == tmem_cols::dQ_RoPE + D_ROPE / 2);
static_assert(tmem_cols::dP == tmem_cols::P + B_TOPK / 2);
static_assert(tmem_cols::kNumUsedCols == tmem_cols::dP + B_TOPK / 2);

struct alignas(128) SharedMemoryPlan {
    union {
        struct {
            array_aligned<bf16, cosize_v<SmemLayoutKNoPE>> k_nope;
            array_aligned<bf16, cosize_v<SmemLayoutKRoPE>> k_rope;
            array_aligned<bf16, cosize_v<SmemLayoutKV>> kv_peer;
            array_aligned<bf16, cosize_v<SmemLayoutQNoPE>> q_nope;
            array_aligned<bf16, cosize_v<SmemLayoutQRoPE>> q_rope;
        } q_kv;
        array_aligned<bf16, cosize_v<SmemLayoutQ>> dq;
    } u;

    array_aligned<bf16, cosize_v<SmemLayoutdO>> dO;
    struct {
        array_aligned<bf16, cosize_v<SmemLayoutdSTransposed>> s;
        array_aligned<bf16, cosize_v<SmemLayoutdSTransposed>> ds;
    } s_ds;
    char is_k_valid[B_TOPK / 8];

    transac_bar_t bar_prologue_q_nope;
    transac_bar_t bar_prologue_q_rope;
    transac_bar_t bar_prologue_kv;
    transac_bar_t bar_prologue_dO;
    transac_bar_t bar_p_ready;
    transac_bar_t bar_dp_ready;
    transac_bar_t bar_s_ready;
    transac_bar_t bar_ds_ready;
    transac_bar_t bar_k_valid_free;
    transac_bar_t bar_k_valid_ready;
    transac_bar_t bar_kv_peer_nope_ready;
    transac_bar_t bar_kv_peer_rope_ready;
    transac_bar_t bar_dq_ready;

    array_aligned<uint32_t, 1> tmem_start_addr;
};

static constexpr size_t SMEM_SIZE = sizeof(SharedMemoryPlan);

}  // namespace sm100::bwd::head128_2kernels::dq
