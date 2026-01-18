#pragma once

#include <math_constants.h>
#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "defines.h"
#include "params.h"

namespace sm100::bwd::head64 {

using namespace cute;

// TMA Parameters structure
template<
    typename Shape_Q, typename TMA_Q,
    typename Shape_dO, typename TMA_dO,
    typename Shape_dQ, typename TMA_dQ
>
struct TmaParams {
    Shape_Q shape_Q; TMA_Q tma_Q;
    Shape_dO shape_dO; TMA_dO tma_dO;
    Shape_dQ shape_dQ; TMA_dQ tma_dQ;
    CUtensorMap tensor_map_kv;
};

// Helper struct for float2x2
struct float2x2 {
    float2 lo, hi;
};

// Core dimensions
constexpr int D_Q = 576;
constexpr int D_K = 576;
constexpr int D_V = 512;
constexpr float MAX_INIT_VAL = -1e30f;

// Block dimensions - Head64 configuration
constexpr int B_H = 64;         // Head block size
constexpr int B_TOPK = 64;      // TopK block size
constexpr int NUM_BUFS = 3;     // Pipeline depth
constexpr int NUM_THREADS = 384; // 128 scale threads + 128 TMA threads + 128 MMA threads

// TMEM column layout for backward
namespace tmem_cols {
    //   0 ~ 256: dQ accumulator
    // 256 ~ 384: dKV accumulator (reused for dK and dV)
    // 384 ~ 448: P/dP matrix
    // 448 ~ 512: Q partition for dK computation
    constexpr int dQ = 0;
    constexpr int dKV = 256;
    constexpr int P = 384;
    constexpr int Q_part = 448;
}

// Shared memory layouts
using SmemLayoutQ = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<D_V>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutQRoPE = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_H>, Int<D_Q-D_V>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutdO = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<D_V>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutKTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_TOPK>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutK = SmemLayoutKTiles<D_K/64>;

using SmemLayoutV = decltype(coalesce(
    composition(
        SmemLayoutKTiles<D_V/64>{},
        Layout<Shape<Int<D_V>, Int<B_TOPK>>, Stride<Int<B_TOPK>, _1>>{}
    )
, Shape<_1, _1>{}));

using SmemLayoutKRoPE = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_TOPK>, Int<64>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutP = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_INTER_Atom<bf16>{},
    Shape<Int<B_H>, Int<B_TOPK>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutdP = SmemLayoutP;

template<int NUM_TILES>
using SmemLayoutdQTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutdQ = SmemLayoutdQTiles<D_Q/64>;

// Shared Memory Plan for backward
struct SharedMemoryPlan {
    union {
        // Configuration 1: For dQ computation (dQ += dS @ K)
        struct {
            array_aligned<bf16, cosize_v<SmemLayoutP>> dS;           // dS matrix [B_H, B_TOPK]
            array_aligned<bf16, cosize_v<SmemLayoutK>> k_buf[NUM_BUFS]; // K buffers
        } dQ_cfg;
        
        // Configuration 2: For dK/dV computation
        struct {
            array_aligned<bf16, cosize_v<SmemLayoutP>> p_buf;        // P matrix [B_H, B_TOPK]
            array_aligned<bf16, cosize_v<SmemLayoutdO>> do_buf;      // dO matrix [B_H, D_V]
            array_aligned<bf16, cosize_v<SmemLayoutQ>> q_buf;        // Q matrix [B_H, D_V]
            array_aligned<bf16, cosize_v<SmemLayoutQRoPE>> q_rope_buf; // Q RoPE [B_H, 64]
        } dKV_cfg;
        
        // Configuration 3: For output
        array_aligned<bf16, cosize_v<SmemLayoutdQ>> dQ_out;
    } u;
    
    // P exchange buffer for cross-warp communication
    float p_exchange_buf[4][32 * (B_TOPK/2)];
    
    // S/dP storage (can overlap with Q_RoPE in some phases)
    union {
        bf16 s[B_H * B_TOPK];
        array_aligned<bf16, cosize_v<SmemLayoutQRoPE>> q_rope;
    } s_q_rope;
    
    // Validity masks for indices
    char is_k_valid[NUM_BUFS][B_TOPK/8];
    
    // Synchronization barriers
    transac_bar_t bar_prologue_q, bar_prologue_q_rope;
    transac_bar_t bar_prologue_do;
    transac_bar_t bar_prologue_utccp_nope, bar_prologue_utccp_rope;
    
    // Pipeline barriers
    transac_bar_t bar_k_ready[NUM_BUFS][2];      // K data ready
    transac_bar_t bar_v_ready[NUM_BUFS][2];      // V data ready
    transac_bar_t bar_p_computed[NUM_BUFS];      // P computation done
    transac_bar_t bar_dp_computed[NUM_BUFS];     // dP computation done
    transac_bar_t bar_dq_accumulated[NUM_BUFS];  // dQ accumulation done
    transac_bar_t bar_dkv_ready[NUM_BUFS];       // dK/dV ready for atomic
    
    transac_bar_t bar_k_valid_ready[NUM_BUFS], bar_k_valid_free[NUM_BUFS];
    transac_bar_t bar_p_free;
    transac_bar_t bar_so_ready;
    
    // TMEM allocation tracking
    array_aligned<uint32_t, 1> tmem_start_addr;
    
    // Temporary buffers for softmax
    float rowwise_max_buf[128];
    float rowwise_li_buf[128];
    float delta_buf[B_H];  // Delta = rowsum(O * dO)
};

// TiledMMA configurations for backward
// P = Q @ K^T (recompute attention scores)
using TiledMMA_P = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_TS_NOELECT<bf16, bf16, float, B_H, 128, UMMA::Major::K, UMMA::Major::K>{}
));

// dP_mid = dO @ V^T
using TiledMMA_dP = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, B_H, B_TOPK, UMMA::Major::K, UMMA::Major::MN>{}
));

// dQ += dS @ K
using TiledMMA_dQ = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, B_H, D_V, UMMA::Major::K, UMMA::Major::K>{}
));

// dK = dS^T @ Q
using TiledMMA_dK = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, B_TOPK, D_V, UMMA::Major::MN, UMMA::Major::K>{}
));

// dV = P^T @ dO
using TiledMMA_dV = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, B_TOPK, D_V, UMMA::Major::MN, UMMA::Major::K>{}
));

// Named barriers for synchronization
enum NamedBarriers : int {
    wg0_sync = 0,
    wg0_warp02_sync = 1,
    wg0_warp13_sync = 2,
    pepi_sync = 3,
    dq_sync = 4,
    dkv_sync = 5,
};

} // namespace sm100::bwd::head64
