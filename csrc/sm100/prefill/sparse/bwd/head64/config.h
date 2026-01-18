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
constexpr int D_ROPE = D_Q - D_V;  // 64 - RoPE 维度
constexpr float MAX_INIT_VAL = -1e30f;

// Block dimensions - Head64 configuration
constexpr int B_H = 64;         // Head block size
constexpr int B_TOPK = 64;      // TopK block size
constexpr int NUM_BUFS = 2;     // Pipeline depth (减少到2以节省共享内存)
constexpr int NUM_THREADS = 384; // 128 scale threads + 128 TMA threads + 128 MMA threads

// 共享内存使用估算 (227 KB 限制):
// - k_buf[2]: 64 × 576 × 2 × 2 = 147,456 bytes
// - s_buf + ds_buf: 8,192 × 2 = 16,384 bytes
// - p_exchange_buf: 16,384 bytes
// - barriers + misc: ~3,000 bytes
// 总计: ~183 KB ✓

// TMEM column layout for backward
// 优化：复用 TMEM 空间以适应 512 列限制
// 
// SM100 TMEM 规格: 128行 × 512列 × 4字节 = 256KB
// 
// 分时复用策略：
// - dQ 累加器在整个 kernel 生命周期内持续使用
// - dKV_nope 和 dK_rope 每个 k block 计算后立即原子加到全局内存，可以复用
// - P 矩阵和 Q_part 可以与 dKV 区域复用（分阶段使用）
//
// 布局（512列内）：
//   0 ~ 288: dQ accumulator (B_H=64 rows, D_Q=576 -> 288 TMEM cols)
// 288 ~ 320: P matrix (B_H=64 rows, B_TOPK=64 -> 32 cols) [阶段1]
// 288 ~ 480: dKV_nope accumulator (B_TOPK=64, D_V=512/2=256 cols, 分两批处理)
//            - 第一批: cols 0-127 使用 TMEM 288-416
//            - 第二批: cols 128-255 复用 TMEM 288-416
// 480 ~ 512: dK_rope (32 cols) + Q_part 临时区 (复用)
namespace tmem_cols {
    constexpr int dQ = 0;            // 0 ~ 288: dQ 累加器（常驻）
    constexpr int P = 288;           // 288 ~ 320: P 矩阵（与 dKV 分时复用）
    constexpr int dKV_nope = 288;    // 288 ~ 416: dKV_nope 分批处理 (128 cols/批)
    constexpr int dK_rope = 416;     // 416 ~ 448: dK_rope (32 cols)
    constexpr int Q_part = 448;      // 448 ~ 512: Q partition (64 cols)
    
    // 分批处理常量
    constexpr int DKV_BATCH_COLS = 128;  // 每批处理 128 列 (256维)
    constexpr int DKV_NUM_BATCHES = 2;   // 共 2 批
}

// 总使用: 288 + 128 + 32 + 64 = 512 列 ✓

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
            array_aligned<bf16, cosize_v<SmemLayoutK>> k_buf[NUM_BUFS]; // K buffers
        } dQ_cfg;
        
        // Configuration 2: For dK/dV computation
        struct {
            array_aligned<bf16, cosize_v<SmemLayoutdO>> do_buf;      // dO matrix [B_H, D_V]
            array_aligned<bf16, cosize_v<SmemLayoutQ>> q_buf;        // Q matrix [B_H, D_V]
            array_aligned<bf16, cosize_v<SmemLayoutQRoPE>> q_rope_buf; // Q RoPE [B_H, 64]
        } dKV_cfg;
        
        // Configuration 3: For output
        array_aligned<bf16, cosize_v<SmemLayoutdQ>> dQ_out;
    } u;
    
    // P exchange buffer for cross-warp communication
    float p_exchange_buf[4][32 * (B_TOPK/2)];
    
    // 修复：S 和 dS 使用独立的存储空间
    // S 用于计算 dV = S^T @ dO
    // dS 用于计算 dK = dS^T @ Q 和 dQ += dS @ K
    bf16 s_buf[B_H * B_TOPK];    // 存储 S (softmax 输出)
    bf16 ds_buf[B_H * B_TOPK];   // 存储 dS (softmax 梯度)
    
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
    transac_bar_t bar_s_ready;   // S 计算完成信号
    transac_bar_t bar_ds_ready;  // dS 计算完成信号
    
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
