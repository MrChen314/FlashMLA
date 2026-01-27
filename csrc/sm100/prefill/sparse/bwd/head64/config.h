#pragma once

#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "defines.h"

namespace sm100::bwd::head64 {

using namespace cute;

// ============================================================================
// 维度常量定义
// ============================================================================
constexpr int D_Q = 576;                    // Query 维度
constexpr int D_K = 576;                    // Key 维度  
constexpr int D_V = 512;                    // Value/NoPE 维度
constexpr int D_ROPE = D_Q - D_V;           // RoPE 维度 = 64
constexpr float MAX_INIT_VAL = -1e30f;      // 用于 max logits 初始化

constexpr int B_H = 64;                     // Query head 块大小
constexpr int B_TOPK = 32;                  // TopK 块大小 (反向中使用32)
constexpr int NUM_THREADS = 128 + 128 + 128;  // 3个 WarpGroup，每个128线程

// ============================================================================
// TMA 参数模板
// ============================================================================
template<
    typename Shape_Q_NoPE, typename TMA_Q_NoPE,
    typename Shape_Q_RoPE, typename TMA_Q_RoPE,
    typename Shape_dO, typename TMA_dO,
    typename Shape_O, typename TMA_O
>
struct TmaParams {
    Shape_Q_NoPE shape_Q_nope; TMA_Q_NoPE tma_Q_nope;
    Shape_Q_RoPE shape_Q_rope; TMA_Q_RoPE tma_Q_rope;
    Shape_dO shape_dO; TMA_dO tma_dO;
    Shape_O shape_O; TMA_O tma_O;
    CUtensorMap tensor_map_kv_nope;
};

// ============================================================================
// SMEM Layout 定义
// ============================================================================

// Q NoPE Layout: [B_H, D_V] = [64, 512]
using SmemLayoutQNoPE = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<D_V>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// Q RoPE Layout: [B_H, D_ROPE] = [64, 64]
using SmemLayoutQRoPE = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_H>, Int<D_ROPE>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// Q NoPE 用于 TiledMMA 的 Layout: [B_H, D_V] = [64, 512]
// 与 SmemLayoutQNoPE 相同，用于 Dual GEMM
using SmemLayoutQNoPE_TiledMMA = SmemLayoutQNoPE;

// Q RoPE 用于 TiledMMA 的 Layout: [B_H*2, D_ROPE/2] = [128, 32]
// 用于 Dual GEMM 的 Q RoPE 视图
using SmemLayoutQRoPE_TiledMMA = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_H*2>, Int<D_ROPE/2>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// KV NoPE Layout: [B_TOPK, D_V] = [32, 512]
using SmemLayoutKVNoPE = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_TOPK>, Int<D_V>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// KV RoPE Layout: [B_TOPK, D_ROPE] = [32, 64]
using SmemLayoutKVRoPE = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_TOPK>, Int<D_ROPE>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// KV NoPE 用于 TiledMMA 的 Layout: [B_TOPK*2, D_V/2] = [64, 256]
// 用于 Implicit Dual GEMM
using SmemLayoutKVNoPE_TiledMMA = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_TOPK*2>, Int<D_V/2>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// KV RoPE 用于 TiledMMA 的 Layout: [B_TOPK*2, D_ROPE/2] = [64, 32]
using SmemLayoutKVRoPE_TiledMMA = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_TOPK*2>, Int<D_ROPE/2>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// dO Layout: [B_H, D_V] = [64, 512]
using SmemLayoutdO = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<D_V>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// dQ NoPE Layout: [B_H, D_V] = [64, 512]
using SmemLayoutdQNoPE = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<D_V>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// dQ RoPE Layout: [B_H, D_ROPE] = [64, 64]
using SmemLayoutdQRoPE = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_H>, Int<D_ROPE>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// S/dS 矩阵 Layout: [B_H, B_TOPK] = [64, 32]
using SmemLayoutS = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_INTER_Atom<bf16>{},
    Shape<Int<B_H>, Int<B_TOPK>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// ============================================================================
// TiledMMA 定义
// ============================================================================

// TiledMMA_P: 用于计算 P = Q @ K^T
// 使用 Dual GEMM, N = 2*B_TOPK = 64
// 指令: utcmma_ss (SMEM-SMEM)
using TiledMMA_P = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, B_H, B_TOPK*2, UMMA::Major::K, UMMA::Major::K>{}
));

// TiledMMA_dV: 用于计算 dV = P^T @ dO (分块处理，每次 N=256)
// 指令: utcmma_ss (SMEM-SMEM)
// 由于 D_V=512 > 256，需要分两次计算
using TiledMMA_dV = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, B_TOPK, D_V/2, UMMA::Major::MN, UMMA::Major::K>{}
));

// TiledMMA_dK: 用于计算 dK = dP^T @ Q (分块处理，每次 N=256)
// 指令: utcmma_ss (SMEM-SMEM)
// 由于 D_V=512 > 256，需要分两次计算
using TiledMMA_dK = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, B_TOPK, D_V/2, UMMA::Major::MN, UMMA::Major::K>{}
));

// TiledMMA_dQ: 用于计算 dQ = dP @ K (分块处理，每次 N=256)
// 指令: utcmma_ss (SMEM-SMEM)
// 由于 D_V=512 > 256，需要分两次计算
using TiledMMA_dQ = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, B_H, D_V/2, UMMA::Major::K, UMMA::Major::MN>{}
));

// ============================================================================
// Named Barriers 枚举
// ============================================================================
enum NamedBarriers : int {
    wg0_sync = 0,
    wg0_warp02_sync = 1,
    wg0_warp13_sync = 2,
    wg1_sync = 3,
    wg2_sync = 4,
    pepi_sync = 5,
};

// ============================================================================
// TMEM 列映射
// 基于 SM100反向资源占用分析.md (已更新为分块计算):
//   P: [64, 32] float32, columns 0~16
//   dP: [64, 32] float32, columns 16~32
//   dKV_NoPE: [32, 512] float32, columns 32~160 (分两部分: part0 32~96, part1 96~160)
//   dKV_RoPE: [32, 64] float32, columns 160~176
//   dQ_NoPE: [64, 512] float32, columns 176~432 (分两部分: part0 176~304, part1 304~432)
//   dQ_RoPE: [64, 64] float32, columns 432~464
// 总占用: 464 列 (90.6% 利用率)
// ============================================================================
namespace tmem_cols {
    // P 矩阵: Attention Scores (softmax 后)
    // Shape: [B_H, B_TOPK] = [64, 32], 需要 32*64/128 = 16 列
    constexpr int P = 0;
    
    // dP 矩阵: dAttention Scores
    // Shape: [B_H, B_TOPK] = [64, 32], 需要 16 列
    constexpr int dP = 16;
    
    // dKV NoPE 累加器: dKey/dValue 梯度累加 (NoPE 部分)
    // 由于 MMA N=256 限制，分成两部分存储
    // Shape: [B_TOPK, 256] × 2 = [32, 512], 每部分需要 256*32/128 = 64 列
    constexpr int dKV = 32;           // dKV_part0: 32~160
    
    // dKV RoPE 累加器: dKey/dValue 梯度累加 (RoPE 部分)
    // Shape: [B_TOPK, D_ROPE] = [32, 64], 需要 64*32/128 = 16 列
    constexpr int dKV_RoPE = 160;
    
    // dQ NoPE 累加器: dQuery 梯度累加 (NoPE 部分)
    // 由于 MMA N=256 限制，分成两部分存储
    // Shape: [B_H, 256] × 2 = [64, 512], 每部分需要 256*64/128 = 128 列
    constexpr int dQ = 176;           // dQ_part0: 176~432
    
    // dQ RoPE 累加器: dQuery 梯度累加 (RoPE 部分)
    // Shape: [B_H, D_ROPE] = [64, 64], 需要 64*64/128 = 32 列
    constexpr int dQ_RoPE = 432;      // dQ_RoPE: 432~464
}


// ============================================================================
// Shared Memory 规划结构体
// 基于 SM100反向资源占用分析.md 的内存复用策略:
//   Q + KV 同时驻留: 72 KB (Q) + 36 KB (KV) = 108 KB
//   dO 全程驻留: 64 KB
//   dQ 输出阶段与 Q+KV 复用
// 实际峰值: ~180 KB (78% SM100 限制)
// ============================================================================
struct SharedMemoryPlan {
    // 主联合体: dQ 与 Q+KV 空间复用
    union {
        // Q + KV 计算阶段: Q 和 KV 同时驻留
        struct {
            // Q 缓冲区
            array_aligned<bf16, cosize_v<SmemLayoutQNoPE>> q_nope;      // [64, 512] bf16 = 64 KB
            array_aligned<bf16, cosize_v<SmemLayoutQRoPE>> q_rope;      // [64, 64] bf16 = 8 KB
            // KV 缓冲区 (单 buffer，不采用多缓冲)
            array_aligned<bf16, cosize_v<SmemLayoutKVNoPE>> kv_nope;    // [32, 512] bf16 = 32 KB
            array_aligned<bf16, cosize_v<SmemLayoutKVRoPE>> kv_rope;    // [32, 64] bf16 = 4 KB
        } q_kv;
        
        // dQ 输出阶段 (与 Q+KV 空间复用)
        struct {
            array_aligned<bf16, cosize_v<SmemLayoutdQNoPE>> dq_nope;    // [64, 512] bf16 = 64 KB
            array_aligned<bf16, cosize_v<SmemLayoutdQRoPE>> dq_rope;    // [64, 64] bf16 = 8 KB
        } dq;
    } u;
    
    // dO 缓冲区: 全程驻留
    array_aligned<bf16, cosize_v<SmemLayoutdO>> dO;                     // [64, 512] bf16 = 64 KB
    
    // S/dS 矩阵联合体
    union {
        array_aligned<bf16, cosize_v<SmemLayoutS>> s;                   // P_shared_cast [64, 32]
        array_aligned<bf16, cosize_v<SmemLayoutS>> ds;                  // dP_shared_cast [64, 32]
    } s_ds;
    
    // P 交换缓冲区 (用于 warp 间数据交换)
    float p_exchange_buf[4][32 * (B_TOPK/2)];
    
    // KV 有效性掩码 (单 buffer)
    char is_kv_valid[B_TOPK/8];
    
    // ========================================================================
    // 同步屏障
    // ========================================================================
    // Prologue 阶段屏障
    transac_bar_t bar_prologue_q_nope;      // Q NoPE TMA 完成
    transac_bar_t bar_prologue_q_rope;      // Q RoPE TMA 完成
    transac_bar_t bar_prologue_dO;          // dO TMA 完成
    transac_bar_t bar_prologue_utccp;       // Q UTCCP 完成
    
    // QK^T 计算屏障
    transac_bar_t bar_qk_nope_done;             // P = Q×K^T (NoPE 部分) 完成
    transac_bar_t bar_qk_rope_done;             // P = Q×K^T (RoPE 部分) 完成
    
    // dP 计算屏障
    transac_bar_t bar_dp_done;                  // dP = P * (dO×K^T - Delta) 完成
    
    // dQ/dKV 累加屏障
    transac_bar_t bar_dq_done;                  // dQ += dP×K 完成
    transac_bar_t bar_dkv_done;                 // dKV += dP^T×Q + P^T×dO 完成
    
    // KV 就绪屏障 (单 buffer)
    transac_bar_t bar_kv_nope_ready;            // KV NoPE 数据就绪
    transac_bar_t bar_kv_rope_ready;            // KV RoPE 数据就绪
    
    // P/dP 空闲屏障
    transac_bar_t bar_p_free;                   // P 矩阵空闲
    transac_bar_t bar_dp_free;                  // dP 矩阵空闲
    
    // S/dO 就绪屏障
    transac_bar_t bar_s_do_ready;               // S 和 dO 就绪
    
    // KV 有效性屏障 (单 buffer)
    transac_bar_t bar_kv_valid_ready;
    transac_bar_t bar_kv_valid_free;
    
    // TMEM 起始地址
    array_aligned<uint32_t, 1> tmem_start_addr;
    
    // Rowwise 缓冲区 (用于 softmax 和 Delta)
    float rowwise_max_buf[128];                 // max logits
    float rowwise_li_buf[128];                  // log-sum-exp
    float rowwise_delta_buf[128];               // Delta = sum(O * dO)
};

}  // namespace sm100::bwd::head64