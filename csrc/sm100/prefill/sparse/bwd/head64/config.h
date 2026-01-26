#pragma once

#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "defines.h"

namespace sm100::bwd::head64 {

using namespace cute;

// ============================================================================
// TMA 参数结构体
// ============================================================================
template<
    typename Shape_Q_NoPE, typename TMA_Q_NoPE,
    typename Shape_Q_RoPE, typename TMA_Q_RoPE,
    typename Shape_dO, typename TMA_dO,
    typename Shape_dQ, typename TMA_dQ
>
struct TmaParams {
    Shape_Q_NoPE shape_Q_nope; TMA_Q_NoPE tma_Q_nope;   // Q 的 NoPE 部分
    Shape_Q_RoPE shape_Q_rope; TMA_Q_RoPE tma_Q_rope;   // Q 的 RoPE 部分
    Shape_dO shape_dO; TMA_dO tma_dO;                   // dOutput
    Shape_dQ shape_dQ; TMA_dQ tma_dQ;                   // dQuery 输出
    CUtensorMap tensor_map_kv_nope;                     // KV NoPE 部分的 TMA 描述符
    CUtensorMap tensor_map_kv_rope;                     // KV RoPE 部分的 TMA 描述符
};

struct float2x2 {
    float2 lo, hi;
};

// ============================================================================
// 核心维度常量
// 基于 TileLang 反向实现: D=512, D_tail=64, block_H=64, BS=32, threads=384
// ============================================================================
constexpr int D_Q = 576;                // Query/Key 总维度 (NoPE + RoPE)
constexpr int D_K = 576;                // Key 总维度
constexpr int D_V = 512;                // Value/Output 维度 (NoPE 部分)
constexpr int D_ROPE = 64;              // RoPE 维度 (D_tail)
constexpr float MAX_INIT_VAL = -1e30;   // 用于避免 -inf - (-inf) = nan

// ============================================================================
// 分块参数
// ============================================================================
constexpr int B_H = 64;                 // Head 分块大小 (block_H)
constexpr int B_TOPK = 32;              // TopK 分块大小 (BS/block_size)
constexpr int NUM_BUFS = 2;             // KV 缓冲区数量
constexpr int NUM_THREADS = 384;        // 线程数
constexpr int SPLIT_STORE = 2;          // dKV 分批写入因子

// ============================================================================
// TMEM 列映射
// 基于 SM100反向资源占用分析.md:
//   P: [64, 32] float32, columns 0~16
//   dP: [64, 32] float32, columns 16~32
//   dKV_NoPE: [32, 512] float32, columns 32~160
//   dKV_RoPE: [32, 64] float32, columns 160~176
//   dQ: [64, 576] float32, columns 176~464
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
    // Shape: [B_TOPK, D_V] = [32, 512], 需要 512*32/128 = 128 列
    constexpr int dKV = 32;
    
    // dKV RoPE 累加器: dKey/dValue 梯度累加 (RoPE 部分)
    // Shape: [B_TOPK, D_ROPE] = [32, 64], 需要 64*32/128 = 16 列
    constexpr int dKV_RoPE = 160;
    
    // dQ 累加器: dQuery 梯度累加
    // Shape: [B_H, D_Q] = [64, 576], 需要 576*64/128 = 288 列
    constexpr int dQ = 176;
    constexpr int dQ_RoPE = 176 + 256;  // dQ 的 RoPE 部分起始列
}

// ============================================================================
// Shared Memory 布局定义
// 参考前向 config.h 的布局风格，使用 CUTE 的 swizzle 布局
// ============================================================================

// Q NoPE 部分布局: [B_H, D_V] = [64, 512], SW128
using SmemLayoutQNoPE = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<D_V>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// Q RoPE 部分布局: [B_H, D_ROPE] = [64, 64], SW64
using SmemLayoutQRoPE = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_H>, Int<D_ROPE>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// dO 布局: [B_H, D_V] = [64, 512], SW128
using SmemLayoutdO = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<D_V>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// KV NoPE 部分布局: [B_TOPK, D_V] = [32, 512], SW128
template<int NUM_TILES>
using SmemLayoutKVNoPETiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_TOPK>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutKVNoPE = SmemLayoutKVNoPETiles<8>;  // 8 tiles = 512

// KV NoPE 转置布局 (用于 S×V 计算): [D_V, B_TOPK]
using SmemLayoutV = decltype(coalesce(
    composition(
        SmemLayoutKVNoPE{},
        Layout<Shape<Int<D_V>, Int<B_TOPK>>, Stride<Int<B_TOPK>, _1>>{}
    )
, Shape<_1, _1>{}));

// KV RoPE 部分布局: [B_TOPK, D_ROPE] = [32, 64], SW64
using SmemLayoutKVRoPE = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_TOPK>, Int<D_ROPE>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// S/dS 矩阵布局 (P_shared_cast, dP_shared_cast): [B_H, B_TOPK] = [64, 32], INTER
using SmemLayoutS = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_INTER_Atom<bf16>{},
    Shape<Int<B_H>, Int<B_TOPK>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// dQ 输出布局: [B_H, D_V] = [64, 512], SW128
using SmemLayoutdQNoPE = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<D_V>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// dQ RoPE 输出布局: [B_H, D_ROPE] = [64, 64], SW64
using SmemLayoutdQRoPE = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_H>, Int<D_ROPE>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

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

// ============================================================================
// TiledMMA 定义
// 反向传播需要多种 GEMM 配置，所有 MMA 都使用 utcmma_ss (SMEM-SMEM)
// ============================================================================

// P/dP 矩阵计算 MMA: Q×K^T → P, dO×K^T → dP
// Shape: [B_H, B_TOPK] = [64, 32]
// 输入: A (Q/dO) 在 SMEM [64, 512], Major::K
//       B (K^T) 在 SMEM [32, 512]^T, Major::MN (转置后 K 沿 M 维度连续)
// 输出: TMEM P/dP [64, 32]
using TiledMMA_P = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, B_H, B_TOPK, UMMA::Major::K, UMMA::Major::MN>{}
));

// dKV 计算 MMA: S^T×dO → dKV, ds^T×Q → dKV
// Shape: [B_TOPK, D_V] = [32, 512]
// 输入: A (S^T/ds^T) 在 SMEM [64, 32]^T, Major::MN (转置后 S 沿 M 维度连续)
//       B (dO/Q) 在 SMEM [64, 512], Major::K
// 输出: TMEM dKV [32, 512]
using TiledMMA_dKV = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, B_TOPK, D_V, UMMA::Major::MN, UMMA::Major::K>{}
));

// dKV_RoPE 计算 MMA: ds^T×Q_rope → dKV_RoPE
// Shape: [B_TOPK, D_ROPE] = [32, 64]
// 输入: A (ds^T) 在 SMEM [64, 32]^T, Major::MN
//       B (Q_rope) 在 SMEM [64, 64], Major::K
// 输出: TMEM dKV_RoPE [32, 64]
using TiledMMA_dKV_RoPE = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, B_TOPK, D_ROPE, UMMA::Major::MN, UMMA::Major::K>{}
));

// dQ 计算 MMA: ds×K → dQ
// Shape: [B_H, D_V] = [64, 512]
// 输入: A (ds) 在 SMEM [64, 32], Major::K
//       B (K) 在 SMEM [32, 512], Major::K
// 输出: TMEM dQ [64, 512]
using TiledMMA_dQ = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, B_H, D_V, UMMA::Major::K, UMMA::Major::K>{}
));

// dQ_RoPE 计算 MMA: ds×K_rope → dQ_RoPE
// Shape: [B_H, D_ROPE] = [64, 64]
// 输入: A (ds) 在 SMEM [64, 32], Major::K
//       B (K_rope) 在 SMEM [32, 64], Major::K
// 输出: TMEM dQ_RoPE [64, 64]
using TiledMMA_dQ_RoPE = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, B_H, D_ROPE, UMMA::Major::K, UMMA::Major::K>{}
));

// ============================================================================
// Named Barriers 枚举
// ============================================================================
enum NamedBarriers : int {
    wg0_sync = 0,           // Warp Group 0 同步
    wg0_warp02_sync = 1,    // Warp 0,2 同步
    wg0_warp13_sync = 2,    // Warp 1,3 同步
    pepi_sync = 3,          // Prologue-Epilogue 同步
    dkv_sync = 4,           // dKV 写入同步
};


}  // namespace sm100::bwd::head64
