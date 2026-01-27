#pragma once

#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "defines.h"

namespace sm100::bwd::head128 {

using namespace cute;

// ============================================================================
// 维度常量定义
// ============================================================================
constexpr int D_Q = 576;                    // Query 维度
constexpr int D_K = 576;                    // Key 维度  
constexpr int D_V = 512;                    // Value/NoPE 维度
constexpr int D_ROPE = D_Q - D_V;           // RoPE 维度 = 64
constexpr float MAX_INIT_VAL = -1e30f;      // 用于 max logits 初始化

// ============================================================================
// 2CTA 相关常量定义
// ============================================================================
constexpr int B_H = 128;                     // Query head 块大小 (2CTA 共享，每个 CTA 处理 B_H/2=64 行)
constexpr int B_TOPK = 64;                  // TopK 块大小 (2CTA 模式每次处理 64 个 topk，每个 CTA 加载 B_TOPK/2=32 行)
constexpr int NUM_BUFS = 2;                 // KV 双缓冲数量
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
// SMEM Layout 定义 (使用 NUM_TILES 模板形式，支持 2CTA)
// ============================================================================

// Q NoPE Layout 模板: [B_H/2, 64*NUM_TILES]
// 2CTA 模式下每个 CTA 处理 B_H/2 行
template<int NUM_TILES>
using SmemLayoutQNoPETiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H/2>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// Q NoPE 完整 Layout: [B_H/2, D_V] = [64, 512]
using SmemLayoutQNoPE = SmemLayoutQNoPETiles<D_V/64>;

// Q RoPE Layout 模板: [B_H/2, 64*NUM_TILES]
template<int NUM_TILES>
using SmemLayoutQRoPETiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_H/2>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// Q RoPE 完整 Layout: [B_H/2, D_ROPE] = [64, 64]
using SmemLayoutQRoPE = SmemLayoutQRoPETiles<D_ROPE/64>;

// Q NoPE 用于 TiledMMA 的 Layout (与 SmemLayoutQNoPE 相同)
using SmemLayoutQNoPE_TiledMMA = SmemLayoutQNoPE;

// Q RoPE 用于 TiledMMA 的 Layout: [B_H, D_ROPE/2] = [128, 32]
// 2CTA Dual GEMM: 两个 CTA 的 Q RoPE 合并
using SmemLayoutQRoPE_TiledMMA = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_H>, Int<D_ROPE/2>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// KV NoPE Layout 模板: [B_TOPK/2, 64*NUM_TILES]
// 2CTA 模式下每个 CTA 加载 B_TOPK/2 行
template<int NUM_TILES>
using SmemLayoutKVNoPETiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_TOPK/2>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// KV NoPE 完整 Layout: [B_TOPK/2, D_V] = [32, 512]
using SmemLayoutKVNoPE = SmemLayoutKVNoPETiles<D_V/64>;

// KV RoPE Layout 模板: [B_TOPK/2, 64*NUM_TILES]
template<int NUM_TILES>
using SmemLayoutKVRoPETiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_TOPK/2>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// KV RoPE 完整 Layout: [B_TOPK/2, D_ROPE] = [32, 64]
using SmemLayoutKVRoPE = SmemLayoutKVRoPETiles<D_ROPE/64>;

// KV NoPE 用于 TiledMMA 的 Layout: [B_TOPK/2, D_V] = [32, 512]
// 不使用 Dual GEMM，直接使用完整的 KV NoPE
using SmemLayoutKVNoPE_TiledMMA = SmemLayoutKVNoPE;

// KV RoPE 用于 TiledMMA 的 Layout: [B_TOPK, D_ROPE/2] = [64, 32]
using SmemLayoutKVRoPE_TiledMMA = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_TOPK>, Int<D_ROPE/2>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// dO Layout 模板: [B_H/2, 64*NUM_TILES]
template<int NUM_TILES>
using SmemLayoutdOTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H/2>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// dO 完整 Layout: [B_H/2, D_V] = [64, 512]
using SmemLayoutdO = SmemLayoutdOTiles<D_V/64>;

// dQ NoPE Layout 模板: [B_H/2, 64*NUM_TILES]
template<int NUM_TILES>
using SmemLayoutdQNoPETiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H/2>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// dQ NoPE 完整 Layout: [B_H/2, D_V] = [64, 512]
using SmemLayoutdQNoPE = SmemLayoutdQNoPETiles<D_V/64>;

// dQ RoPE Layout 模板: [B_H/2, 64*NUM_TILES]
template<int NUM_TILES>
using SmemLayoutdQRoPETiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_H/2>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// dQ RoPE 完整 Layout: [B_H/2, D_ROPE] = [64, 64]
using SmemLayoutdQRoPE = SmemLayoutdQRoPETiles<D_ROPE/64>;

// S/dS 矩阵 Layout 模板: [B_H/2, 64*NUM_TILES]
// 2CTA 模式下 S 矩阵形状为 [B_H/2, B_TOPK]
template<int NUM_TILES>
using SmemLayoutSTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_INTER_Atom<bf16>{},
    Shape<Int<B_H/2>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// S/dS 完整 Layout: [B_H/2, B_TOPK] = [64, 64]
using SmemLayoutS = SmemLayoutSTiles<B_TOPK/64>;

// ============================================================================
// TiledMMA 定义 (2CTA 模式)
// ============================================================================

// TiledMMA_P: 用于计算 P = Q @ K^T
// 2CTA 模式: [B_H, B_TOPK] = [128, 64]
// 每个 CTA 处理 B_H/2 = 64 行，共同完成完整的 MMA
// 指令: utcmma_ss (SMEM-SMEM), 2x1SM 协作
using TiledMMA_P = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS_NOELECT<bf16, bf16, float, B_H, B_TOPK, UMMA::Major::K, UMMA::Major::K>{}
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
// TMEM 列映射 (2CTA 模式, B_H=128)
// 每个 CTA 的 TMEM 分配:
//   P: [B_H/2, B_TOPK] = [64, 64] float32, 需要 64*64/128 = 32 列
//   dP: [B_H/2, B_TOPK] = [64, 64] float32, 需要 32 列
//   dKV_NoPE: [B_TOPK/2, D_V] = [32, 512] float32, 分两部分存储
//   dKV_RoPE: [B_TOPK/2, D_ROPE] = [32, 64] float32
//   dQ_NoPE: [B_H/2, D_V] = [64, 512] float32, 分两部分存储
//   dQ_RoPE: [B_H/2, D_ROPE] = [64, 64] float32
// ============================================================================
namespace tmem_cols {
    // P 矩阵: Attention Scores (softmax 后)
    // 2CTA 模式: 每个 CTA 存储 [B_H/2, B_TOPK] = [64, 64], 需要 64*64/128 = 32 列
    constexpr int P = 0;
    
    // dP 矩阵: dAttention Scores
    // 2CTA 模式: 每个 CTA 存储 [B_H/2, B_TOPK] = [64, 64], 需要 32 列
    constexpr int dP = 32;
    
    // dKV NoPE 累加器: dKey/dValue 梯度累加 (NoPE 部分)
    // 由于 MMA N=256 限制，分成两部分存储
    // Shape: [B_TOPK/2, 256] × 2 = [32, 512], 每部分需要 256*32/128 = 64 列
    constexpr int dKV = 64;         
    
    // dKV RoPE 累加器: dKey/dValue 梯度累加 (RoPE 部分)
    // Shape: [B_TOPK/2, D_ROPE] = [32, 64], 需要 64*32/128 = 16 列
    constexpr int dKV_RoPE = 192;
    
    // dQ NoPE 累加器: dQuery 梯度累加 (NoPE 部分)
    // 由于 MMA N=256 限制，分成两部分存储
    // Shape: [B_H/2, 256] × 2 = [64, 512], 每部分需要 256*64/128 = 128 列
    constexpr int dQ = 208;           
    
    // dQ RoPE 累加器: dQuery 梯度累加 (RoPE 部分)
    // Shape: [B_H/2, D_ROPE] = [64, 64], 需要 64*64/128 = 32 列
    constexpr int dQ_RoPE = 464;     
}


// ============================================================================
// Shared Memory 规划结构体 (2CTA 版本)
// 基于 SM100反向资源占用分析.md 的内存复用策略:
//   2CTA 模式: 每个 CTA 处理 B_H/2 行 Q 和 B_TOPK/2 行 KV
//   使用双缓冲 (NUM_BUFS=2) 支持流水线
// ============================================================================
struct SharedMemoryPlan {
    // 主联合体: dQ 与 Q+KV 空间复用
    union {
        // Q + KV 计算阶段: Q 和 KV 同时驻留
        struct {
            // Q 缓冲区 (每个 CTA 处理 B_H/2 行)
            array_aligned<bf16, cosize_v<SmemLayoutQNoPE>> q_nope;      // [B_H/2, D_V] bf16
            array_aligned<bf16, cosize_v<SmemLayoutQRoPE>> q_rope;      // [B_H/2, D_ROPE] bf16
            // KV 双缓冲区 (每个 CTA 加载 B_TOPK/2 行)
            array_aligned<bf16, cosize_v<SmemLayoutKVNoPE>> kv_nope;    // [B_TOPK/2, D_V] bf16
            array_aligned<bf16, cosize_v<SmemLayoutKVRoPE>> kv_rope;    // [B_TOPK/2, D_ROPE] bf16
        } q_kv;
        
        // dQ 输出阶段 (与 Q+KV 空间复用)
        struct {
            array_aligned<bf16, cosize_v<SmemLayoutdQNoPE>> dq_nope;    // [B_H/2, D_V] bf16
            array_aligned<bf16, cosize_v<SmemLayoutdQRoPE>> dq_rope;    // [B_H/2, D_ROPE] bf16
        } dq;
    } u;
    
    // dO 缓冲区: 全程驻留 (每个 CTA 处理 B_H/2 行)
    array_aligned<bf16, cosize_v<SmemLayoutdO>> dO;                     // [B_H/2, D_V] bf16
    
    // S/dS 矩阵: [B_H/2, B_TOPK]
    array_aligned<bf16, cosize_v<SmemLayoutS>> s;
    
    // P 交换缓冲区 (用于 warp 间数据交换, 2CTA 模式)
    float p[(B_H/2)*B_TOPK];
    
    // KV 有效性掩码 (双缓冲)
    char is_kv_valid[NUM_BUFS][B_TOPK/8];
    
    // ========================================================================
    // 同步屏障 (2CTA 同步)
    // ========================================================================
    // Prologue 阶段屏障
    transac_bar_t bar_prologue_q_nope;          // Q NoPE TMA 完成
    transac_bar_t bar_prologue_q_rope;          // Q RoPE TMA 完成
    transac_bar_t bar_prologue_dO;              // dO TMA 完成
    transac_bar_t bar_prologue_utccp;           // Q UTCCP 完成
    
    // QK^T 计算屏障 (双缓冲, 支持流水线)
    transac_bar_t bar_qk_done[NUM_BUFS];        // P = Q×K^T 完成 (KV 可释放)
    
    // SV 计算屏障 (双缓冲)
    transac_bar_t bar_sv_part_done[NUM_BUFS];   // O += S×V 部分完成
    transac_bar_t bar_sv_done[NUM_BUFS];        // O += S×V 完全完成 (V 可释放)
    
    // KV 就绪屏障 (双缓冲, 不使用 Dual GEMM)
    transac_bar_t bar_kv_nope_ready[NUM_BUFS];        // KV NoPE 完整数据就绪
    transac_bar_t bar_kv_rope_ready[NUM_BUFS];        // KV RoPE 数据就绪
    
    // P/dP 空闲屏障 (双缓冲)
    transac_bar_t bar_p_free[NUM_BUFS];         // P 矩阵空闲
    
    // S/O 就绪屏障 (双缓冲)
    transac_bar_t bar_so_ready[NUM_BUFS];       // S 和 O ready (2CTA 同步)
    
    // KV 有效性屏障 (双缓冲)
    transac_bar_t bar_kv_valid_ready[NUM_BUFS];
    transac_bar_t bar_kv_valid_free[NUM_BUFS];
    
    // TMEM 起始地址
    array_aligned<uint32_t, 1> tmem_start_addr;
    
    // Rowwise 缓冲区 (用于 softmax 和 Delta)
    float rowwise_max_buf[128];                 // max logits
    float rowwise_li_buf[128];                  // log-sum-exp
    float rowwise_delta_buf[128];               // Delta = sum(O * dO)
};

}  // namespace sm100::bwd::head128