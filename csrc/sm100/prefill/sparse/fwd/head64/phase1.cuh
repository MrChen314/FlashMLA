/**
 * @file phase1.cuh
 * @brief SM100 稀疏注意力前向传播核函数实现 (head_dim=64)
 * 
 * 本文件实现了针对 NVIDIA Blackwell (SM100) 架构优化的稀疏 MLA 注意力前向计算，
 * 使用 TMA (Tensor Memory Accelerator) 和 TMEM (Tensor Memory) 进行高效数据搬运。
 */

#pragma once
#include "phase1.h"

#include <math_constants.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/arch.h>
#include <cutlass/cuda_host_adapter.hpp>

#include <kerutils/kerutils.cuh>

#include "params.h"
#include "utils.h"
#include "sm100/helpers.h"
#include "sm100/prefill/sparse/common_subroutine.h"
#include "config.h"

namespace sm100::fwd::head64 {

using namespace cute;

/*
流水线概览 (Pipeline Overview):
=================================
本核函数采用三阶段软件流水线，实现数据搬运(Copy)、矩阵乘法(MMA)和缩放指数(Scale&Exp)的重叠执行。

| Copy |    MMA    |   Scale & Exp   |

KV0                                      // 预取第0块KV
KV1                                      // 预取第1块KV  
KV2                                      // 预取第2块KV
        P0 = QK0^T                       // 计算注意力分数 P0
                    S0 = exp(P0)         // 对P0应用softmax指数
                    scale(O) w.r.t P0    // 根据新max缩放累加器O
        P1 = QK1^T                       // 计算注意力分数 P1
                    S1 = exp(P1)         // 对P1应用softmax指数
        O += S0V0                        // 累加 S0 @ V0 到输出
KV3                 scale(O) w.r.t P1    // 预取KV3，同时缩放O
        P2 = QK2^T
                    S2 = exp(P2)
        O += S1V1
KV4                 scale(O) w.r.t P2
        P3 = QK3^T
                    S3 = exp(P3)
        O += S2V2
KV5                 scale(O) w.r.t P3

... (持续迭代直到处理完所有K块)

        O += S(n-3)V(n-3)
                    scale(O) w.r.t P(n-2)
        P(n-1) = QK(n-1)^T
                   S(n-1) = exp(P(n-1))
        O += S(n-2)V(n-2)
                   scale(O) w.r.t P(n-1)
        O += S(n-1)V(n-1)                // 最后一次累加
*/

using FwdMode = SparseAttnFwdMode;

/**
 * @brief 稀疏注意力前向核函数
 * @tparam HAVE_ROPE 是否包含RoPE位置编码
 * @tparam TmaParams TMA参数类型
 * @param params 注意力计算参数 (Q, K, V指针, 维度信息等)
 * @param tma_params TMA描述符参数
 * 
 * 线程组织结构:
 * - Grid: [s_q, 1, 1] - 每个block处理一个query token
 * - Block: 384线程 = 3个warpgroup (每个128线程)
 *   - Warpgroup 0: Scale & Exp (softmax计算和输出缩放)
 *   - Warpgroup 1: KV数据生产者 (TMA加载)
 *   - Warpgroup 2: MMA消费者 (矩阵乘法计算)
 */
template<bool HAVE_ROPE, typename TmaParams>
__global__ void __launch_bounds__(NUM_THREADS, 1, 1)
sparse_attn_fwd_kernel(__grid_constant__ const SparseAttnFwdParams params, __grid_constant__ const TmaParams tma_params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
    // Grid形状: [s_q, 1, 1] - 每个block处理一个query位置

    // === 线程索引计算 ===
    const int s_q_idx = blockIdx.x;                                     // 当前处理的query索引
    const int warp_idx = cutlass::canonical_warp_idx_sync();            // warp索引 (0-11)
    const int lane_idx = threadIdx.x % 32;                              // lane索引 (0-31)
    const int warpgroup_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);  // warpgroup索引 (0-2)
    const int idx_in_warpgroup = threadIdx.x % 128;                     // warpgroup内的线程索引
    const int topk_length = params.topk_length != nullptr ? __ldg(params.topk_length + s_q_idx) : params.topk;  // TopK长度
    const int num_k_blocks = max(cute::ceil_div(topk_length, (int)B_TOPK), 1);  // K块数量，至少为1

    // === 共享内存设置 ===
    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);  // 共享内存布局

    int* gIndices = params.indices + s_q_idx*params.stride_indices_s_q; // TopK索引数组指针

    // === TMEM张量分配 ===
    // 注: 这些tXXX张量用于构造布局，使CuTe能在cute::gemm中生成正确的地址
    TiledMMA tiled_mma_P = TiledMMA_P{};  // 用于计算 P = Q @ K^T 的TiledMMA
    TiledMMA tiled_mma_O = TiledMMA_O{};  // 用于计算 O = S @ V 的TiledMMA
    
    // P矩阵片段 (注意力分数)
    Tensor tP = partition_fragment_C(tiled_mma_P, Shape<Int<B_H>, _128>{});
    // Q矩阵NoPE部分的两个分片 (用于分块计算)
    //     TMEM 的 K 维度存储单位:
    // - 每个 "K unit" = 2 个 bf16 元素
    // - 256 个元素 = 128 个 "K units"

    // partition_shape_A 的 K 参数:
    // - 指定 "K units" 的数量
    // - 不是元素的数量

    // 因此:
    // - partition_shape_A: [64, 128]  ← 128 个 K units
    // - 实际矩阵: [64, 256]          ← 256 个元素
    Tensor tQ_nope_part0 = tiled_mma_P.get_slice(_0{}).make_fragment_A(
        partition_shape_A(tiled_mma_P, Shape<Int<B_H>, Int<(D_V/2)/2>>{})
    );
    Tensor tQ_nope_part1 = tiled_mma_P.get_slice(_0{}).make_fragment_A(
        partition_shape_A(tiled_mma_P, Shape<Int<B_H>, Int<(D_V/2)/2>>{})
    );
    // Q矩阵RoPE部分
    Tensor tQ_rope = tiled_mma_P.get_slice(_0{}).make_fragment_A(
        partition_shape_A(tiled_mma_P, Shape<Int<B_H>, Int<64/2>>{})
    );
    // 输出矩阵O的片段
    Tensor tO = partition_fragment_C(tiled_mma_O, Shape<Int<B_H>, Int<D_V>>{});
    
    // 设置TMEM基地址
    tP.data().get() = tmem_cols::P;                // P矩阵的TMEM列起始地址
    tQ_nope_part0.data().get() = tmem_cols::Q;    // Q NoPE第一部分
    tQ_nope_part1.data().get() = tmem_cols::Q + 64;  // Q NoPE第二部分
    tQ_rope.data().get() = tmem_cols::Q_RoPE;     // Q RoPE部分
    tO.data().get() = tmem_cols::O;               // 输出O矩阵

    // === Warp 0: 序言阶段 - 初始化barrier和加载Q ===
    if (warp_idx == 0) {
        if (elect_one_sync()) {  // 只有一个线程执行初始化
            // --- 预取TMA描述符 ---
            if constexpr (HAVE_ROPE) {
                cute::prefetch_tma_descriptor(tma_params.tma_Q_rope.get_tma_descriptor());
            }
            cute::prefetch_tma_descriptor(tma_params.tma_Q_nope.get_tma_descriptor());

            // --- 初始化Q加载的barrier ---
            plan.bar_prologue_q_nope.init(1);
            plan.bar_prologue_q_rope.init(1);
            fence_barrier_init();
            
            // --- 使用TMA加载Q矩阵 ---
            if constexpr (HAVE_ROPE) {
                // 加载Q的RoPE部分: [h_q, d_rope] 从全局内存到共享内存
                Tensor gQ_rope = tma_params.tma_Q_rope.get_tma_tensor(tma_params.shape_Q_rope)(_, _, s_q_idx);
                Tensor sQ_rope = make_tensor(make_smem_ptr(plan.s_q_rope.q_rope.data()), SmemLayoutQRoPE{});
                ku::launch_tma_copy(tma_params.tma_Q_rope, gQ_rope, sQ_rope, plan.bar_prologue_q_rope, TMA::CacheHintSm90::EVICT_FIRST);
            }

            // 加载Q的NoPE部分: [h_q, d_v] 从全局内存到共享内存
            Tensor gQ_nope = tma_params.tma_Q_nope.get_tma_tensor(tma_params.shape_Q_nope)(_, _, s_q_idx);
            Tensor sQ_nope = make_tensor(make_smem_ptr(plan.u.q_full.q_nope.data()), SmemLayoutQNoPE{});
            ku::launch_tma_copy(tma_params.tma_Q_nope, gQ_nope, sQ_nope, plan.bar_prologue_q_nope, TMA::CacheHintSm90::EVICT_FIRST);

            // 预取输出和KV的TMA描述符
            cute::prefetch_tma_descriptor(tma_params.tma_O.get_tma_descriptor());
            cute::prefetch_tma_descriptor(&(tma_params.tensor_map_kv_nope));
            
            // --- 初始化流水线同步barrier ---
            plan.bar_prologue_utccp_rope.init(1);   // UTCCP RoPE完成信号
            plan.bar_prologue_utccp_nope.init(1);   // UTCCP NoPE完成信号
            CUTE_UNROLL
            for (int i = 0; i < NUM_BUFS; ++i) {
                plan.bar_qk_nope_done[i].init(1);   // QK NoPE计算完成
                plan.bar_sv_done[i].init(1);        // SV计算完成
                plan.bar_kv_nope_ready[i][0].init(1);  // KV NoPE第0部分就绪
                plan.bar_kv_nope_ready[i][1].init(1);  // KV NoPE第1部分就绪
                plan.bar_k_valid_ready[i].init(B_TOPK/8);  // K有效性掩码就绪
                plan.bar_k_valid_free[i].init(128);        // K有效性缓冲区空闲
            }
            plan.bar_p_free.init(128);      // P矩阵缓冲区空闲
            plan.bar_so_ready.init(128);    // S和O缩放就绪
            plan.bar_qk_rope_done.init(1);  // QK RoPE计算完成
            plan.bar_kv_rope_ready.init(64);  // KV RoPE数据就绪
            fence_barrier_init();
        }

        // --- 分配TMEM (Tensor Memory) ---
        cute::TMEM::Allocator1Sm().allocate(512, plan.tmem_start_addr.data());
        TRAP_ONLY_DEVICE_ASSERT(plan.tmem_start_addr.data()[0] == 0);  // 确保从地址0开始
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }

    __syncthreads();  // 所有线程同步，确保初始化完成

    // ========================================
    // Warpgroup 0: Scale & Exp 计算单元
    // 负责: softmax计算、输出缩放、结果写回
    // ========================================
    if (warpgroup_idx == 0) {

        // === Online Softmax 状态变量 ===
        // mi: 用于缩放Pi的最大logits (即 O := exp2(Pi*scale - mi) @ V)
        // li: 指数和，即 li := sum(exp(Pi*scale - mi))
        // real_mi: 真实的最大logits，即 real_mi := max(Pi*scale)
        // 其中 Pi 是 P = QK^T 的第i行
        // 注意: mi和real_mi在控制同一行的两个线程间保持一致 (线程0+64, 1+65, 2+66, ...)
        float mi = MAX_INIT_VAL;           // 初始化为一个较大的负值
        float li = 0.0f;                   // 指数和累加器
        float real_mi = -CUDART_INF_F;     // 真实最大值，初始化为负无穷

        // S矩阵在共享内存中的基地址
        bf16* sS_base = plan.s_q_rope.s + lane_idx*8 + (warp_idx&1)*(B_H/2)*8 + (warp_idx/2)*B_H*(B_TOPK/2);
        static constexpr int NUM_ELEMS_PER_THREAD = B_TOPK / 2;  // 每个线程处理的元素数

        // === 主循环: 迭代处理所有K块 ===
        CUTE_NO_UNROLL
        for (int k = 0; k < num_k_blocks; ++k) {
            // --- 等待P矩阵计算完成 ---
            NamedBarrier::arrive_and_wait(64, NamedBarriers::wg0_warp02_sync+(warp_idx&1));
            plan.bar_qk_nope_done[k%NUM_BUFS].wait((k/NUM_BUFS)&1);
            plan.bar_k_valid_ready[k%NUM_BUFS].wait((k/NUM_BUFS)&1);  // 等待K有效性掩码就绪
            ku::tcgen05_after_thread_sync();
            
            // --- 从TMEM加载P矩阵并应用mask ---
            float p[NUM_ELEMS_PER_THREAD];
            retrieve_mask_and_reduce_p<
                NUM_ELEMS_PER_THREAD,
                tmem_cols::P,
                NamedBarriers::wg0_warp02_sync,
                NamedBarriers::wg0_warp13_sync,
                false
            >(
                plan.is_k_valid[k%NUM_BUFS],
                warp_idx, lane_idx, 
                [&]() {plan.bar_p_free.arrive();},  // P缓冲区释放回调
                plan.p_exchange_buf,
                p
            );
            plan.bar_k_valid_free[k%NUM_BUFS].arrive();  // 释放K有效性缓冲区
            
            // --- 计算当前块的行最大值 ---
            float cur_pi_max = get_max<NUM_ELEMS_PER_THREAD>(p);
            cur_pi_max *= params.sm_scale_div_log2;  // 应用softmax缩放因子

            // 跨线程交换最大值 (同一行的两个线程需要一致的max)
            plan.rowwise_max_buf[idx_in_warpgroup] = cur_pi_max;
            NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
            cur_pi_max = max(cur_pi_max, plan.rowwise_max_buf[idx_in_warpgroup^64]);
            real_mi = max(real_mi, cur_pi_max);  // 更新全局真实最大值
            
            // 判断是否需要重新缩放O (阈值6.0对应exp2(6)≈64倍的精度损失)
            bool should_scale_o = __any_sync(0xffffffff, cur_pi_max - mi > 6.0f);
            // 此时: cur_pi_max, real_mi, mi 在同一行的线程间一致
            // should_scale_o 在整个warp内一致


            // --- 计算缩放因子并更新li ---
            float new_max, scale_for_old;
            if (!should_scale_o) {
                // 无需缩放O
                scale_for_old = 1.0f;
                new_max = mi;
            } else {
                // 需要缩放: scale_for_old = exp2(mi_old - mi_new)
                new_max = max(cur_pi_max, mi);
                scale_for_old = exp2f(mi - new_max);
            }
            mi = new_max;  // 更新最大值

            // --- 计算S矩阵: S = exp2(P*scale - max) ---
            nv_bfloat162 s[NUM_ELEMS_PER_THREAD/2];
            float cur_sum = get_s_from_p<NUM_ELEMS_PER_THREAD>(s, p, params.sm_scale_div_log2, new_max);
            li = fma(li, scale_for_old, cur_sum);  // 更新指数和: li = li*scale + cur_sum

            // --- 等待上一轮SV计算完成，然后写入S ---
            if (k > 0) {
                plan.bar_sv_done[(k-1)%NUM_BUFS].wait(((k-1)/NUM_BUFS)&1);
            }
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD/8; i += 1) {
                *(uint128_t*)(sS_base + B_H*8*i) = *(uint128_t*)(s + i*4);  // 写入共享内存
            }

            // --- 根据新的max缩放O矩阵 ---
            if (k > 0 && should_scale_o) {
                ku::tcgen05_after_thread_sync();
                rescale_O<D_V, 32, tmem_cols::O>(scale_for_old);  // O = O * scale_for_old
                ku::tcgen05_before_thread_sync();
            }
            
            fence_view_async_shared();
            plan.bar_so_ready.arrive();  // 通知S和O已准备好
        }

        // ========================================
        // 尾声阶段 (Epilogue): 最终化softmax并写回O
        // ========================================

        // 处理无有效TopK索引的边界情况
        if (real_mi == -CUDART_INF_F) {
            // real_mi == -CUDART_INF_F 表示没有有效的TopK索引
            // 设置li=0以满足定义 li := sum(exp(x[i] - mi))
            li = 0.0f;
            mi = -CUDART_INF_F;
        }
        
        // --- 跨线程交换并累加li ---
        plan.rowwise_li_buf[idx_in_warpgroup] = li;
        NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
        li += plan.rowwise_li_buf[idx_in_warpgroup^64];  // 合并同一行两个线程的li

        // --- 写入max_logits和LSE (Log-Sum-Exp) ---
        if (idx_in_warpgroup < 64) {
            int global_index = s_q_idx*params.h_q + idx_in_warpgroup;
            // LSE = mi*ln(2) + log(li) = log(exp(mi*ln2) * li) = log(sum(exp(Pi*scale)))
            float cur_lse = fmaf(mi, CUDART_LN2_F, logf(li));
            cur_lse = cur_lse == -CUDART_INF_F ? +CUDART_INF_F : cur_lse;  // 处理全mask情况
            params.max_logits[global_index] = real_mi*CUDART_LN2_F;  // 转换为自然对数
            params.lse[global_index] = cur_lse;
        }

        // --- 等待最后一次SV GEMM完成 ---
        plan.bar_sv_done[(num_k_blocks-1)%NUM_BUFS].wait(((num_k_blocks-1)/NUM_BUFS)&1);
        ku::tcgen05_after_thread_sync();

        // --- 计算输出缩放因子并准备写回O ---
        // attention_sink: 用于处理attention sink token的额外项
        float attn_sink = params.attn_sink == nullptr ? -CUDART_INF_F : __ldg(params.attn_sink + (idx_in_warpgroup%64))*CUDART_L2E_F;
        float output_scale = __fdividef(1.0f, li + exp2f(attn_sink - mi));  // 最终的softmax归一化因子
        
        // 设置输出张量布局
        Tensor sO = make_tensor(make_smem_ptr(plan.u.o.data()), SmemLayoutO{});
        constexpr int B_EPI = 64;  // 每次处理的列数
        Tensor tma_gO = flat_divide(
            tma_params.tma_O.get_tma_tensor(tma_params.shape_O)(_, _, s_q_idx),
            Shape<Int<B_H>, Int<B_EPI>>{}
        )(_, _, _0{}, _);
        Tensor sO_divided = flat_divide(
            sO,
            Shape<Int<B_H>, Int<B_EPI>>{}
        )(_, _, _0{}, _);
        auto thr_tma = tma_params.tma_O.get_slice(_0{});

        // 寄存器缓冲区用于临时存储O
        float2 o[B_EPI/2];
        // 检查是否有有效索引，防止部分线程li==0导致TMEM加载死锁
        bool have_valid_indices = __any_sync(0xffffffff, li != 0);
        if (!have_valid_indices) {
            // 无有效索引时，直接输出0
            CUTE_UNROLL
            for (int i = 0; i < B_EPI/2; ++i)
                o[i].x = o[i].y = 0.0f;
            output_scale = 1.0f;
        }

        float2 output_scale_float2 = make_float2(output_scale, output_scale);

        // 预计算共享内存写入地址
        bf16* sO_addrs[8];
        CUTE_UNROLL
        for (int i = 0; i < B_EPI/8; ++i) {
            sO_addrs[i] = &sO(idx_in_warpgroup%64, i*8);
        }

        // --- 分块处理O矩阵: 从TMEM加载、缩放、转换、写回 ---
        CUTE_UNROLL
        for (int c = 0; c < 2; ++c) {
            // 每个tile: 64行 x 256列
            CUTE_UNROLL
            for (int k = 0; k < (D_V/4)/B_EPI; ++k) {
                // 从TMEM加载O
                if (have_valid_indices) {
                    ku::tmem_ld_32dp32bNx<B_EPI>(tmem_cols::O + c*128 + k*B_EPI, o);
                    cutlass::arch::fence_view_async_tmem_load();
                }

                // 缩放并转换为bf16
                CUTE_UNROLL
                for (int i = 0; i < B_EPI/8; ++i) {
                    nv_bfloat162 o_bf16[4];
                    CUTE_UNROLL
                    for (int j = 0; j < 4; ++j) {
                        o[i*4+j] = ku::float2_mul(o[i*4+j], output_scale_float2);  // 应用归一化
                        o_bf16[j] = __float22bfloat162_rn(o[i*4+j]);  // float32 -> bf16
                    }
                    // 写入共享内存
                    *(uint128_t*)(sO_addrs[i] + (c*(D_V/2) + (idx_in_warpgroup/64)*(D_V/4) + k*B_EPI)*64) = *(uint128_t*)(o_bf16);
                }

                // 同步后使用TMA写回全局内存
                fence_view_async_shared();
                NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
                
                // Warp 0负责写入前半部分
                if (warp_idx == 0 && elect_one_sync()) {
                    int epi_chunk_idx = c*(D_V/2/B_EPI) + k;
                    cute::copy(
                        tma_params.tma_O,
                        thr_tma.partition_S(sO_divided(_, _, epi_chunk_idx)),
                        thr_tma.partition_D(tma_gO(_, _, epi_chunk_idx))
                    );
                }
                // Warp 1负责写入后半部分
                if (warp_idx == 1 && elect_one_sync()) {
                    int epi_chunk_idx = c*(D_V/2/B_EPI) + (D_V/B_EPI/4) + k;
                    cute::copy(
                        tma_params.tma_O,
                        thr_tma.partition_S(sO_divided(_, _, epi_chunk_idx)),
                        thr_tma.partition_D(tma_gO(_, _, epi_chunk_idx))
                    );
                }
            }
        }

        // --- 释放TMEM ---
        if (warp_idx == 0) {
            cute::TMEM::Allocator1Sm().free(0, 512);
        }
    // ========================================
    // Warpgroup 1: KV数据生产者
    // 负责: 使用TMA从全局内存加载KV到共享内存
    // ========================================
    } else if (warpgroup_idx == 1) {
        int warp_idx = cutlass::canonical_warp_idx_sync() - 4;  // 本warpgroup内的warp索引
        constexpr int NUM_WARPS = 4, NUM_LOCAL_ROWS_PER_WARP = (B_TOPK/4)/NUM_WARPS;  // 每个warp处理的行数
        
        if (elect_one_sync()) {  // 每个warp只需一个线程发起TMA
            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks; ++k) {
                // --- 加载TopK索引 ---
                int4 indices[NUM_LOCAL_ROWS_PER_WARP];  // 每个int4包含4个索引
                int max_indices = -1, min_indices = params.s_kv;
                CUTE_UNROLL
                for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                    indices[local_row] = __ldg((int4*)(gIndices + k*B_TOPK) + local_row*NUM_WARPS + warp_idx);
                    max_indices = max(max_indices, int4_max(indices[local_row]));
                    min_indices = min(min_indices, int4_min(indices[local_row]));
                }
                // 检查是否所有索引都无效 (用于跳过无用的TMA)
                bool is_all_rows_invalid = min_indices == params.s_kv || max_indices == -1;
                bool should_skip_tma = is_all_rows_invalid && k >= NUM_BUFS;

                // Q NoPE和K[2]共享同一块内存，需要等待Q NoPE拷贝完成
                if (k == 2) {
                    plan.bar_prologue_utccp_nope.wait(0);
                }

                // --- 使用TMA gather加载KV NoPE部分 ---
                int cur_buf = k%NUM_BUFS;  // 双缓冲索引
                plan.bar_sv_done[cur_buf].wait((k/NUM_BUFS)&1^1);  // 等待上一轮SV完成
                bf16* sK_nope_base = plan.u.k.k_nope[cur_buf].data() + warp_idx*4*64;

                // Lambda: 加载KV NoPE的一个分片
                auto load_kv_nope_part = [&](int part_idx) {
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                        CUTE_UNROLL
                        for (int local_col = part_idx*(D_V/2/64); local_col < (part_idx+1)*(D_V/2/64); ++local_col) {
                            // TMA gather: 根据indices从非连续位置加载数据
                            ku::tma_gather4(
                                &(tma_params.tensor_map_kv_nope),
                                plan.bar_kv_nope_ready[cur_buf][part_idx],
                                sK_nope_base + local_row*(4*NUM_WARPS)*64 + local_col*(B_TOPK*64),
                                local_col*64,
                                indices[local_row],
                                (int64_t)TMA::CacheHintSm90::EVICT_LAST
                            );
                        }
                    }
                };

                if (!should_skip_tma) {
                    // 正常加载KV NoPE的两个部分
                    load_kv_nope_part(0);
                    load_kv_nope_part(1);
                } else {
                    // 跳过TMA但仍需完成barrier事务 (参见head128/phase1.cuh)
                    CUTE_UNROLL
                    for (int part_idx = 0; part_idx < 2; ++part_idx)
                        plan.bar_kv_nope_ready[cur_buf][part_idx].complete_transaction(NUM_LOCAL_ROWS_PER_WARP*4*D_V/2*sizeof(bf16));
                }
            }
        }
    // ========================================
    // Warpgroup 2: MMA计算单元
    // 负责: UTCCP数据搬运、P=QK^T计算、O+=SV计算
    // ========================================
    } else {
        // === Warp 8: MMA主控warp (UTCCP + GEMM) ===
        if (warp_idx == 8 && elect_one_sync()) {
            // --- 构建UMMA描述符 (用于SMEM->TMEM拷贝) ---
            // Q NoPE描述符: 用于双GEMM布局
            UMMA::SmemDescriptor sQ_nope_desc = UMMA::make_umma_desc<UMMA::Major::K>(
                make_tensor(
                    make_smem_ptr(plan.u.q_full.q_nope.data()),
                    tile_to_shape(
                        UMMA::Layout_K_SW128_Atom<bf16>{},
                        Shape<Int<B_H*2>, Int<64>>{}    // 双GEMM使用的形状
                    )
                )
            );
            // Q RoPE描述符
            UMMA::SmemDescriptor sQ_rope_desc = UMMA::make_umma_desc<UMMA::Major::K>(
                make_tensor(
                    make_smem_ptr(plan.s_q_rope.q_rope.data()),
                    tile_to_shape(
                        UMMA::Layout_K_SW64_Atom<bf16>{},
                        Shape<Int<B_H*2>, Int<32>>{}
                    )
                )
            );
            
            // --- UTCCP: 将Q从SMEM拷贝到TMEM ---
            if constexpr (HAVE_ROPE) {
                // 拷贝RoPE tile: UTCCP视角128行*32列(64B), 实际64行*64列
                plan.bar_prologue_q_rope.arrive_and_expect_tx(B_H*(D_Q-D_V)*sizeof(bf16));
                plan.bar_prologue_q_rope.wait(0);
                ku::tcgen05_after_thread_sync();
                CUTE_UNROLL
                for (int subtile_idx = 0; subtile_idx < 2; ++subtile_idx) {
                    // 每个subtile: UTCCP视角128行*16列(256bit=32B), 实际64行*16列*2
                    SM100_UTCCP_128dp256bit_1cta::copy(
                        sQ_rope_desc + (subtile_idx*32) / 16,
                        tmem_cols::Q_RoPE + subtile_idx*8
                    );
                }
                ku::umma_arrive_noelect(plan.bar_prologue_utccp_rope);
            }

            // 拷贝NoPE tile: 等待TMA完成后执行UTCCP
            plan.bar_prologue_q_nope.arrive_and_expect_tx(B_H*D_V*sizeof(bf16));
            plan.bar_prologue_q_nope.wait(0);
            ku::tcgen05_after_thread_sync();
            CUTE_UNROLL
            for (int tile_idx = 0; tile_idx < D_V/64/2; ++tile_idx) {
                // 每个tile: UTCCP视角128行*64列(128B), 实际64行*128列
                CUTE_UNROLL
                for (int subtile_idx = 0; subtile_idx < 4; ++subtile_idx) {
                    // 每个subtile: UTCCP视角128行*16列(256bit=32B)
                    SM100_UTCCP_128dp256bit_1cta::copy(
                        sQ_nope_desc + (tile_idx*(B_H*128*2) + subtile_idx*32) / 16,  // 4 LSBs不包含在地址中
                        tmem_cols::Q + tile_idx*32 + subtile_idx*8
                    );
                }
            }
            ku::umma_arrive_noelect(plan.bar_prologue_utccp_nope);

            if constexpr (HAVE_ROPE) {
                plan.bar_prologue_utccp_rope.wait(0);
            }

            // --- 主计算循环: P=QK^T 和 O+=SV ---
            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks+1; ++k) {
                // ---- 计算 Pi = Q @ Ki^T ----
                if (k < num_k_blocks) {
                    int cur_buf = k%NUM_BUFS;
                    Tensor sK_nope = make_tensor(make_smem_ptr(plan.u.k.k_nope[cur_buf].data()), SmemLayoutKNoPE_TiledMMA{});
                    Tensor sK_rope = make_tensor(make_smem_ptr(plan.u.k.k_rope.data()), SmemLayoutKRoPE_TiledMMA{});

                    plan.bar_p_free.wait(k&1^1);  // 等待P缓冲区可用
                    ku::tcgen05_after_thread_sync();
                    
                    // 计算 P = Q(rope) @ K(rope)^T
                    if constexpr (HAVE_ROPE) {
                        plan.bar_kv_rope_ready.wait(k&1);  // 等待K RoPE数据就绪
                        ku::tcgen05_after_thread_sync();
                        ku::utcmma_ts(tiled_mma_P, tQ_rope, sK_rope, tP, true);  // TMEM-SMEM MMA
                        ku::umma_arrive_noelect(plan.bar_qk_rope_done);
                    }

                    // 计算 P += Q(nope) @ K(nope)^T
                    if (k == 0) {
                        plan.bar_prologue_utccp_nope.wait(0);  // 首次迭代等待Q NoPE拷贝完成
                    }
                    Tensor sK_nope_divided = flat_divide(sK_nope, Tile<Int<B_TOPK*2>, Int<D_V/4>>{})(_, _, _0{}, _);
                    CUTE_UNROLL
                    for (int kv_nope_part_idx = 0; kv_nope_part_idx < 2; ++kv_nope_part_idx) {
                        plan.bar_kv_nope_ready[cur_buf][kv_nope_part_idx].arrive_and_expect_tx(B_TOPK*D_V/2*sizeof(bf16));
                        plan.bar_kv_nope_ready[cur_buf][kv_nope_part_idx].wait((k/NUM_BUFS)&1);
                        ku::tcgen05_after_thread_sync();

                        // 分两次累加NoPE部分的乘积
                        bool clear_accum = (!HAVE_ROPE) && kv_nope_part_idx == 0;  // 首次清零累加器
                        ku::utcmma_ts(tiled_mma_P, kv_nope_part_idx ? tQ_nope_part1 : tQ_nope_part0, sK_nope_divided(_, _, kv_nope_part_idx), tP, clear_accum);
                    }
                    ku::umma_arrive_noelect(plan.bar_qk_nope_done[cur_buf]);  // 通知P计算完成
                }
                
                // ---- 计算 O += S(i-1) @ V(i-1) ----
                if (k > 0) {
                    int cur_buf = (k-1)%NUM_BUFS;

                    Tensor sS = make_tensor(make_smem_ptr(plan.s_q_rope.s), SmemLayoutS{});
                    Tensor sV = make_tensor(make_smem_ptr(plan.u.k.k_nope[cur_buf].data()), SmemLayoutV{});

                    // 等待S计算完成且O已缩放
                    plan.bar_so_ready.wait((k-1)&1);
                    ku::tcgen05_after_thread_sync();

                    // O += S @ V (SMEM-SMEM MMA)
                    ku::utcmma_ss(tiled_mma_O, sS, sV, tO, k == 1);  // k==1时清零O
                    ku::umma_arrive_noelect(plan.bar_sv_done[cur_buf]);  // 通知SV计算完成
                }
            }
        // === Warp 9: K有效性掩码加载warp ===
        } else if (warp_idx == 9) {
            // 只有前B_TOPK/8个lane参与工作
            if (lane_idx < B_TOPK/8) {
                CUTE_NO_UNROLL
                for (int k = 0; k < num_k_blocks; ++k) {
                    // 加载索引并生成有效性掩码 (用于mask无效的TopK位置)
                    char k_validness_mask = load_indices_and_generate_mask(
                        lane_idx,
                        gIndices + k*B_TOPK,
                        params.s_kv,
                        k*B_TOPK,
                        topk_length
                    );

                    // 写入有效性掩码到共享内存
                    int cur_buf = k%NUM_BUFS;
                    plan.bar_k_valid_free[cur_buf].wait((k/NUM_BUFS)&1^1);  // 等待缓冲区空闲
                    plan.is_k_valid[cur_buf][lane_idx] = k_validness_mask;
                    plan.bar_k_valid_ready[cur_buf].arrive();  // 通知掩码就绪
                }
            }
        // === Warp 10-11: K RoPE数据加载warp ===
        } else if (warp_idx == 10 || warp_idx == 11) {
            if constexpr (HAVE_ROPE) {
                int thread_idx = threadIdx.x - 10*32;  // 在这两个warp内的线程索引
                constexpr int GROUP_SIZE = 8, NUM_GROUPS = 64/GROUP_SIZE, ROWS_PER_THREAD = B_TOPK/NUM_GROUPS;
                int group_idx = thread_idx / GROUP_SIZE, idx_in_group = thread_idx % GROUP_SIZE;
                
                Tensor sK_rope = make_tensor(make_smem_ptr(plan.u.k.k_rope.data()), SmemLayoutKRoPE{});
                bf16* sK_rope_base = &sK_rope(group_idx, idx_in_group*8);
                
                CUTE_NO_UNROLL
                for (int k = 0; k < num_k_blocks; ++k) {
                    // 预加载当前块的索引
                    int indices[ROWS_PER_THREAD];
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < ROWS_PER_THREAD; ++local_row)
                        indices[local_row] = __ldg(gIndices + k*B_TOPK + group_idx + local_row*NUM_GROUPS);
                    
                    // 等待上一轮QK RoPE计算完成
                    plan.bar_qk_rope_done.wait(k&1^1);
                    
                    // 使用cp.async加载K RoPE数据 (比TMA更快)
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < ROWS_PER_THREAD; ++local_row) {
                        int index = indices[local_row];
                        ku::cp_async_cacheglobal<ku::PrefetchSize::B128>(
                            params.kv + (int64_t)index*params.stride_kv_s_kv + 512 + idx_in_group*8,  // RoPE部分在offset 512处
                            sK_rope_base + local_row*NUM_GROUPS*32,
                            index >= 0 && index < params.s_kv  // 有效性检查
                        );
                        // 注: 这里只检查index范围，不检查topk_length，避免潜在的NaN问题
                    }
                    cutlass::arch::cpasync_barrier_arrive_noinc((uint64_t*)&(plan.bar_kv_rope_ready));
                }
            }
        }
    }


#else
    // 非SM100架构的错误处理
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm100");
    }
#endif
}

/**
 * @brief 启动前向Phase1核函数的主机端包装函数
 * @tparam D_QK Query/Key的维度 (576或512)
 * @param params 注意力计算参数结构体
 * 
 * 功能:
 * 1. 参数校验
 * 2. 创建TMA描述符 (Q_nope, Q_rope, O, KV)
 * 3. 配置并启动CUDA核函数
 */
template<int D_QK>
void run_fwd_phase1_kernel(const SparseAttnFwdParams& params) {
    // === 参数校验 ===
    KU_ASSERT(params.h_kv == 1);            // KV head数必须为1 (MLA特性)
    KU_ASSERT(params.topk % B_TOPK == 0);   // TopK必须是B_TOPK的倍数 (省去边界检查)
    KU_ASSERT(params.h_q == B_H);           // Query head数必须等于B_H
    KU_ASSERT(params.d_qk == D_QK);
    static_assert(D_QK == 576 || D_QK == 512);  // 仅支持这两种QK维度

    // === 创建Q NoPE的TMA描述符 ===
    // Q NoPE形状: [h_q, d_v, s_q]
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

    // === 创建Q RoPE的TMA描述符 ===
    // Q RoPE形状: [h_q, d_qk-d_v, s_q]，从Q的offset D_V处开始
    auto shape_Q_rope = make_shape(params.h_q, D_Q-D_V, params.s_q);
    auto tma_Q_rope = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.q + D_V),  // RoPE部分从D_V偏移开始
            make_layout(
                shape_Q_rope,
                make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q)
            )
        ),
        SmemLayoutQRoPE{}
    );

    // === 创建输出O的TMA描述符 ===
    // O形状: [h_q, d_v, s_q]
    auto shape_O = make_shape(params.h_q, params.d_v, params.s_q);
    auto tma_O = cute::make_tma_copy(
        SM90_TMA_STORE{},
        make_tensor(
            make_gmem_ptr((bf16*)params.out),
            make_layout(
                shape_O,
                make_stride(params.d_v, _1{}, params.h_q*params.d_v)
            )
        ),
        SmemLayoutOTiles<1>{}
    );

    // === 创建KV NoPE的TensorMap (用于TMA gather) ===
    CUtensorMap tensor_map_kv_nope;
    {
        uint64_t size[2] = {D_V, (unsigned long)params.s_kv};  // [d_v, s_kv]
        uint64_t stride[1] = {params.stride_kv_s_kv*sizeof(bf16)};  // 行stride
        uint32_t box_size[2] = {64, 1};    // 每次加载的box大小: 64列 x 1行
        uint32_t elem_stride[2] = {1, 1};  // 元素stride
        CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &tensor_map_kv_nope,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,                                                      // 2D张量
            params.kv,                                              // 全局内存指针
            size,
            stride,
            box_size,
            elem_stride,
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,   // 无交错
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,         // 128字节swizzle
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,  // L2缓存提升
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE   // 越界不填充
        );
        KU_ASSERT(res == CUresult::CUDA_SUCCESS);
    }

    // === 组装TMA参数结构体 ===
    TmaParams<
        decltype(shape_Q_nope), decltype(tma_Q_nope),
        decltype(shape_Q_rope), decltype(tma_Q_rope),
        decltype(shape_O), decltype(tma_O)
    > tma_params = {
        shape_Q_nope, tma_Q_nope,
        shape_Q_rope, tma_Q_rope,
        shape_O, tma_O,
        tensor_map_kv_nope
    };
    
    // D_QK==576时启用RoPE (576 = 512 + 64)
    auto kernel = &sparse_attn_fwd_kernel<D_QK == 576, decltype(tma_params)>;

    // === 配置并启动核函数 ===
    constexpr size_t smem_size = sizeof(SharedMemoryPlan);  // 动态共享内存大小
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    // Grid: s_q个block，每个block处理一个query token
    // Block: NUM_THREADS (384)个线程
    kernel<<<params.s_q, NUM_THREADS, smem_size, params.stream>>>(params, tma_params);
    KU_CHECK_KERNEL_LAUNCH();
}

}  // namespace sm100::fwd::head64
