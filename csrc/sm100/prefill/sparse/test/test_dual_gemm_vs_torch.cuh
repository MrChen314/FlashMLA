#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "../fwd/head64/config.h"
#include "../common_subroutine.h"

namespace dual_gemm_test {

using kerutils::bf16;

// 使用 fwd/head64 的配置尺寸
constexpr int B_H = sm100::fwd::head64::B_H;
constexpr int B_TOPK = sm100::fwd::head64::B_TOPK;
constexpr int D_V = sm100::fwd::head64::D_V;

inline void cuda_check(cudaError_t status, const char* file, int line, const char* expr) {
    if (status != cudaSuccess) {
        std::cerr << "CUDA 错误: " << expr << " (" << file << ":" << line << "): "
                  << cudaGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expr) ::dual_gemm_test::cuda_check((expr), __FILE__, __LINE__, #expr)

// bf16 <-> float 转换
__host__ __device__ inline bf16 float_to_bf16(float v) {
    return bf16(v);
}

__host__ __device__ inline float bf16_to_float(bf16 v) {
    return static_cast<float>(v);
}

struct ErrorStats {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    float mean_abs = 0.0f;
    int max_row = -1;
    int max_col = -1;
    float ref_at_max = 0.0f;
    float out_at_max = 0.0f;
};

// 空操作回调，用于替代设备端 lambda
struct EmptyArrive {
    __device__ void operator()() const {}
};

// 共享内存布局，保持与真实流程一致
struct alignas(16) SharedStorage {
    cute::array_aligned<bf16, cute::cosize_v<sm100::fwd::head64::SmemLayoutQNoPE>> q_nope;
    cute::array_aligned<bf16, cute::cosize_v<sm100::fwd::head64::SmemLayoutKNoPE>> k_nope;
    char k_valid[B_TOPK / 8];
    float p_exchange_buf[4][32 * (B_TOPK / 2)];
    cute::array_aligned<uint32_t, 1> tmem_start_addr;
};

// 严格按流程：Q -> SMEM -> TMEM，K -> SMEM，然后 utcmma_ts
__global__ void dual_gemm_utcmma_ts_kernel(const bf16* __restrict__ q,
                                           const bf16* __restrict__ k,
                                           float* __restrict__ out) {
    using namespace cute;
    using namespace sm100::fwd::head64;

    extern __shared__ unsigned char smem_buf[];
    auto* smem = reinterpret_cast<SharedStorage*>(smem_buf);

    const int thread_idx = threadIdx.x;
    const int warp_idx = thread_idx / 32;
    const int lane_idx = thread_idx % 32;

    // 1) Q 从全局内存写入 SMEM
    Tensor sQ_nope = make_tensor(make_smem_ptr(smem->q_nope.data()), SmemLayoutQNoPE{});
    for (int idx = thread_idx; idx < B_H * D_V; idx += blockDim.x) {
        int row = idx / D_V;
        int col = idx % D_V;
        sQ_nope(row, col) = q[idx];
    }

    // 2) K 从全局内存写入 SMEM
    Tensor sK_nope = make_tensor(make_smem_ptr(smem->k_nope.data()), SmemLayoutKNoPE{});
    for (int idx = thread_idx; idx < B_TOPK * D_V; idx += blockDim.x) {
        int row = idx / D_V;
        int col = idx % D_V;
        sK_nope(row, col) = k[idx];
    }

    // 3) 有效性掩码全部置为有效
    if (thread_idx < B_TOPK / 8) {
        smem->k_valid[thread_idx] = static_cast<char>(0xFF);
    }
    __syncthreads();

    // 4) 分配 TMEM
    if (thread_idx == 0) {
        cute::TMEM::Allocator1Sm().allocate(512, smem->tmem_start_addr.data());
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }
    __syncthreads();

    // 5) 构建 TMEM 片段并设置地址
    TiledMMA tiled_mma_P = TiledMMA_P{};
    Tensor tP = partition_fragment_C(tiled_mma_P, Shape<Int<B_H>, _128>{});
    Tensor tQ_nope_part0 = tiled_mma_P.get_slice(_0{}).make_fragment_A(
        partition_shape_A(tiled_mma_P, Shape<Int<B_H>, Int<(D_V / 2) / 2>>{})
    );
    Tensor tQ_nope_part1 = tiled_mma_P.get_slice(_0{}).make_fragment_A(
        partition_shape_A(tiled_mma_P, Shape<Int<B_H>, Int<(D_V / 2) / 2>>{})
    );
    tP.data().get() = tmem_cols::P;
    tQ_nope_part0.data().get() = tmem_cols::Q;
    tQ_nope_part1.data().get() = tmem_cols::Q + 64;

    // 6) Q 从 SMEM -> TMEM (UTCCP)
    if (thread_idx == 0) {
        UMMA::SmemDescriptor sQ_nope_desc = UMMA::make_umma_desc<UMMA::Major::K>(
            make_tensor(
                make_smem_ptr(smem->q_nope.data()),
                tile_to_shape(
                    UMMA::Layout_K_SW128_Atom<bf16>{},
                    Shape<Int<B_H * 2>, Int<64>>{}
                )
            )
        );
        ku::tcgen05_after_thread_sync();
        CUTE_UNROLL
        for (int tile_idx = 0; tile_idx < D_V / 64 / 2; ++tile_idx) {
            CUTE_UNROLL
            for (int subtile_idx = 0; subtile_idx < 4; ++subtile_idx) {
                SM100_UTCCP_128dp256bit_1cta::copy(
                    sQ_nope_desc + (tile_idx * (B_H * 128 * 2) + subtile_idx * 32) / 16,
                    tmem_cols::Q + tile_idx * 32 + subtile_idx * 8
                );
            }
        }
    }
    __syncthreads();
    ku::tcgen05_after_thread_sync();

    // 7) TMEM-SMEM MMA：P = Q0*K0^T + Q1*K1^T
    if (thread_idx == 0) {
        Tensor sK_nope_tiled = make_tensor(make_smem_ptr(smem->k_nope.data()), SmemLayoutKNoPE_TiledMMA{});
        Tensor sK_nope_divided = flat_divide(sK_nope_tiled, Tile<Int<B_TOPK * 2>, Int<D_V / 4>>{})(_, _, _0{}, _);
        CUTE_UNROLL
        for (int kv_nope_part_idx = 0; kv_nope_part_idx < 2; ++kv_nope_part_idx) {
            bool clear_accum = kv_nope_part_idx == 0;
            ku::utcmma_ts(
                tiled_mma_P,
                kv_nope_part_idx ? tQ_nope_part1 : tQ_nope_part0,
                sK_nope_divided(_, _, kv_nope_part_idx),
                tP,
                clear_accum
            );
        }
    }
    __syncthreads();
    ku::tcgen05_after_thread_sync();

    // 8) 从 TMEM 取回并归约双 GEMM 的两半
    constexpr int NUM_ELEMS_PER_THREAD = B_TOPK / 2;
    float p[NUM_ELEMS_PER_THREAD];
    sm100::retrieve_mask_and_reduce_p<
        NUM_ELEMS_PER_THREAD,
        tmem_cols::P,
        NamedBarriers::wg0_warp02_sync,
        NamedBarriers::wg0_warp13_sync,
        false
    >(
        smem->k_valid,
        warp_idx,
        lane_idx,
        EmptyArrive{},
        smem->p_exchange_buf,
        p
    );

    // 9) 写回全局内存，映射到 [B_H, B_TOPK]
    int row = lane_idx + (warp_idx & 1) * (B_H / 2);
    int col_base = (warp_idx >= 2) ? (B_TOPK / 2) : 0;
    if (row < B_H) {
        float* out_row = out + row * B_TOPK + col_base;
        CUTE_UNROLL
        for (int i = 0; i < NUM_ELEMS_PER_THREAD; ++i) {
            out_row[i] = p[i];
        }
    }

    // 10) 释放 TMEM
    __syncthreads();
    if (thread_idx == 0) {
        cute::TMEM::Allocator1Sm().free(0, 512);
    }
}

inline std::vector<bf16> to_bf16(const std::vector<float>& src) {
    std::vector<bf16> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = float_to_bf16(src[i]);
    }
    return dst;
}

// 参考实现：等价于 torch.matmul（bf16 输入 + float 累加）
inline void matmul_reference(const std::vector<bf16>& q,
                             const std::vector<bf16>& k,
                             std::vector<float>* out) {
    out->assign(B_H * B_TOPK, 0.0f);
    for (int row = 0; row < B_H; ++row) {
        for (int col = 0; col < B_TOPK; ++col) {
            float acc = 0.0f;
            int q_base = row * D_V;
            int k_base = col * D_V;
            for (int d = 0; d < D_V; ++d) {
                acc += bf16_to_float(q[q_base + d]) * bf16_to_float(k[k_base + d]);
            }
            (*out)[row * B_TOPK + col] = acc;
        }
    }
}

inline ErrorStats compare_outputs(const std::vector<float>& ref,
                                  const std::vector<float>& out) {
    ErrorStats stats;
    const float denom_floor = 1e-2f;
    const int total = B_H * B_TOPK;
    for (int idx = 0; idx < total; ++idx) {
        float abs_err = std::fabs(out[idx] - ref[idx]);
        float denom = std::max(std::fabs(ref[idx]), denom_floor);
        float rel_err = abs_err / denom;
        stats.mean_abs += abs_err;
        if (abs_err > stats.max_abs) {
            stats.max_abs = abs_err;
            stats.max_row = idx / B_TOPK;
            stats.max_col = idx % B_TOPK;
            stats.ref_at_max = ref[idx];
            stats.out_at_max = out[idx];
        }
        if (rel_err > stats.max_rel) {
            stats.max_rel = rel_err;
        }
    }
    stats.mean_abs /= static_cast<float>(total);
    return stats;
}

inline bool run_dual_gemm_vs_matmul(uint64_t seed, float abs_tol, float rel_tol, bool verbose = true) {
    if (verbose) {
        std::cout << "============================================\n";
        std::cout << "Dual GEMM vs torch.matmul 精度对比\n";
        std::cout << "形状: Q[" << B_H << "," << D_V << "], K[" << B_TOPK << "," << D_V << "]\n";
        std::cout << "随机种子: " << seed << "\n";
        std::cout << "容差: abs=" << abs_tol << ", rel=" << rel_tol << "\n";
        std::cout << "============================================\n";
    }

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> q_f(B_H * D_V);
    std::vector<float> k_f(B_TOPK * D_V);
    for (float& v : q_f) {
        v = dist(rng);
    }
    for (float& v : k_f) {
        v = dist(rng);
    }

    std::vector<bf16> q_bf16 = to_bf16(q_f);
    std::vector<bf16> k_bf16 = to_bf16(k_f);

    std::vector<float> ref;
    matmul_reference(q_bf16, k_bf16, &ref);

    bf16* d_q = nullptr;
    bf16* d_k = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q, q_bf16.size() * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&d_k, k_bf16.size() * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&d_out, ref.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_q, q_bf16.data(), q_bf16.size() * sizeof(bf16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, k_bf16.data(), k_bf16.size() * sizeof(bf16), cudaMemcpyHostToDevice));

    size_t shared_bytes = sizeof(SharedStorage);
    CUDA_CHECK(cudaFuncSetAttribute(
        dual_gemm_utcmma_ts_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes)
    ));

    dim3 block(128);
    dim3 grid(1);
    dual_gemm_utcmma_ts_kernel<<<grid, block, shared_bytes>>>(d_q, d_k, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> out(ref.size(), 0.0f);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_out));

    ErrorStats stats = compare_outputs(ref, out);
    bool pass = (stats.max_abs <= abs_tol) && (stats.max_rel <= rel_tol);

    if (verbose) {
        std::cout << "最大绝对误差: " << stats.max_abs
                  << " (位置: " << stats.max_row << "," << stats.max_col << ")\n";
        std::cout << "最大相对误差: " << stats.max_rel << "\n";
        std::cout << "平均绝对误差: " << stats.mean_abs << "\n";
        std::cout << "参考值/dualgemm: " << stats.ref_at_max << " / " << stats.out_at_max << "\n";
        std::cout << "结果: " << (pass ? "通过" : "失败") << "\n";
    }

    return pass;
}

}  // namespace dual_gemm_test
