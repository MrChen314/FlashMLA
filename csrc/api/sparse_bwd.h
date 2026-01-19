#pragma once

#include "common.h"
#include "params.h"

#include "sm100/prefill/sparse/bwd/head64/phase1.h"

// BWD 特性枚举
enum class BwdFeatures : int {
    HEAD_64,
    HEAD_128,

    HEAD_DIM_576,
    HEAD_DIM_512,

    TOPK_LENGTH
};

// BWD 实现基类
class BwdImplBase : public ImplBase<
    SparseAttnBwdParams,
    BwdFeatures
> {};

// SM100 Head64 BWD 实现
class Bwd_Sm100_Head64_Impl : public BwdImplBase {
    DECLARE_SUPPORTED_FEATURES(
        BwdFeatures::HEAD_64,
        BwdFeatures::HEAD_DIM_512,
        BwdFeatures::HEAD_DIM_576
    )

protected:
    void run_(const SparseAttnBwdParams &params, const std::vector<FeatureT> &required_features) override {
        // 构建 kernel 需要的参数
        sm100::bwd::head64::SparseAttnBwdParams kernel_params = {
            // 输入张量
            (void*)params.q,
            (void*)params.kv,
            (void*)params.out,
            (void*)params.d_out,
            params.indices,
            params.lse,
            params.delta,  // delta = rowsum(O * dO)，已预计算

            // 输出张量
            (void*)params.d_q,
            params.d_kv,

            // 维度
            params.s_q,
            params.s_kv,
            params.h_q,
            params.h_kv,
            params.d_qk,
            params.d_v,
            params.topk,

            // Strides
            params.stride_q_s_q,
            params.stride_q_h_q,
            params.stride_kv_s_kv,
            params.stride_kv_h_kv,
            params.stride_out_s_q,
            params.stride_out_h_q,
            params.stride_dout_s_q,
            params.stride_dout_h_q,
            params.stride_dq_s_q,
            params.stride_dq_h_q,
            params.stride_dkv_s_kv,
            params.stride_dkv_h_kv,
            params.stride_indices_s_q,

            // Scaling factors
            params.sm_scale,
            params.sm_scale_div_log2,

            // Optional
            params.topk_length,

            // Stream
            params.stream
        };

        DISPATCH_HEAD_DIM(params.d_qk, HEAD_DIM_QK, [&]() {
            sm100::bwd::head64::run_bwd_phase1_kernel<HEAD_DIM_QK>(kernel_params);
        });
    }
};

/**
 * 稀疏注意力反向传播接口
 * 
 * @param d_out 输出梯度 [s_q, h_q, d_v], bfloat16
 * @param q Query 张量 [s_q, h_q, d_qk], bfloat16
 * @param kv KV 张量 [s_kv, h_kv, d_qk], bfloat16
 * @param out 前向输出 [s_q, h_q, d_v], bfloat16
 * @param indices 稀疏索引 [s_q, h_kv, topk], int32
 * @param lse Log-sum-exp [s_q, h_q], float32
 * @param delta Delta = rowsum(O * dO) [s_q, h_q], float32，由 Python 端使用 tilelang 预计算
 * @param sm_scale softmax 缩放因子
 * @param d_v Value 维度 (512)
 * @param topk_length 可选的 per-query topk 长度
 * @return tuple<d_q, d_kv> 梯度
 */
static std::vector<at::Tensor> sparse_attn_bwd_interface(
    const at::Tensor &d_out,
    const at::Tensor &q,
    const at::Tensor &kv,
    const at::Tensor &out,
    const at::Tensor &indices,
    const at::Tensor &lse,
    const at::Tensor &delta,
    float sm_scale,
    int d_v,
    const std::optional<at::Tensor> &topk_length
) {
    using bf16 = cutlass::bfloat16_t;

    Arch arch = Arch();
    bool is_sm100f = arch.is_sm100f();
    TORCH_CHECK(is_sm100f, "Sparse Attention Backward Kernel is only supported on SM100f architectures.");

    // 检查输入张量维度
    KU_CHECK_NDIM(d_out, 3);
    KU_CHECK_NDIM(q, 3);
    KU_CHECK_NDIM(kv, 3);
    KU_CHECK_NDIM(out, 3);
    KU_CHECK_NDIM(indices, 3);
    KU_CHECK_NDIM(lse, 2);
    KU_CHECK_NDIM(delta, 2);
    KU_CHECK_NDIM(topk_length, 1);

    int s_q = q.size(0);
    int s_kv = kv.size(0);
    int h_q = q.size(1);
    int h_kv = kv.size(1);
    int d_qk = q.size(2);
    int topk = indices.size(2);
    bool have_topk_length = topk_length.has_value();

    TORCH_CHECK(d_qk == 576 || d_qk == 512, "Invalid d_qk: ", d_qk);
    TORCH_CHECK(d_v == 512, "Invalid d_v: ", d_v);

    // 检查设备
    KU_CHECK_DEVICE(d_out);
    KU_CHECK_DEVICE(q);
    KU_CHECK_DEVICE(kv);
    KU_CHECK_DEVICE(out);
    KU_CHECK_DEVICE(indices);
    KU_CHECK_DEVICE(lse);
    KU_CHECK_DEVICE(delta);
    KU_CHECK_DEVICE(topk_length);

    // 检查数据类型
    KU_CHECK_DTYPE(d_out, torch::kBFloat16);
    KU_CHECK_DTYPE(q, torch::kBFloat16);
    KU_CHECK_DTYPE(kv, torch::kBFloat16);
    KU_CHECK_DTYPE(out, torch::kBFloat16);
    KU_CHECK_DTYPE(indices, torch::kInt32);
    KU_CHECK_DTYPE(lse, torch::kFloat32);
    KU_CHECK_DTYPE(delta, torch::kFloat32);
    KU_CHECK_DTYPE(topk_length, torch::kInt32);

    // 检查形状
    KU_CHECK_SHAPE(d_out, s_q, h_q, d_v);
    KU_CHECK_SHAPE(q, s_q, h_q, d_qk);
    KU_CHECK_SHAPE(kv, s_kv, h_kv, d_qk);
    KU_CHECK_SHAPE(out, s_q, h_q, d_v);
    KU_CHECK_SHAPE(indices, s_q, h_kv, topk);
    KU_CHECK_SHAPE(lse, s_q, h_q);
    KU_CHECK_SHAPE(delta, s_q, h_q);
    KU_CHECK_SHAPE(topk_length, s_q);

    // 检查连续性
    KU_CHECK_LAST_DIM_CONTIGUOUS(d_out);
    KU_CHECK_LAST_DIM_CONTIGUOUS(q);
    KU_CHECK_LAST_DIM_CONTIGUOUS(kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(out);
    KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
    KU_CHECK_LAST_DIM_CONTIGUOUS(lse);
    KU_CHECK_LAST_DIM_CONTIGUOUS(delta);
    KU_CHECK_LAST_DIM_CONTIGUOUS(topk_length);

    // 分配输出张量
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto opts = q.options();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    at::Tensor d_q = torch::empty({s_q, h_q, d_qk}, opts);
    // d_kv 使用 float32 用于原子累加
    at::Tensor d_kv = torch::zeros({s_kv, h_kv, d_qk}, opts.dtype(torch::kFloat));
    KU_CHECK_CONTIGUOUS(d_q);
    KU_CHECK_CONTIGUOUS(d_kv);

    // 构建参数
    SparseAttnBwdParams params = {
        s_q, s_kv, h_q, h_kv, d_qk, d_v, topk,
        sm_scale, sm_scale * LOG_2_E,

        // 输入张量
        (bf16*)q.data_ptr(),
        (bf16*)kv.data_ptr(),
        (bf16*)out.data_ptr(),
        (bf16*)d_out.data_ptr(),
        (int*)indices.data_ptr(),
        (float*)lse.data_ptr(),
        (float*)delta.data_ptr(),

        // 输出张量
        (bf16*)d_q.data_ptr(),
        (float*)d_kv.data_ptr(),

        // Strides
        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)),
        int64_stride_to_int(kv.stride(0)), int64_stride_to_int(kv.stride(1)),
        int64_stride_to_int(out.stride(0)), int64_stride_to_int(out.stride(1)),
        int64_stride_to_int(d_out.stride(0)), int64_stride_to_int(d_out.stride(1)),
        int64_stride_to_int(d_q.stride(0)), int64_stride_to_int(d_q.stride(1)),
        int64_stride_to_int(d_kv.stride(0)), int64_stride_to_int(d_kv.stride(1)),
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),

        ku::get_optional_tensor_ptr<int>(topk_length),
        stream
    };

    // 构建特性列表
    std::vector<BwdFeatures> required_features;
    if (h_q == 64) {
        required_features.push_back(BwdFeatures::HEAD_64);
    } else if (h_q == 128) {
        required_features.push_back(BwdFeatures::HEAD_128);
    } else {
        TORCH_CHECK(false, "Unsupported h_q: ", h_q);
    }
    if (d_qk == 576) {
        required_features.push_back(BwdFeatures::HEAD_DIM_576);
    } else if (d_qk == 512) {
        required_features.push_back(BwdFeatures::HEAD_DIM_512);
    } else {
        TORCH_CHECK(false, "Unsupported d_qk: ", d_qk);
    }
    if (have_topk_length) {
        required_features.push_back(BwdFeatures::TOPK_LENGTH);
    }

    // 执行反向传播
    if (is_sm100f) {
        if (h_q == 64) {
            Bwd_Sm100_Head64_Impl bwd_impl;
            bwd_impl.run(params, required_features);
        } else {
            TORCH_CHECK(false, "Sparse BWD currently only supports h_q=64. Unsupported h_q: ", h_q);
        }
    } else {
        TORCH_CHECK(false, "Unsupported architecture for sparse backward");
    }

    return {d_q, d_kv};
}
