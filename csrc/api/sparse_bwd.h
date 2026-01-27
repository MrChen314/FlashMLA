#pragma once

#include "common.h"

#include "params.h"

#include "sm100/prefill/sparse/bwd/head128/phase1.h"

// 反向传播特性枚举
enum class BwdFeatures : int {
    HEAD_64,
    HEAD_128,

    HEAD_DIM_576,
    HEAD_DIM_512,

    TOPK_LENGTH
};

// 反向传播实现基类
class BwdImplBase : public ImplBase<
    SparseAttnBwdParams,
    BwdFeatures
> {};

// SM100 Head64 反向实现
class Bwd_Sm100_Head64_Impl : public BwdImplBase {
    DECLARE_SUPPORTED_FEATURES(
        BwdFeatures::HEAD_64,
        BwdFeatures::HEAD_DIM_512,
        BwdFeatures::HEAD_DIM_576,
        BwdFeatures::TOPK_LENGTH
    )

protected:
    void run_(const SparseAttnBwdParams &params, const std::vector<FeatureT> &required_features) override {
        DISPATCH_HEAD_DIM(params.d_qk, HEAD_DIM_QK, [&]() {
            sm100::bwd::head128::run_bwd_phase1_kernel<HEAD_DIM_QK>(params);
        });
    }
};

/**
 * @brief 稀疏注意力反向传播接口函数
 * 
 * @param q Query张量 [s_q, h_q, d_qk], bfloat16
 * @param kv Key/Value张量 [s_kv, h_kv, d_qk], bfloat16
 * @param o 前向输出张量 [s_q, h_q, d_v], bfloat16
 * @param dO 输出梯度张量 [s_q, h_q, d_v], bfloat16
 * @param indices TopK索引 [s_q, h_kv, topk], int32
 * @param lse Log-Sum-Exp [s_q, h_q], float32
 * @param sm_scale Softmax缩放因子
 * @param d_v Value维度 (512)
 * @param topk_length 可选的TopK长度 [s_q], int32
 * @return std::vector<at::Tensor> {dQ, dKV}
 */
static std::vector<at::Tensor> sparse_attn_bwd_interface(
    const at::Tensor &q,
    const at::Tensor &kv,
    const at::Tensor &o,
    const at::Tensor &dO,
    const at::Tensor &indices,
    const at::Tensor &lse,
    float sm_scale,
    int d_v,
    const std::optional<at::Tensor> &topk_length
) {
    using bf16 = cutlass::bfloat16_t;

    Arch arch = Arch();
    bool is_sm100f = arch.is_sm100f();
    TORCH_CHECK(is_sm100f, "Sparse Attention Backward Kernel is only supported on SM100f architecture.");

    // 参数校验
    KU_CHECK_NDIM(q, 3);
    KU_CHECK_NDIM(kv, 3);
    KU_CHECK_NDIM(o, 3);
    KU_CHECK_NDIM(dO, 3);
    KU_CHECK_NDIM(indices, 3);
    KU_CHECK_NDIM(lse, 2);
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

    KU_CHECK_DEVICE(q);
    KU_CHECK_DEVICE(kv);
    KU_CHECK_DEVICE(o);
    KU_CHECK_DEVICE(dO);
    KU_CHECK_DEVICE(indices);
    KU_CHECK_DEVICE(lse);
    KU_CHECK_DEVICE(topk_length);

    KU_CHECK_DTYPE(q, torch::kBFloat16);
    KU_CHECK_DTYPE(kv, torch::kBFloat16);
    KU_CHECK_DTYPE(o, torch::kBFloat16);
    KU_CHECK_DTYPE(dO, torch::kBFloat16);
    KU_CHECK_DTYPE(indices, torch::kInt32);
    KU_CHECK_DTYPE(lse, torch::kFloat32);
    KU_CHECK_DTYPE(topk_length, torch::kInt32);

    KU_CHECK_SHAPE(q, s_q, h_q, d_qk);
    KU_CHECK_SHAPE(kv, s_kv, h_kv, d_qk);
    KU_CHECK_SHAPE(o, s_q, h_q, d_v);
    KU_CHECK_SHAPE(dO, s_q, h_q, d_v);
    KU_CHECK_SHAPE(indices, s_q, h_kv, topk);
    KU_CHECK_SHAPE(lse, s_q, h_q);
    KU_CHECK_SHAPE(topk_length, s_q);

    KU_CHECK_LAST_DIM_CONTIGUOUS(q);
    KU_CHECK_LAST_DIM_CONTIGUOUS(kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(o);
    KU_CHECK_LAST_DIM_CONTIGUOUS(dO);
    KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
    KU_CHECK_LAST_DIM_CONTIGUOUS(lse);
    KU_CHECK_LAST_DIM_CONTIGUOUS(topk_length);

    // 分配输出张量
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto opts = q.options();

    at::Tensor dQ = torch::empty({s_q, h_q, d_qk}, opts);
    at::Tensor dKV = torch::zeros({s_kv, h_kv, d_qk}, opts.dtype(torch::kFloat32));  // float32累加
    KU_CHECK_CONTIGUOUS(dQ);
    KU_CHECK_CONTIGUOUS(dKV);

    SparseAttnBwdParams params = {
        s_q, s_kv, h_q, h_kv, d_qk, d_v, topk,
        sm_scale, sm_scale * LOG_2_E,

        // 输入张量
        (bf16*)q.data_ptr(),
        (bf16*)kv.data_ptr(),
        (bf16*)o.data_ptr(),
        (bf16*)dO.data_ptr(),
        (int*)indices.data_ptr(),
        (float*)lse.data_ptr(),
        ku::get_optional_tensor_ptr<int>(topk_length),

        // Strides
        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)),
        int64_stride_to_int(kv.stride(0)), int64_stride_to_int(kv.stride(1)),
        int64_stride_to_int(o.stride(0)), int64_stride_to_int(o.stride(1)),
        int64_stride_to_int(dO.stride(0)), int64_stride_to_int(dO.stride(1)),
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),

        // 输出张量
        (bf16*)dQ.data_ptr(),
        (float*)dKV.data_ptr(),
        int64_stride_to_int(dQ.stride(0)), int64_stride_to_int(dQ.stride(1)),
        int64_stride_to_int(dKV.stride(0)), int64_stride_to_int(dKV.stride(1)),

        arch.num_sms,
        at::cuda::getCurrentCUDAStream().stream()
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

    // 目前只支持 SM100 Head64
    if (is_sm100f) {
        if (h_q == 64) {
            Bwd_Sm100_Head64_Impl bwd_impl;
            bwd_impl.run(params, required_features);
        } else {
            TORCH_CHECK(false, "Sparse attention backward currently only supports h_q=64. Got h_q=", h_q);
        }
    } else {
        TORCH_CHECK(false, "Unsupported architecture for sparse attention backward");
    }

    // 将 dKV 从 float32 转换为 bfloat16
    at::Tensor dKV_bf16 = dKV.to(torch::kBFloat16);

    return {dQ, dKV_bf16};
}
