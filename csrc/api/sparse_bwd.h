#pragma once

#include "common.h"

#include "params.h"

#include "sm100/prefill/sparse/bwd/head128/phase1.h"

// Backward feature enum
enum class BwdFeatures : int {
    HEAD_128,

    HEAD_DIM_576,

    TOPK_LENGTH
};

// Base class for backward implementations
class BwdImplBase : public ImplBase<
    SparseAttnBwdParams,
    BwdFeatures
> {};

// SM100 Head128 backward implementation
class Bwd_Sm100_Head128Impl : public BwdImplBase {
    DECLARE_SUPPORTED_FEATURES(
        BwdFeatures::HEAD_128,
        BwdFeatures::HEAD_DIM_576,
        BwdFeatures::TOPK_LENGTH
    )

protected:
    void run_(const SparseAttnBwdParams &params, const std::vector<FeatureT> &required_features) override {
        // Only support D_QK == 576 for backward kernel
        sm100::bwd::head128::run_bwd_phase1_kernel<576>(params);
    }
};

/**
 * @brief Sparse attention backward interface function
 * 
 * @param q Query tensor [s_q, h_q, d_qk], bfloat16
 * @param kv Key/Value tensor [s_kv, h_kv, d_qk], bfloat16
 * @param o Forward output tensor [s_q, h_q, d_v], bfloat16
 * @param dO Output gradient tensor [s_q, h_q, d_v], bfloat16
 * @param indices TopK indices [s_q, h_kv, topk], int32
 * @param lse Log-Sum-Exp [s_q, h_q], float32
 * @param sm_scale Softmax scaling factor
 * @param d_v Value dimension (512)
 * @param topk_length Optional TopK length [s_q], int32
 * @param q_start_index_s The starting position of the current chunk in the global sequence (used for causal masking)
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
    const std::optional<at::Tensor> &topk_length,
    int q_start_index_s
) {
    using bf16 = cutlass::bfloat16_t;

    Arch arch = Arch();
    bool is_sm100f = arch.is_sm100f();
    TORCH_CHECK(is_sm100f, "Sparse Attention Backward Kernel is only supported on SM100f architecture.");

    // Parameter validation
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

    TORCH_CHECK(d_qk == 576, "Invalid d_qk: ", d_qk);
    TORCH_CHECK(d_v == 512, "Invalid d_v: ", d_v);
    TORCH_CHECK(q_start_index_s >= 0, "q_start_index_s must be >= 0");
    TORCH_CHECK(h_kv == 1, "Sparse attention backward currently only supports h_kv=1. Got h_kv=", h_kv);
    TORCH_CHECK(topk > 0 && topk % 64 == 0, "Sparse attention backward requires topk to be a positive multiple of 64. Got topk=", topk);

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

    // Allocate output and intermediate tensors
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto opts = q.options();

    // Output tensors
    at::Tensor dQ = torch::empty({s_q, h_q, d_qk}, opts);
    at::Tensor dKV = torch::zeros({s_kv, h_kv, d_qk}, opts.dtype(torch::kFloat32));  // float32 accumulation
    
    // Intermediate tensor: delta is precomputed in preprocess_delta.cuh and consumed by the main kernel.
    at::Tensor delta = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat32));  // Delta = sum(O * dO, dim=-1)
    
    KU_CHECK_CONTIGUOUS(dQ);
    KU_CHECK_CONTIGUOUS(dKV);
    KU_CHECK_CONTIGUOUS(delta);

    SparseAttnBwdParams params = {
        s_q, s_kv, h_q, h_kv, d_qk, d_v, topk,
        q_start_index_s,
        sm_scale, sm_scale * LOG_2_E,

        // Input tensors
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

        // Output tensors
        (bf16*)dQ.data_ptr(),
        (float*)dKV.data_ptr(),
        (float*)delta.data_ptr(),
        int64_stride_to_int(dQ.stride(0)), int64_stride_to_int(dQ.stride(1)),
        int64_stride_to_int(dKV.stride(0)), int64_stride_to_int(dKV.stride(1)),
        int64_stride_to_int(delta.stride(0)), int64_stride_to_int(delta.stride(1)),

        arch.num_sms,
        at::cuda::getCurrentCUDAStream().stream()
    };

    // Build required feature list
    std::vector<BwdFeatures> required_features;
    if (h_q == 128) {
        required_features.push_back(BwdFeatures::HEAD_128);
    } else {
        TORCH_CHECK(false, "Unsupported h_q: ", h_q);
    }
    if (d_qk == 576) {
        required_features.push_back(BwdFeatures::HEAD_DIM_576);
    } else {
        TORCH_CHECK(false, "Unsupported d_qk: ", d_qk);
    }
    if (have_topk_length) {
        required_features.push_back(BwdFeatures::TOPK_LENGTH);
    }

    // Only SM100 Head128 is currently supported
    if (is_sm100f) {
        if (h_q == 128) {
            Bwd_Sm100_Head128Impl bwd_impl;
            bwd_impl.run(params, required_features);
        } else {
            TORCH_CHECK(false, "Sparse attention backward currently only supports h_q=128. Got h_q=", h_q);
        }
    } else {
        TORCH_CHECK(false, "Unsupported architecture for sparse attention backward");
    }

    // Convert dKV from float32 to bfloat16
    at::Tensor dKV_bf16 = dKV.to(torch::kBFloat16);

    return {dQ, dKV_bf16};
}
