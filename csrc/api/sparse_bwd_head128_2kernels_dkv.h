#pragma once

#include "sparse_bwd.h"

#include "sm100/prefill/sparse/bwd/head128_2kernels/phase.h"

class Bwd_Sm100_Head128_2KernelsDKVImpl : public BwdImplBase {
    DECLARE_SUPPORTED_FEATURES(
        BwdFeatures::HEAD_128,
        BwdFeatures::HEAD_DIM_576,
        BwdFeatures::TOPK_LENGTH
    )

protected:
    void run_(const SparseAttnBwdParams &params, const std::vector<FeatureT> &required_features) override {
        sm100::bwd::head128_2kernels::dkv::run_bwd_dkv_phase_kernel<576>(params);
    }
};

struct SparseBwdHead128_2KernelsDKVLaunchContext {
    int h_q;
    int d_qk;
    bool have_topk_length;
    at::Tensor dKV;
    SparseAttnBwdParams params;
};

static SparseBwdHead128_2KernelsDKVLaunchContext prepare_sparse_attn_bwd_head128_2kernels_dkv_context(
    const at::Tensor &q,
    const at::Tensor &dO,
    const at::Tensor &indices,
    const at::Tensor &s,
    const at::Tensor &ds,
    int s_kv,
    int d_v,
    const std::optional<at::Tensor> &topk_length,
    int q_start_index_s,
    int num_sm,
    cudaStream_t stream
) {
    using bf16 = cutlass::bfloat16_t;

    KU_CHECK_NDIM(q, 3);
    KU_CHECK_NDIM(dO, 3);
    KU_CHECK_NDIM(indices, 3);
    KU_CHECK_NDIM(s, 3);
    KU_CHECK_NDIM(ds, 3);
    KU_CHECK_NDIM(topk_length, 1);

    int s_q = q.size(0);
    int h_q = q.size(1);
    int d_qk = q.size(2);
    int h_kv = indices.size(1);
    int topk = indices.size(2);
    bool have_topk_length = topk_length.has_value();

    TORCH_CHECK(s_kv > 0, "s_kv must be > 0. Got s_kv=", s_kv);
    TORCH_CHECK(d_qk == 576, "Invalid d_qk: ", d_qk);
    TORCH_CHECK(d_v == 512, "Invalid d_v: ", d_v);
    TORCH_CHECK(q_start_index_s >= 0, "q_start_index_s must be >= 0");
    TORCH_CHECK(h_kv == 1, "Sparse attention backward currently only supports h_kv=1. Got h_kv=", h_kv);
    TORCH_CHECK(topk > 0 && topk % 64 == 0, "Sparse attention backward requires topk to be a positive multiple of 64. Got topk=", topk);

    KU_CHECK_DEVICE(q);
    KU_CHECK_DEVICE(dO);
    KU_CHECK_DEVICE(indices);
    KU_CHECK_DEVICE(s);
    KU_CHECK_DEVICE(ds);
    KU_CHECK_DEVICE(topk_length);

    KU_CHECK_DTYPE(q, torch::kBFloat16);
    KU_CHECK_DTYPE(dO, torch::kBFloat16);
    KU_CHECK_DTYPE(indices, torch::kInt32);
    KU_CHECK_DTYPE(s, torch::kBFloat16);
    KU_CHECK_DTYPE(ds, torch::kBFloat16);
    KU_CHECK_DTYPE(topk_length, torch::kInt32);

    KU_CHECK_SHAPE(q, s_q, h_q, d_qk);
    KU_CHECK_SHAPE(dO, s_q, h_q, d_v);
    KU_CHECK_SHAPE(indices, s_q, h_kv, topk);
    KU_CHECK_SHAPE(s, s_q, h_q, topk);
    KU_CHECK_SHAPE(ds, s_q, h_q, topk);
    KU_CHECK_SHAPE(topk_length, s_q);

    KU_CHECK_LAST_DIM_CONTIGUOUS(q);
    KU_CHECK_LAST_DIM_CONTIGUOUS(dO);
    KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
    KU_CHECK_LAST_DIM_CONTIGUOUS(s);
    KU_CHECK_LAST_DIM_CONTIGUOUS(ds);
    KU_CHECK_LAST_DIM_CONTIGUOUS(topk_length);

    auto opts = q.options();
    at::Tensor dKV = torch::zeros({s_kv, h_kv, d_qk}, opts.dtype(torch::kFloat32));

    KU_CHECK_CONTIGUOUS(dKV);

    SparseAttnBwdParams params = {
        s_q, s_kv, h_q, h_kv, d_qk, d_v, topk,
        q_start_index_s,
        0.0f, 0.0f,

        (bf16*)q.data_ptr(),
        nullptr,
        nullptr,
        (bf16*)dO.data_ptr(),
        (int*)indices.data_ptr(),
        nullptr,
        ku::get_optional_tensor_ptr<int>(topk_length),

        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)),
        0, 0,
        0, 0,
        int64_stride_to_int(dO.stride(0)), int64_stride_to_int(dO.stride(1)),
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),

        nullptr,
        (float*)dKV.data_ptr(),
        nullptr,
        0, 0,
        int64_stride_to_int(dKV.stride(0)), int64_stride_to_int(dKV.stride(1)),
        0, 0,
        (bf16*)s.data_ptr(),
        (bf16*)ds.data_ptr(),
        int64_stride_to_int(s.stride(0)), int64_stride_to_int(s.stride(1)),
        int64_stride_to_int(ds.stride(0)), int64_stride_to_int(ds.stride(1)),

        num_sm,
        stream
    };

    return {
        h_q,
        d_qk,
        have_topk_length,
        dKV,
        params
    };
}

static std::vector<BwdFeatures> build_sparse_bwd_head128_2kernels_dkv_required_features(
    int h_q,
    int d_qk,
    bool have_topk_length
) {
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
    return required_features;
}

static at::Tensor sparse_attn_bwd_head128_2kernels_dkv_interface(
    const at::Tensor &q,
    const at::Tensor &dO,
    const at::Tensor &indices,
    const at::Tensor &s,
    const at::Tensor &ds,
    int s_kv,
    int d_v,
    const std::optional<at::Tensor> &topk_length,
    int q_start_index_s
) {
    Arch arch = Arch();
    bool is_sm100f = arch.is_sm100f();
    TORCH_CHECK(is_sm100f, "Sparse Attention Backward Kernel is only supported on SM100f architecture.");

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto ctx = prepare_sparse_attn_bwd_head128_2kernels_dkv_context(
        q, dO, indices, s, ds,
        s_kv, d_v, topk_length, q_start_index_s,
        arch.num_sms,
        at::cuda::getCurrentCUDAStream().stream()
    );
    auto required_features = build_sparse_bwd_head128_2kernels_dkv_required_features(
        ctx.h_q, ctx.d_qk, ctx.have_topk_length);

    if (ctx.h_q == 128) {
        Bwd_Sm100_Head128_2KernelsDKVImpl bwd_impl;
        bwd_impl.run(ctx.params, required_features);
    } else {
        TORCH_CHECK(false, "Sparse attention backward currently only supports h_q=128. Got h_q=", ctx.h_q);
    }

    return ctx.dKV.to(torch::kBFloat16);
}
