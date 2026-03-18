#pragma once

#include "sparse_bwd.h"

#include "sm100/prefill/sparse/bwd/head128_2kernels/phase.h"

class Bwd_Sm100_Head128_2KernelsDQImpl : public BwdImplBase {
    DECLARE_SUPPORTED_FEATURES(
        BwdFeatures::HEAD_128,
        BwdFeatures::HEAD_DIM_576,
        BwdFeatures::TOPK_LENGTH
    )

protected:
    void run_(const SparseAttnBwdParams &params, const std::vector<FeatureT> &required_features) override {
        sm100::bwd::head128_2kernels::dq::run_bwd_dq_phase_kernel<576>(params);
    }
};

struct SparseBwdHead128_2KernelsDQLaunchContext {
    int h_q;
    int d_qk;
    bool have_topk_length;
    at::Tensor dQ;
    at::Tensor s;
    at::Tensor ds;
    at::Tensor delta;
    SparseAttnBwdParams params;
};

static SparseBwdHead128_2KernelsDQLaunchContext prepare_sparse_attn_bwd_head128_2kernels_dq_context(
    const at::Tensor &q,
    const at::Tensor &kv,
    const at::Tensor &o,
    const at::Tensor &dO,
    const at::Tensor &indices,
    const at::Tensor &lse,
    float sm_scale,
    int d_v,
    const std::optional<at::Tensor> &topk_length,
    int q_start_index_s,
    int num_sm,
    cudaStream_t stream
) {
    using bf16 = cutlass::bfloat16_t;

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

    auto opts = q.options();
    at::Tensor dQ = torch::empty({s_q, h_q, d_qk}, opts);
    at::Tensor s = torch::empty({s_q, h_q, topk}, opts);
    at::Tensor ds = torch::empty({s_q, h_q, topk}, opts);
    at::Tensor delta = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat32));

    KU_CHECK_CONTIGUOUS(dQ);
    KU_CHECK_CONTIGUOUS(s);
    KU_CHECK_CONTIGUOUS(ds);
    KU_CHECK_CONTIGUOUS(delta);

    SparseAttnBwdParams params = {
        s_q, s_kv, h_q, h_kv, d_qk, d_v, topk,
        q_start_index_s,
        sm_scale, sm_scale * LOG_2_E,

        (bf16*)q.data_ptr(),
        (bf16*)kv.data_ptr(),
        (bf16*)o.data_ptr(),
        (bf16*)dO.data_ptr(),
        (int*)indices.data_ptr(),
        (float*)lse.data_ptr(),
        ku::get_optional_tensor_ptr<int>(topk_length),

        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)),
        int64_stride_to_int(kv.stride(0)), int64_stride_to_int(kv.stride(1)),
        int64_stride_to_int(o.stride(0)), int64_stride_to_int(o.stride(1)),
        int64_stride_to_int(dO.stride(0)), int64_stride_to_int(dO.stride(1)),
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),

        (bf16*)dQ.data_ptr(),
        nullptr,
        (float*)delta.data_ptr(),
        int64_stride_to_int(dQ.stride(0)), int64_stride_to_int(dQ.stride(1)),
        0, 0,
        int64_stride_to_int(delta.stride(0)), int64_stride_to_int(delta.stride(1)),
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
        dQ,
        s,
        ds,
        delta,
        params
    };
}

static std::vector<BwdFeatures> build_sparse_bwd_head128_2kernels_required_features(
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

static std::vector<at::Tensor> sparse_attn_bwd_head128_2kernels_dq_interface(
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
    Arch arch = Arch();
    bool is_sm100f = arch.is_sm100f();
    TORCH_CHECK(is_sm100f, "Sparse Attention Backward Kernel is only supported on SM100f architecture.");

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto ctx = prepare_sparse_attn_bwd_head128_2kernels_dq_context(
        q, kv, o, dO, indices, lse,
        sm_scale, d_v, topk_length, q_start_index_s,
        arch.num_sms,
        at::cuda::getCurrentCUDAStream().stream()
    );
    auto required_features = build_sparse_bwd_head128_2kernels_required_features(
        ctx.h_q, ctx.d_qk, ctx.have_topk_length);

    if (ctx.h_q == 128) {
        Bwd_Sm100_Head128_2KernelsDQImpl bwd_impl;
        bwd_impl.run(ctx.params, required_features);
    } else {
        TORCH_CHECK(false, "Sparse attention backward currently only supports h_q=128. Got h_q=", ctx.h_q);
    }

    return {ctx.dQ, ctx.s, ctx.ds};
}
