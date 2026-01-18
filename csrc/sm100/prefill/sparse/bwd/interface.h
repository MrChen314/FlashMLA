#pragma once

#include <cuda_runtime.h>
#include <torch/torch.h>

namespace sm100::bwd {

// Interface for sparse attention backward pass
void sparse_attn_bwd(
    at::Tensor q,           // [s_q, h_q, d_qk]
    at::Tensor kv,          // [s_kv, h_kv, d_qk]
    at::Tensor out,         // [s_q, h_q, d_v]
    at::Tensor d_out,       // [s_q, h_q, d_v]
    at::Tensor indices,     // [s_q, topk]
    at::Tensor lse,         // [s_q, h_q]
    at::Tensor d_q,         // [s_q, h_q, d_qk] - output
    at::Tensor d_kv,        // [s_kv, h_kv, d_qk] - output (float32)
    float sm_scale,
    cudaStream_t stream
);

// Preprocess: compute Delta = rowsum(O * dO)
at::Tensor preprocess_delta(
    at::Tensor out,         // [s_q, h_q, d_v]
    at::Tensor d_out,       // [s_q, h_q, d_v]
    cudaStream_t stream
);

// Postprocess: convert d_kv from float32 to bf16
at::Tensor postprocess_dkv(
    at::Tensor d_kv_fp32,   // [s_kv, h_kv, d_qk] float32
    cudaStream_t stream
);

} // namespace sm100::bwd
