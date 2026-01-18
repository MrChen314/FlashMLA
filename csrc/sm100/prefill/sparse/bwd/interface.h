#pragma once

#include <cuda_runtime.h>
#include <torch/torch.h>

namespace sm100::bwd {

/**
 * Preprocess kernel: compute Delta = rowsum(O * dO)
 * 
 * This kernel computes the softmax gradient correction term Delta.
 * Delta[b, s, h] = sum_d(O[b, s, h, d] * dO[b, s, h, d])
 * 
 * @param out Forward output tensor [B, S, H, D_V]
 * @param d_out Output gradient tensor [B, S, H, D_V]
 * @param stream CUDA stream
 * @return Delta tensor [B, S, H] in float32
 */
at::Tensor preprocess_delta(
    at::Tensor out,         // [B, S, H, D_V]
    at::Tensor d_out,       // [B, S, H, D_V]
    cudaStream_t stream = nullptr
);

/**
 * Main backward kernel for sparse attention
 * 
 * Computes gradients dQ, dK, dV for sparse MLA attention.
 * 
 * @param q Query tensor [B, S, H, D_QK]
 * @param kv KV tensor [B, S_KV, H_KV, D_QK]
 * @param out Forward output [B, S, H, D_V]
 * @param d_out Output gradient [B, S, H, D_V]
 * @param indices Sparse indices [B, S, H_KV, topk]
 * @param lse Log-sum-exp from forward [B, S, H]
 * @param delta Precomputed Delta [B, S, H]
 * @param d_q Output: Query gradient [B, S, H, D_QK]
 * @param d_kv Output: KV gradient [B, S_KV, H_KV, D_QK] (float32 for atomic add)
 * @param sm_scale Softmax scale factor
 * @param stream CUDA stream
 */
void sparse_attn_bwd(
    at::Tensor q,           // [B, S, H, D_QK]
    at::Tensor kv,          // [B, S_KV, H_KV, D_QK]
    at::Tensor out,         // [B, S, H, D_V]
    at::Tensor d_out,       // [B, S, H, D_V]
    at::Tensor indices,     // [B, S, H_KV, topk]
    at::Tensor lse,         // [B, S, H]
    at::Tensor delta,       // [B, S, H]
    at::Tensor d_q,         // [B, S, H, D_QK] - output
    at::Tensor d_kv,        // [B, S_KV, H_KV, D_QK] - output (float32)
    float sm_scale,
    cudaStream_t stream = nullptr
);

/**
 * Postprocess kernel: convert d_kv from float32 to bf16
 * 
 * @param d_kv_fp32 KV gradient in float32 [B, S_KV, H_KV, D_QK]
 * @param stream CUDA stream
 * @return KV gradient in bf16 [B, S_KV, H_KV, D_QK]
 */
at::Tensor postprocess_dkv(
    at::Tensor d_kv_fp32,   // [B, S_KV, H_KV, D_QK] float32
    cudaStream_t stream = nullptr
);

/**
 * Combined backward pass for sparse MLA
 * 
 * This function combines preprocess, main backward, and postprocess.
 * 
 * @param q Query tensor [B, S, H, D_QK]
 * @param kv KV tensor [B, S_KV, H_KV, D_QK]
 * @param out Forward output [B, S, H, D_V]
 * @param d_out Output gradient [B, S, H, D_V]
 * @param indices Sparse indices [B, S, H_KV, topk]
 * @param lse Log-sum-exp from forward [B, S, H]
 * @param sm_scale Softmax scale factor (default: 1/sqrt(D_QK))
 * @return Tuple of (d_q, d_kv) gradients
 */
std::tuple<at::Tensor, at::Tensor> sparse_mla_bwd(
    at::Tensor q,
    at::Tensor kv,
    at::Tensor out,
    at::Tensor d_out,
    at::Tensor indices,
    at::Tensor lse,
    float sm_scale = -1.0f  // -1 means auto-compute
);

} // namespace sm100::bwd
