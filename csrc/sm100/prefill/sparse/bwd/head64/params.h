#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace sm100::bwd::head64 {

struct SparseAttnBwdParams {
    // Input tensors
    void* q;                    // Query tensor [s_q, h_q, d_qk]
    void* kv;                   // KV tensor [s_kv, h_kv, d_qk]
    void* out;                  // Forward output [s_q, h_q, d_v]
    void* d_out;                // Output gradient [s_q, h_q, d_v]
    int* indices;               // Sparse indices [s_q, topk]
    float* lse;                 // Log-sum-exp from forward [s_q, h_q]
    
    // Output tensors
    void* d_q;                  // Query gradient [s_q, h_q, d_qk]
    void* d_kv;                 // KV gradient [s_kv, h_kv, d_qk] (float32 for atomic add)
    
    // Dimensions
    int s_q;                    // Query sequence length
    int s_kv;                   // KV sequence length
    int h_q;                    // Number of query heads
    int h_kv;                   // Number of KV heads (typically 1 for MLA)
    int d_qk;                   // Query/Key dimension (576)
    int d_v;                    // Value dimension (512)
    int topk;                   // Number of top-k indices
    
    // Strides
    int stride_q_s_q;           // Stride for q along s_q dimension
    int stride_q_h_q;           // Stride for q along h_q dimension
    int stride_kv_s_kv;         // Stride for kv along s_kv dimension
    int stride_kv_h_kv;         // Stride for kv along h_kv dimension
    int stride_out_s_q;         // Stride for out along s_q dimension
    int stride_out_h_q;         // Stride for out along h_q dimension
    int stride_dout_s_q;        // Stride for d_out along s_q dimension
    int stride_dout_h_q;        // Stride for d_out along h_q dimension
    int stride_dq_s_q;          // Stride for d_q along s_q dimension
    int stride_dq_h_q;          // Stride for d_q along h_q dimension
    int stride_dkv_s_kv;        // Stride for d_kv along s_kv dimension
    int stride_dkv_h_kv;        // Stride for d_kv along h_kv dimension
    int stride_indices_s_q;     // Stride for indices along s_q dimension
    
    // Scaling factors
    float sm_scale;             // Softmax scale = 1/sqrt(d_qk)
    float sm_scale_div_log2;    // sm_scale / log(2) for exp2 optimization
    
    // Optional
    int* topk_length;           // Per-query topk length (nullptr if fixed)
    
    // CUDA stream
    cudaStream_t stream;
};

} // namespace sm100::bwd::head64
