#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace sm100::bwd {

/**
 * Preprocess kernel to compute Delta = rowsum(O * dO)
 * 
 * Grid: (H, ceil(S/BLOCK_S), B)
 * Block: (BLOCK_S, BLOCK_D/WARP_SIZE)
 */
template<int BLOCK_S = 32, int BLOCK_D = 32>
__global__ void preprocess_delta_kernel(
    const __nv_bfloat16* __restrict__ O,      // [B, S, H, D]
    const __nv_bfloat16* __restrict__ dO,     // [B, S, H, D]
    float* __restrict__ Delta,                 // [B, S, H]
    int B, int S, int H, int D
) {
    const int h = blockIdx.x;
    const int s_block = blockIdx.y;
    const int b = blockIdx.z;
    
    const int s = s_block * BLOCK_S + threadIdx.x;
    const int d_start = threadIdx.y * BLOCK_D;
    
    if (s >= S) return;
    
    // Compute base offset
    const int base_idx = ((b * S + s) * H + h) * D;
    
    // Accumulate O * dO for this row
    float acc = 0.0f;
    
    #pragma unroll
    for (int d = d_start; d < min(d_start + BLOCK_D, D); ++d) {
        float o_val = __bfloat162float(O[base_idx + d]);
        float do_val = __bfloat162float(dO[base_idx + d]);
        acc += o_val * do_val;
    }
    
    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    
    // Block-level reduction across warps
    __shared__ float shared_acc[32];  // Max 32 warps
    
    if (threadIdx.y == 0) {
        shared_acc[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    
    if (threadIdx.x % 32 == 0) {
        atomicAdd(&shared_acc[threadIdx.x / 32], acc);
    }
    __syncthreads();
    
    // Final reduction and write
    if (threadIdx.y == 0) {
        float final_acc = shared_acc[threadIdx.x];
        
        // Reduce across all threads in first warp
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            final_acc += __shfl_xor_sync(0xffffffff, final_acc, offset);
        }
        
        if (threadIdx.x == 0) {
            Delta[(b * S + s_block * BLOCK_S) * H + h] = final_acc;
        }
    }
}

/**
 * Optimized preprocess kernel using vectorized loads
 */
template<int BLOCK_S = 32>
__global__ void preprocess_delta_kernel_v2(
    const __nv_bfloat16* __restrict__ O,      // [B, S, H, D]
    const __nv_bfloat16* __restrict__ dO,     // [B, S, H, D]
    float* __restrict__ Delta,                 // [B, S, H]
    int B, int S, int H, int D,
    int stride_s, int stride_h
) {
    const int h = blockIdx.x;
    const int s = blockIdx.y * BLOCK_S + threadIdx.y;
    const int b = blockIdx.z;
    const int lane = threadIdx.x;
    
    if (s >= S) return;
    
    // Base offset for this (b, s, h) position
    const int base_idx = b * stride_s * S + s * stride_s + h * stride_h;
    
    // Each thread accumulates D/32 elements
    float acc = 0.0f;
    const int elements_per_thread = (D + 31) / 32;
    
    #pragma unroll 4
    for (int i = 0; i < elements_per_thread; ++i) {
        int d = lane + i * 32;
        if (d < D) {
            float o_val = __bfloat162float(O[base_idx + d]);
            float do_val = __bfloat162float(dO[base_idx + d]);
            acc += o_val * do_val;
        }
    }
    
    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    
    // Write result
    if (lane == 0) {
        Delta[b * S * H + s * H + h] = acc;
    }
}

/**
 * Launch preprocess kernel
 */
inline void launch_preprocess_delta(
    const __nv_bfloat16* O,
    const __nv_bfloat16* dO,
    float* Delta,
    int B, int S, int H, int D,
    cudaStream_t stream
) {
    constexpr int BLOCK_S = 32;
    
    dim3 grid(H, (S + BLOCK_S - 1) / BLOCK_S, B);
    dim3 block(32, BLOCK_S);  // 32 threads per row, BLOCK_S rows
    
    preprocess_delta_kernel_v2<BLOCK_S><<<grid, block, 0, stream>>>(
        O, dO, Delta, B, S, H, D, H * D, D
    );
}

} // namespace sm100::bwd
