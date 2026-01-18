#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace sm100::bwd::head64 {

// Helper functions for backward pass

// Get maximum value from int4
CUTE_DEVICE int int4_max(int4 v) {
    return max(max(v.x, v.y), max(v.z, v.w));
}

// Get minimum value from int4
CUTE_DEVICE int int4_min(int4 v) {
    return min(min(v.x, v.y), min(v.z, v.w));
}

// Float2 multiply
CUTE_DEVICE float2 float2_mul(float2 a, float2 b) {
    return make_float2(a.x * b.x, a.y * b.y);
}

// Float2 fma
CUTE_DEVICE float2 float2_fma(float2 a, float2 b, float2 c) {
    return make_float2(fmaf(a.x, b.x, c.x), fmaf(a.y, b.y, c.y));
}

// Get maximum from array
template<int N>
CUTE_DEVICE float get_max(const float* arr) {
    float max_val = arr[0];
    CUTE_UNROLL
    for (int i = 1; i < N; ++i) {
        max_val = max(max_val, arr[i]);
    }
    return max_val;
}

// Compute softmax output S from logits P
template<int N>
CUTE_DEVICE float get_s_from_p(nv_bfloat162* s, const float* p, float scale_log2, float max_val) {
    float sum = 0.0f;
    CUTE_UNROLL
    for (int i = 0; i < N/2; ++i) {
        float2 p2 = make_float2(p[i*2], p[i*2+1]);
        float2 s2;
        s2.x = exp2f(p2.x * scale_log2 - max_val);
        s2.y = exp2f(p2.y * scale_log2 - max_val);
        sum += s2.x + s2.y;
        s[i] = __float22bfloat162_rn(s2);
    }
    return sum;
}

// Rescale output O in TMEM
template<int D, int STRIDE, int TMEM_COL>
CUTE_DEVICE void rescale_O(float scale) {
    float2 scale2 = make_float2(scale, scale);
    CUTE_UNROLL
    for (int i = 0; i < D/2; ++i) {
        float2 o;
        ku::tmem_ld_32dp32bNx<1>(TMEM_COL + i*STRIDE, &o);
        cutlass::arch::fence_view_async_tmem_load();
        o = float2_mul(o, scale2);
        ku::tmem_st_32dp32bNx<1>(TMEM_COL + i*STRIDE, &o);
    }
}

// Retrieve mask and reduce P values
template<int N, int TMEM_COL, int SYNC_BAR1, int SYNC_BAR2, bool EXCHANGE>
CUTE_DEVICE void retrieve_mask_and_reduce_p(
    const char* is_k_valid,
    int warp_idx, int lane_idx,
    auto&& on_p_free,
    float (*p_exchange_buf)[32*(N/2)],
    float* p_out
) {
    // Load P from TMEM
    float2 p[N/2];
    ku::tmem_ld_32dp32bNx<N>(TMEM_COL, p);
    cutlass::arch::fence_view_async_tmem_load();
    
    // Signal P free
    on_p_free();
    
    // Copy to output
    CUTE_UNROLL
    for (int i = 0; i < N/2; ++i) {
        p_out[i*2] = p[i].x;
        p_out[i*2+1] = p[i].y;
    }
}

} // namespace sm100::bwd::head64
