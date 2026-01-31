#pragma once

#include "params.h"

namespace sm100::bwd::head128 {

// 反向传播核函数声明
template<int D_QK>
void run_bwd_preprocess_delta_kernel(const SparseAttnBwdParams& params);

template<int D_QK>
void run_bwd_phase1_kernel(const SparseAttnBwdParams& params);

}  // namespace sm100::bwd::head128
