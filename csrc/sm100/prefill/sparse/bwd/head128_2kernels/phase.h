#pragma once

#include "params.h"

namespace sm100::bwd::head128_2kernels::dq {

template<int D_QK>
void run_bwd_dq_phase_kernel(const SparseAttnBwdParams& params);

}  // namespace sm100::bwd::head128_2kernels::dq
