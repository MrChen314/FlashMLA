#pragma once

#include "params.h"

namespace sm100::bwd::head64 {

// Run the backward kernel for sparse attention
// D_QK: Query/Key dimension (576 or 512)
template<int D_QK>
void run_bwd_phase1_kernel(const SparseAttnBwdParams& params);

} // namespace sm100::bwd::head64
