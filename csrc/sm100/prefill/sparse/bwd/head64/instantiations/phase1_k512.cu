#include "../phase1.cuh"

namespace sm100::bwd::head64 {

template void run_bwd_phase1_kernel<512>(const SparseAttnBwdParams& params);

} // namespace sm100::bwd::head64
