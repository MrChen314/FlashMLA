#include "../phase1.h"
#include "../phase1.cuh"

namespace sm100::bwd::head128 {

template void run_bwd_phase1_kernel<512>(const SparseAttnBwdParams& params);

}
