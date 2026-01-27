#include "test_dual_gemm_vs_torch.cuh"

#include <cstdint>
#include <iostream>
#include <string>

namespace {

// 打印使用说明
void print_usage(const char* argv0) {
    std::cout << "用法: " << argv0 << " [seed] [abs_tol] [rel_tol]\n";
    std::cout << "  seed     : 可选，随机种子 (默认 2026)\n";
    std::cout << "  abs_tol  : 可选，绝对误差阈值 (默认 5e-2)\n";
    std::cout << "  rel_tol  : 可选，相对误差阈值 (默认 5e-2)\n";
}

}  // namespace

int main(int argc, char** argv) {
    uint64_t seed = 2026;
    float abs_tol = 5e-2f;
    float rel_tol = 5e-2f;

    if (argc > 1) {
        std::string arg1 = argv[1];
        if (arg1 == "-h" || arg1 == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        try {
            seed = static_cast<uint64_t>(std::stoull(arg1));
        } catch (...) {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (argc > 2) {
        try {
            abs_tol = std::stof(argv[2]);
        } catch (...) {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (argc > 3) {
        try {
            rel_tol = std::stof(argv[3]);
        } catch (...) {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (argc > 4) {
        print_usage(argv[0]);
        return 1;
    }

    bool ok = dual_gemm_test::run_dual_gemm_vs_matmul(seed, abs_tol, rel_tol, true);
    return ok ? 0 : 1;
}
