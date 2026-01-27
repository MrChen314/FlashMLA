# Makefile for Dual GEMM 精度测试
#
# 用法:
#   make              # 编译测试程序
#   make run          # 编译并运行测试
#   make clean        # 清理编译产物
#
# 环境变量:
#   CUTLASS_PATH    - CUTLASS 库路径 (默认: $(HOME)/cutlass)
#   CUDA_PATH       - CUDA 安装路径 (默认: /usr/local/cuda)

# 编译器设置
NVCC := nvcc
CXX := g++

# CUDA 架构 (SM100 = Blackwell)
CUDA_ARCH := sm_100a

# 编译标志
NVCC_FLAGS := -std=c++20 -O3 -gencode arch=compute_100a,code=$(CUDA_ARCH)
NVCC_FLAGS += -Xcompiler "-Wall -Wextra"
NVCC_FLAGS += -lineinfo  # 启用行号信息，方便调试

# 路径设置
CUTLASS_PATH ?= $(HOME)/cutlass
CUDA_PATH ?= /usr/local/cuda
KERUTILS_PATH := ../../../../kerutils/include

# 包含路径
INCLUDES := -I$(CUTLASS_PATH)/include
INCLUDES += -I$(CUTLASS_PATH)/tools/util/include
INCLUDES += -I$(KERUTILS_PATH)
INCLUDES += -I../../../../
INCLUDES += -I.

# 库路径
LDFLAGS := -L$(CUDA_PATH)/lib64
LIBS := -lcudart

# 目标
TARGET := test_dual_gemm

# 源文件
SRCS := test_dual_gemm_vs_torch_main.cu
HEADERS := test_dual_gemm_vs_torch.cuh

# 默认目标
all: $(TARGET)

# 编译规则
$(TARGET): $(SRCS) $(HEADERS)
	@echo "============================================"
	@echo "编译 Dual GEMM 测试程序..."
	@echo "============================================"
	@echo "CUTLASS_PATH: $(CUTLASS_PATH)"
	@echo "KERUTILS_PATH: $(KERUTILS_PATH)"
	@echo "CUDA_ARCH: $(CUDA_ARCH)"
	@echo ""
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $(SRCS) $(LDFLAGS) $(LIBS)
	@echo ""
	@echo "编译完成: $(TARGET)"

# 运行测试
run: $(TARGET)
	@echo ""
	@echo "============================================"
	@echo "运行 Dual GEMM 精度测试..."
	@echo "============================================"
	./$(TARGET)

# 使用不同随机种子运行多次测试
test-multiple: $(TARGET)
	@echo "运行多个随机种子的测试..."
	@for seed in 42 123 456 789 1024; do \
		echo "\n=== 种子: $$seed ===" ; \
		./$(TARGET) $$seed || exit 1; \
	done
	@echo "\n所有测试通过!"

# 清理
clean:
	rm -f $(TARGET)

# 帮助信息
help:
	@echo "Dual GEMM 精度测试 Makefile"
	@echo ""
	@echo "目标:"
	@echo "  all           - 编译测试程序 (默认)"
	@echo "  run           - 编译并运行测试"
	@echo "  test-multiple - 使用多个随机种子运行测试"
	@echo "  clean         - 清理编译产物"
	@echo "  help          - 显示此帮助信息"
	@echo ""
	@echo "环境变量:"
	@echo "  CUTLASS_PATH  - CUTLASS 库路径 (当前: $(CUTLASS_PATH))"
	@echo "  CUDA_PATH     - CUDA 安装路径 (当前: $(CUDA_PATH))"
	@echo ""
	@echo "示例:"
	@echo "  make                                    # 编译"
	@echo "  make run                                # 编译并运行"
	@echo "  CUTLASS_PATH=/opt/cutlass make run     # 指定 CUTLASS 路径"

.PHONY: all run test-multiple clean help
