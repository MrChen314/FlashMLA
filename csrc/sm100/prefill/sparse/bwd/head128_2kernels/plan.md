# 完善 `head128_2kernels` 的 dKV 双 CTA 实现

**Summary**
- 把 [`dkv_config.h`](/Users/chenql/Desktop/workspace/operator/FlashMLA/csrc/sm100/prefill/sparse/bwd/head128_2kernels/dkv_config.h) 补成一个完整可编译的配置头，语义对齐现有 [`dq_config.h`](/Users/chenql/Desktop/workspace/operator/FlashMLA/csrc/sm100/prefill/sparse/bwd/head128_2kernels/dq_config.h)。
- 在 [`dkv_phase.cuh`](/Users/chenql/Desktop/workspace/operator/FlashMLA/csrc/sm100/prefill/sparse/bwd/head128_2kernels/dkv_phase.cuh) 实现 `dKV-only` kernel：`Q / dO / S / dS` 全部走 TMA；MMA 使用双 CTA cluster；`S / dS` 按“两个 64 block 组一个 128-row MMA tile”执行。
- `N` 方向不额外发明新规则，直接复用前向 [`fwd/head128/config.h`](/Users/chenql/Desktop/workspace/operator/FlashMLA/csrc/sm100/prefill/sparse/fwd/head128/config.h) 里 `TiledMMA_O` 的做法：NoPE 保持 `atom N=256`，但通过 `make_tiled_mma(..., Tile<...>)` 让两个 CTA 共同覆盖逻辑上的 `512` 列。
- 实现范围只考虑 `topk` 是 `128` 的整数倍，不处理 `64`、`192` 这类非整倍数规模。

**Key Changes**
- 在 `dkv_config.h` 中补齐完整头文件骨架：`#pragma once`、必要 include、`namespace sm100::bwd::head128_2kernels::dkv`、`using namespace cute;`、`TmaParams`、常量、layout、TMEM 列定义、`SharedMemoryPlan`、`SMEM_SIZE`。
- 明确把 dKV kernel 的 MMA 行维从“当前 scratch 单块 `64`”提升为“paired tile 的 `128`”。新增本地常量并固定语义：
  - `DKV_TILE_M = B_H = 128`
  - `DKV_ROWS_PER_CTA = DKV_TILE_M / 2 = 64`
  - `NUM_THREADS = 192`，warp 分工固定为 `4` 个 TMEM/atomic transfer warps、`1` 个 TMA/control warp、`1` 个 MMA warp
- `Q / dO` 的 staging 按列切给 CTA：
  - NoPE 每 CTA 落地 `[128, 256]`
  - RoPE 每 CTA 落地 `[128, 32]`
  - 这些 layout 和 `TiledMMA_O` 同口径，服务于逻辑 `N=512/64`
- `S / dS` 的 staging 按 paired-topk tile 设计：
  - 全局 scratch 仍是 dQ kernel 已写出的 `[s_q, h_q, topk]`
  - `topk` 视为若干个固定的 paired tile，每个 paired tile 覆盖 `128` 行
  - 第 `k_pair` 轮时 `local_k_block = 2 * k_pair + cta_idx`
  - CTA0 读取前一个 `64`-row block，CTA1 读取后一个 `64`-row block，两个 CTA 合起来组成一个逻辑 `128-row` A operand
- 为 MMA 使用单独的 CuTe 视图，不直接把 TMA landing layout 当 row-major 读：
  - `Q/dO` 提供 `*_MMA` 视图，按前向 `TiledMMA_O` 的 tile/permutation 映射
  - `S/dS` 提供 `*_MMA` 视图，保证 A operand 在 `M=128, K=128` 视角下是 layout-safe 的
- `TiledMMA_dKV` 改成“前向 `TiledMMA_O` 风格”的双 CTA 配置：
  - atom 仍用 `SM100_MMA_F16BF16_2x1SM_SS_NOELECT<..., 128, 256, UMMA::Major::MN, UMMA::Major::MN>`
  - `make_tiled_mma` 额外传入和 `TiledMMA_O` 同类的 `Tile<Int<128>, Layout<Shape<_128, _2, _2>, Stride<_1, _256, _128>>, _16>{}`
  - `partition_fragment_C` 的逻辑输出按 `[64, 512]` 建立 NoPE 累加片段
- `TiledMMA_dKV_RoPE` 保持双 CTA，但 `N=64` 可直接表达，不需要 NoPE 那个额外 tile trick；逻辑输出按 `[64, 64]`
- `tmem_cols` 固定为：
  - `dKV = 0`
  - `dKV_RoPE = 256`
  - `kNumUsedCols = 288`
  - TMEM 逻辑只使用 `288` 列，但 allocator/free 仍固定使用 `512` 列
- `SharedMemoryPlan` 只保留 `q_nope / q_rope / dO / s / ds` 和 7 个 barrier：
  - `bar_q_nope_ready`
  - `bar_q_rope_ready`
  - `bar_dO_ready`
  - `bar_s_ready`
  - `bar_ds_ready`
  - `bar_dkv_nope_ready`
  - `bar_dkv_rope_ready`
  - 不再保留 fused/dQ 路径里的 `kv_peer`、`is_k_valid`、`part0/part1 done` 之类状态
- `dkv_phase.cuh` 的 kernel 结构按下面固定：
  - warp0 elected lane 负责 barrier init、descriptor prefetch、TMEM allocate、TMA launch
  - 先做一次 `cluster_sync()`，保证 cluster barrier 初始化完成
  - Q_nope / Q_rope / dO 只在 prologue TMA 一次，每 CTA 选自己的列 slice
  - kernel 只支持 `topk % 128 == 0`，按 `num_k_pairs = topk / 128` 循环执行
  - 每轮两个 CTA 分别 TMA 取 `S` 和 `dS` 的一个 `64`-row half，不需要 odd-tail 清零分支
  - 每 CTA 等自己本地 TMA barrier 后执行 `cluster_sync()`，然后只允许 `cta_idx == 0` 的 MMA warp + elected lane 发起 2CTA UMMA
  - NoPE 路径先 `dV = S^T @ dO` 清零写入 `tdKV`，再 `dK_nope = dS^T @ Q_nope` 累加到同一 `tdKV`
  - RoPE 路径单独写 `tdKV_RoPE`
  - MMA 完成后使用 `umma_arrive_multicast_2x1SM_noelect(..., 1 | 2)` 同时通知两个 CTA 的 transfer warps
  - 4 个 transfer warps 读取本 CTA 对应的 `64` 行 TMEM 结果并 `atomic_add` 到 `params.dKV`
  - NoPE 一次性 drain 完整个逻辑 `[64, 512]`；RoPE 再 drain `[64, 64]`
  - 行号映射固定为 `row_global = (2 * k_pair + cta_idx) * 64 + row`
  - 写回前仍检查 `row_global < topk_length`、`kv_idx >= 0`、`kv_idx < params.s_kv`、`kv_idx <= max_kv_i`
  - 结果 drain 完成后做一次 `cluster_sync()` 作为 TMEM free 前的统一 fence
  - 循环结束后 warp0 free TMEM，释放大小固定为 `512` 列
- launch helper 固定沿用 dQ 的 cluster launch 模式：
  - 构建 `Q_nope / Q_rope / dO / S / dS` 的 TMA descriptors
  - `grid.x = 2 * params.s_q`
  - `clusterDim.x = 2`
  - `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE)`
  - `cudaLaunchKernelEx` 启动
- 外部接口不改：`run_bwd_dkv_phase_kernel<576>(const SparseAttnBwdParams&)` 保持不变。唯一接口变化是 dKV 内部 `TmaParams` 新增 `shape_S/tma_S` 与 `shape_dS/tma_dS`

**Test Plan**
- 先做编译验证：`MAX_JOBS=192 python setup.py build_ext --inplace`
- 再做 GPU 数值验证，覆盖 `topk` 为 `128` 的整数倍：
  - 基本 `topk = 128` case，验证单个 paired tile 的 dKV 数值正确
  - `topk = 256` 或更大整倍数 case，验证多 pair 循环的 dKV 数值正确
  - 含重复 KV index 的 case，验证 `atomic_add` 累加正确
  - 含 `-1` 或 `index > max_kv_i` 的 case，验证被正确跳过
  - 若仍保留 `topk_length` 语义，再补一个 `topk = 256` 且 `topk_length < topk` 的 case 验证写回保护
- 对比基准优先用 fused backward 的 `dKV` 结果，容差沿现有 CUDA 测试口径
- 如果本机无可用 GPU，只做静态编译检查，并把 GPU 验证留到远端机执行

**Assumptions**
- `dq_phase.cuh` 和 `dq_config.h` 本次不改；`S/dS` scratch 仍按当前 64-block 方式写出
- dKV kernel 当前只支持 `params.topk % 128 == 0`，不处理其他 `topk` 规模
- 双 CTA UMMA 只由 CTA0 的单一 elected 线程 issue；CTA1 不发指令，但必须参与 TMA、barrier wait、transfer 和 `cluster_sync()`
- `N` 方向的限制处理完全参考前向 `TiledMMA_O`，不再单独设计另一套 NoPE 切分
- TMEM allocator/free 口径固定保持 `512` 列，即使逻辑列布局只使用 `288` 列
- `csrc/utils.py` 和两份资源分析文档当前仍落后于这套 paired-tile 方案，本次实现默认不一起更新，除非后续单独要求
