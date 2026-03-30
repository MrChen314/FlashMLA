# dKV 256-Block 四阶段流水修订版

## Summary
- 把 [`dkv_phase.cuh`](/Users/chenql/Desktop/workspace/operator/FlashMLA/csrc/sm100/prefill/sparse/bwd/head128_2kernels/dkv_phase.cuh) 从旧的 128-row `k_pair` 流水改成 256-row `k_block` 流水，保留你已在 [`dkv_config.h`](/Users/chenql/Desktop/workspace/operator/FlashMLA/csrc/sm100/prefill/sparse/bwd/head128_2kernels/dkv_config.h) 改好的 `Part0 / Part1_2 / RoPE` MMA 和 TMEM 列布局。
- 每轮 `k_block` 覆盖 256 个 topk 行，两个 CTA 各负责 128 行。
- 四段 MMA 顺序固定为：
  1. `part0` 计算 `[0,256)`
  2. 若 `k_block > 0`，等上一轮 `part2` 全局累加完成后计算 `RoPE` `[512,576)`
  3. `part1` 计算 `[256,384)`
  4. 等当前轮 `RoPE` 全局累加完成后计算 `part2` `[384,512)`

## Key Changes
- 常量与循环语义改成：
  - `TOPK_GRANULARITY = DKV_TILE_M = 256`
  - `DKV_ROWS_PER_CTA = 128`
  - `num_k_blocks = max(params.topk / DKV_TILE_M, 1)`
  - `row_global = k_block * 256 + cta_idx * 128 + row`
- `S/dS` 的 TMA shape/stride 改成 256-block 语义，第四维步长从 `128` 改成 `256`，循环变量统一从 `k_pair` 改成 `k_block`。
- `SharedMemoryPlan` 里的 dKV 屏障重构为：
  - `bar_dkv_part0_ready[NUM_S_DS_BUFS]`
  - `bar_dkv_rope_ready[NUM_S_DS_BUFS]`
  - `bar_dkv_part1_ready[NUM_S_DS_BUFS]`
  - `bar_dkv_part2_ready[NUM_S_DS_BUFS]`
  - `bar_dkv_part0_done`
  - `bar_dkv_rope_done`
  - `bar_dkv_part1_done`
  - `bar_dkv_part2_done`
- `warp_idx == 9` 的 `S` TMA 和 `warp_idx == 10` 的 `dS` TMA 都等上一轮同 buffer 的 `bar_dkv_part2_ready[buf]`。
  - 原因：`s[buf]` 和 `ds[buf]` 的最后一次消费都发生在 `part2`，所以必须等到 `part2` 已经发起完成，才能安全覆写该 ping-pong buffer。
- MMA issuer 仍固定为 `cta_idx == 0 && warp_idx == 8 && elect_one_sync()`，每轮顺序明确为：
  1. 等上一轮 `bar_dkv_part0_done`，复用 `tmem_cols::dKV_part0`
  2. `part0`: `S @ dO[0:256)` 清零 + `dS @ Q_nope[0:256)` 累加，随后 multicast `bar_dkv_part0_ready[buf]`
  3. 若 `k_block > 0`，等上一轮 `bar_dkv_part2_done`，复用与 `part2` 复叠的 `tmem_cols::dKV_RoPE`
  4. `RoPE`: `dS @ Q_rope` 清零，随后 multicast `bar_dkv_rope_ready[buf]`
  5. 等上一轮 `bar_dkv_part1_done`，复用 `tmem_cols::dKV_part1`
  6. `part1`: `S @ dO[256:384)` 清零 + `dS @ Q_nope[256:384)` 累加，随后 multicast `bar_dkv_part1_ready[buf]`
  7. 等当前轮 `bar_dkv_rope_done`
  8. `part2`: `S @ dO[384:512)` 清零 + `dS @ Q_nope[384:512)` 累加，随后 multicast `bar_dkv_part2_ready[buf]`
- TMEM fragment 显式拆成：
  - `tdKV_part0`
  - `tdKV_part1`
  - `tdKV_part2`
  - `tdKV_RoPE`
  它们分别绑定到 `tmem_cols::dKV_part0 / part1 / part2 / dKV_RoPE`。
- `QNoPE` 和 `dO` 的 MMA 视图不再走旧的单 `tdKV` 路径，而是显式切成 `[0:256)`、`[256:384)`、`[384:512)` 三段子视图给 `part0/1/2` 使用。

## Transfer Plan
- transfer 侧改成“WG0 和 WG1 都参与 `part0 / part1 / part2 / RoPE` 四段写回”，不再保留旧的 “WG0 只写一半、WG1 写另一半并顺带 RoPE” 分工。
- `DKV_ROWS_PER_CTA = 128` 后，线程映射固定为：
  - `row = local_warp_idx * 32 + lane_idx`
  - 不再使用旧的 `half` 维度
  - `WG0` 和 `WG1` 都覆盖同一批 `0..127` 行
- 两个 WG 通过分摊列块并行 drain：
  - `part0` 对 `[0,256)` 分 2 组列块并行写回
  - `part1` 对 `[256,384)` 分 2 组列块并行写回
  - `part2` 对 `[384,512)` 分 2 组列块并行写回
  - `RoPE` 对 `[512,576)` 分 2 组列块并行写回
- `*_done` barrier 的计数按“每个 CTA 内所有参与该 part drain 的 transfer 线程数”初始化，因为 WG0/WG1 在四个 part 中都会 arrive。

## Test Plan
- 编译验证：`MAX_JOBS=192 python setup.py build_ext --inplace`
- GPU 数值验证至少覆盖：
  - `topk = 256`，验证单轮四阶段顺序
  - `topk = 512`，验证跨轮 `part0/part1/part2` 与 `RoPE/part2` 复用依赖
  - 重复 `kv_idx`，验证全局 `atomic_add`
  - `topk_length < topk`、非法 index、`index > max_kv_i`，验证写回保护
- 如果本机无 GPU，只做静态检查，并把上述命令留到远端机执行。

## Assumptions
- [`dkv_config.h`](/Users/chenql/Desktop/workspace/operator/FlashMLA/csrc/sm100/prefill/sparse/bwd/head128_2kernels/dkv_config.h) 里 `dKV_RoPE` 与 `dKV_part2` 在 TMEM 列 `384` 复叠的设计保持不变。
- 这次只改 dKV 双-kernel 路径，不同步修改 dQ kernel、分析文档或其他说明文件。
- `CTA0` 继续是唯一 2CTA UMMA issuer，`CTA1` 只参与 TMA、barrier、transfer 和 `cluster_sync()`。
