# SM100 Head128 反向双kernel dKV 资源占用分析

基于当前 [`dkv_config.h`](/Users/chenql/Desktop/workspace/operator/FlashMLA/csrc/sm100/prefill/sparse/bwd/head128_2kernels/dkv_config.h) 的静态定义，本文重新计算 `head128_2kernels` 路径里 `dKV-only kernel` 在 **256-row topk block + 双 S/dS buffer** 配置下的 TMEM 和 SMEM 占用。

这份文档只回答当前这套配置到底占多少资源，不讨论实现是否正确，也不讨论要不要把 buffer 数改成 1。

当前关键配置是：

- `D_QK = 576`，其中 `D_V = 512`，`D_ROPE = 64`
- `B_H = 128`
- `TOPK_GRANULARITY = DKV_TILE_M = 256`
- `DKV_ROWS_PER_CTA = 128`
- `NOPE_COLS_PER_CTA = 256`
- `ROPE_COLS_PER_CTA = 32`
- `NUM_S_DS_BUFS = 2`

---

## 1. 当前布局概要

和旧版 128-row 文档相比，现在的 dKV 资源模型已经变成了另一套东西：

1. `topk_block` 从 `128` 提升到了 `256`
- 一个 `k_block` 覆盖 `256` 个 topk 行
- 两个 CTA 各负责 `128` 行

2. NoPE MMA 被拆成三段
- `TiledMMA_dKV_Part0`: `M=256, N=256`
- `TiledMMA_dKV_Part1_2`: `M=256, N=128`
- `TiledMMA_dKV_RoPE`: `M=256, N=64`

3. TMEM 列布局改成了四阶段版本
- `dKV_part0 = 0`
- `dKV_part1 = 256`
- `dKV_part2 = 384`
- `dKV_RoPE = 384`
- `kNumUsedCols = 512`

4. `S/dS` 仍然保留双 buffer
- `s[2]`
- `ds[2]`
- 这是当前 SMEM 暴涨的主要来源

---

## 2. TMEM 占用分析

当前 `tmem_cols` 定义如下：

```cpp
struct tmem_cols {
    static constexpr int dKV_part0 = 0;
    static constexpr int dKV_part1 = 256;
    static constexpr int dKV_part2 = 384;
    static constexpr int dKV_RoPE = 384;
    static constexpr int kNumUsedCols = 512;
};
```

这里最关键的一点是：

- `part2` 和 `RoPE` 在列 `384` 处复叠
- 峰值不是 `256 + 128 + 128 + 64 = 576 cols`
- 真正峰值是 `256 + 128 + max(128, 64) = 512 cols`

按当前逻辑片段大小可得到：

| 缓冲 | Logical Shape | 起始列 | 结束列 | 占用列数 | 占用大小 |
| :--- | :--- | :---: | :---: | ---: | ---: |
| `dKV_part0` | `[128, 256]` fp32 | 0 | 256 | 256 | 128.00 KiB |
| `dKV_part1` | `[128, 128]` fp32 | 256 | 384 | 128 | 64.00 KiB |
| `dKV_part2` | `[128, 128]` fp32 | 384 | 512 | 128 | 64.00 KiB |
| `dKV_RoPE` | `[128, 64]` fp32 | 384 | 448 | 64 | 32.00 KiB |
| **峰值总计** |  | 0 | 512 | **512** | **256.00 KiB** |

### 结论

- **TMEM 使用量**: `512 / 512 cols`
- **占用大小**: `256.00 KiB`
- **使用率**: `100%`
- `part2` 和 `RoPE` 的列复用只避免了超过 `512 cols`，但没有留下任何 TMEM 裕量

---

## 3. SMEM 占用分析

### 3.1 `SharedMemoryPlan` 的当前形状

当前 dKV kernel 的共享内存结构可以概括为：

```cpp
struct alignas(128) SharedMemoryPlan {
    q_nope   : [128, 256]
    q_rope   : [128,  32]
    dO       : [128, 256]
    s[2]     : 2 x [128, 128]
    ds[2]    : 2 x [128, 128]

    19 个 barrier
    tmem_start_addr
};
```

其中 `s[2]` / `ds[2]` 是和旧版文档差异最大的地方：

- 旧版是 `64-row` half tile
- 现在是 `128-row` per-CTA tile
- 又因为保留双 buffer，所以 `S` 和 `dS` 各自都要乘 `2`

### 3.2 数据区大小

按 `bf16 = 2 B` 计算：

| 组件 | Shape | 大小 (Bytes) | 大小 (KiB) |
| :--- | :--- | ---: | ---: |
| `q_nope` | `[128, 256]` | 65,536 | 64.00 |
| `q_rope` | `[128, 32]` | 8,192 | 8.00 |
| `dO` | `[128, 256]` | 65,536 | 64.00 |
| `s[2]` | `2 x [128, 128]` | 65,536 | 64.00 |
| `ds[2]` | `2 x [128, 128]` | 65,536 | 64.00 |
| **数据区合计** |  | **270,336** | **264.00** |

也就是说，仅数据区本身就已经达到 **264.00 KiB**。

### 3.3 barrier 与对齐开销

按当前文档一直使用的估算口径：

- `transac_bar_t` 按 `8 B / 个`
- `array_aligned<T, N>` 按 `16 B` 对齐
- 结构体整体按 `alignas(128)` 收尾

当前 barrier 数量是：

- `bar_q_nope_ready`
- `bar_q_rope_ready`
- `bar_dO_ready`
- `bar_s_ready[2]`
- `bar_ds_ready[2]`
- `bar_dkv_part0_ready[2]`
- `bar_dkv_rope_ready[2]`
- `bar_dkv_part1_ready[2]`
- `bar_dkv_part2_ready[2]`
- `bar_dkv_part0_done`
- `bar_dkv_rope_done`
- `bar_dkv_part1_done`
- `bar_dkv_part2_done`

合计：

- `19` 个 `transac_bar_t`
- `19 * 8 = 152 B`

再加上：

- `tmem_start_addr = 16 B`
- 中间对齐 padding = `8 B`
- 尾部 `128 B` 对齐 padding = `80 B`

因此同步与辅助区总开销为：

- `152 + 16 + 8 + 80 = 256 B`

### 3.4 总 SMEM 占用

| 组件 | 大小 (Bytes) | 大小 (KiB) |
| :--- | ---: | ---: |
| 数据区 | 270,336 | 264.00 |
| barriers + addr + padding | 256 | 0.25 |
| **总计** | **270,592** | **264.25** |

和 SM100 可用共享内存上限对比：

- **SMEM 上限**: `227 KiB = 232,448 B`
- **当前估算占用**: `264.25 KiB = 270,592 B`
- **超出上限**: `38,144 B = 37.25 KiB`

---

## 4. 这和当前 launch 报错的关系

如果 `SMEM_SIZE = sizeof(SharedMemoryPlan)` 的真实值接近上面的静态估算，那么当前双 buffer 配置下：

- `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE)`
- 很可能就会因为请求的 dynamic shared memory 超过硬件上限而直接返回 `invalid argument`

这和现在远端报错位置落在 [`dkv_phase.cuh`](/Users/chenql/Desktop/workspace/operator/FlashMLA/csrc/sm100/prefill/sparse/bwd/head128_2kernels/dkv_phase.cuh):534 是一致的。

换句话说，**在 256-row tile + 双 S/dS buffer 的前提下，dKV kernel 当前的 SMEM 预算已经明显超了**，而且超出的量不是边界抖动，而是大约 **37 KiB**。

---

## 5. 总结

1. 当前 256-block 四阶段 dKV 设计的 **TMEM 峰值已经打满**
- `512 / 512 cols`
- `256.00 KiB`

2. 在 **双 S/dS buffer** 前提下，当前 dKV kernel 的 **SMEM 估算为 264.25 KiB**
- 明显超过 SM100 的 `227 KiB` 上限
- 超出约 `37.25 KiB`

3. 因此，如果保持当前 256-row tile 和双 buffer 不变，当前资源角度的直接结论是：
- **TMEM 没有裕量**
- **SMEM 也已经无法 launch**

> 注：本文仍按 `transac_bar_t = 8 B` 的历史估算口径手工重算。即使真实 barrier 大小和这里略有偏差，也不足以改变“当前双 buffer SMEM 已超 227 KiB 上限”的结论。
