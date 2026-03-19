# SM100 Head128 反向双kernel dKV 资源占用分析

基于当前 `FlashMLA/csrc/sm100/prefill/sparse/bwd/head128_2kernels/dkv_config.h` 的静态定义，以及 `FlashMLA/csrc/sm100/prefill/sparse/bwd/head128/config.h` 中沿用的公共常量，以下是 `Head128` 反向双-kernel 方案里 `dKV kernel` 的 Shared Memory 和 TMEM 资源占用分析。

需要先说明两点：

1. `dkv_config.h` 只覆盖了 `dKV-only` kernel 自己的布局，没有重新定义 `D_QK / D_V / D_ROPE / B_H / B_TOPK`，因此这些公共常量仍沿用 `head128/config.h` 的当前值。
2. 当前 `python3 csrc/utils.py --config csrc/sm100/prefill/sparse/bwd/head128_2kernels/dkv_config.h` 还保留着旧版 `D_tQ / D_sQ` profile 假设，会直接报错；因此本文按头文件静态定义手工重算，不再复用旧文档里的结论。

当前配置为：`D_QK = 576`（`D_V = 512`, `D_ROPE = 64`），`B_H = 128`，`B_TOPK = 64`。

---

## 1. 当前配置概要

和旧文档相比，当前 `dkv_config.h` 已经完全不是“`tQ / sQ + 单 TMEM 缓冲三阶段复用`”那套设计了。现在的关键特征是：

1. `SharedMemoryPlan` 只保留 `Q / QRoPE / dO / S / dS`
- 没有 `K`
- 没有 `V`
- 没有 `kv_peer`
- 也没有 `is_k_valid`

2. NoPE 路径切成了 **2CTA cluster MMA**
- `TiledMMA_dKV` 使用 `SM100_MMA_F16BF16_2x1SM_SS_NOELECT<..., B_TOPK, 256, ...>`
- `SmemLayoutQ / SmemLayoutdO` 也只保留 `[B_H, 256]`
- 结合 `Tile<Int<128>, Layout<Shape<_128, _2, _2>, ...>>` 可以推断：**每个 CTA 只暂存 256 列 NoPE 操作数，两个 CTA 合起来覆盖逻辑上的 512 列**

3. RoPE 路径单独走一套更小的 2CTA 累加器
- `SmemLayoutQRoPE` 是 `[B_H, 32]`
- `tmem_cols::dKV_RoPE` 单独保留 RoPE 输出段
- 同理可推断：**每个 CTA 只持有 32 列 RoPE 操作数，两 CTA cluster 合起来覆盖逻辑上的 64 列**

4. TMEM 不再是一个被 `dV / dK_tQ / dK_sQ` 轮流复用的大缓冲
- 现在是 `dKV` 和 `dKV_RoPE` 两段并存
- `kNumUsedCols = 288`
- 因此 TMEM 峰值已经从旧文档里的“满 512 cols”降到了当前的 `288 cols`

也就是说，当前 `dKV kernel` 的资源模型应该理解成：

- `SMEM`：每 CTA 保存 2CTA MMA 所需的输入 tile
- `TMEM`：保存 NoPE 和 RoPE 两段独立的 fp32 累加结果

---

## 2. TMEM (Tensor Memory) 占用分析

当前 `dkv_config.h` 中的 TMEM 列划分如下：

```cpp
struct tmem_cols {
    // fp32 [64, 512] -> 128kb
    static constexpr int dKV = 0;
    // fp32 [64, 64] -> 16kb
    static constexpr int dKV_RoPE = 256;
    static constexpr int kNumUsedCols = dKV_RoPE + D_ROPE / 2;
};
```

按注释和列号可直接得到：

| 缓冲 | 描述 | 数据类型 | Logical Shape | 起始列 | 结束列 | 占用列数 | 占用大小 |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `dKV` | NoPE 部分 `dV + dK` 累加器 | fp32 | `[64, 512]` | 0 | 256 | **256** | **128.00 KiB** |
| `dKV_RoPE` | RoPE 部分 `dK` 累加器 | fp32 | `[64, 64]` | 256 | 288 | **32** | **16.00 KiB** |
| **总计** |  |  |  |  |  | **288** | **144.00 KiB** |

### 结论

- **TMEM 使用量**: `288 / 512 cols` = `144.00 KiB / 256.00 KiB`
- **使用率**: **56.25%**
- **剩余裕量**: `224 cols` = **112.00 KiB**

这和旧文档最大的不同在于：

1. 现在没有 `dKV` 单缓冲跨 `dV / dK_tQ / dK_sQ` 三阶段复用的说法
2. `B_TOPK` 也不是旧文档假设的 `128`，而是当前沿用的 `64`
3. 因此 TMEM 峰值不再打满，而是稳定落在 **288 cols**

---

## 3. Shared Memory (共享内存) 占用分析

### 3.1 `SharedMemoryPlan` 布局要点

当前 `dkv_config.h` 的 `SharedMemoryPlan` 如下：

```cpp
struct alignas(128) SharedMemoryPlan {
    array_aligned<bf16, cosize_v<SmemLayoutQ>> q;
    array_aligned<bf16, cosize_v<SmemLayoutQRoPE>> q_rope;
    array_aligned<bf16, cosize_v<SmemLayoutdO>> dO;
    struct {
        array_aligned<bf16, cosize_v<SmemLayoutS>> s;
        array_aligned<bf16, cosize_v<SmemLayoutdS>> ds;
    } s_ds;

    transac_bar_t bar_q_nope_ready;
    transac_bar_t bar_q_rope_ready;
    transac_bar_t bar_dO_ready;
    transac_bar_t bar_s_tile_ready;
    transac_bar_t bar_ds_tile_ready;
    transac_bar_t bar_dkv_nope_ready;
    transac_bar_t bar_dkv_rope_ready;

    array_aligned<uint32_t, 1> tmem_start_addr;
};
```

这版布局的几个关键信号非常明确：

1. 没有 union
- `q / q_rope / dO / s / ds` 全都常驻

2. 没有 `is_k_valid`
- 说明 `dKV-only` kernel 当前不再为有效性掩码单独预留片上字节

3. 输入 tile 规模已经换成 2CTA 口径
- `q`: `[128, 256]`
- `q_rope`: `[128, 32]`
- `dO`: `[128, 256]`
- `s / ds`: `[128, 32]`

其中 `q / q_rope / dO` 的 shape 是 **每 CTA 的 staging 大小**，并不是最终逻辑 head 宽度；最终逻辑宽度仍分别对应 `512` 和 `64`，只是由 2CTA cluster 共同覆盖。

### 3.2 内存布局计算

以下仍按当前文档和 `utils.py` 一致的估算口径：

- `array_aligned<T, N>` 按 `16 B` 对齐
- `transac_bar_t` 继续按 **`8 B / 个`** 估算
- 结构体整体按 `alignas(128)` 收尾对齐

#### A. 数据区

1. `q`
- `q`: `[128, 256]` bf16 = `128 * 256 * 2` = **65,536 B**（64.00 KiB）

2. `q_rope`
- `q_rope`: `[128, 32]` bf16 = `128 * 32 * 2` = **8,192 B**（8.00 KiB）

3. `dO`
- `dO`: `[128, 256]` bf16 = `128 * 256 * 2` = **65,536 B**（64.00 KiB）

4. `s_ds`
- `s`: `[128, 32]` bf16 = `128 * 32 * 2` = **8,192 B**（8.00 KiB）
- `ds`: `[128, 32]` bf16 = `128 * 32 * 2` = **8,192 B**（8.00 KiB）
- 合计 = **16,384 B**（16.00 KiB）

因此，纯数据区合计为：

- `65,536 + 8,192 + 65,536 + 8,192 + 8,192`
- **155,648 B**（**152.00 KiB**）

#### B. 同步与辅助区

1. barriers
- 共 `7` 个 `transac_bar_t`
- 按 `8 B / 个` 估算 = **56 B**

2. `tmem_start_addr`
- `array_aligned<uint32_t, 1>` = **16 B**

3. 结构尾部对齐
- 数据区结束后偏移：`155,648 B`
- 加 barrier 后：`155,704 B`
- 向 `16 B` 对齐后：`155,712 B`
- 再加 `tmem_start_addr`：`155,728 B`
- 向 `128 B` 对齐收尾，需要 **48 B** tail padding

### 3.3 总占用量与使用率

| 组件 | 大小 (Bytes) | 大小 (KiB) |
| :--- | ---: | ---: |
| `q` | 65,536 | 64.00 |
| `q_rope` | 8,192 | 8.00 |
| `dO` | 65,536 | 64.00 |
| `s_ds` | 16,384 | 16.00 |
| barriers (估算) | 56 | 0.05 |
| `tmem_start_addr` | 16 | 0.02 |
| tail padding | 48 | 0.05 |
| **总计** | **155,776** | **152.12** |

- **SM100 共享内存上限**: `227 KiB = 232,448 B`
- **预估使用量**: `155,776 B = 152.12 KiB`
- **使用率**: **67.02%**
- **剩余裕量**: `76,672 B = 74.88 KiB`

### 结论

当前 `dKV kernel` 的 SMEM 压力其实已经很宽松，不再是旧文档里那种“只比上限低一点点”的状态。

如果和同目录下当前的 `dQ kernel` 文档对比：

- `dQ kernel` SMEM: `229,504 B`（224.12 KiB）
- `dKV kernel` SMEM: `155,776 B`（152.12 KiB）
- **差值**: `73,728 B`（72.00 KiB）

这部分收益主要来自：

1. `dKV kernel` 不再保留 `K / kv_peer / dq`
2. 没有 `is_k_valid`
3. NoPE / RoPE 输入 staging 已按 2CTA tile 收缩到每 CTA `256 / 32` 列

---

## 4. 总体结论

1. **旧版 `dKV` 资源文档的核心结论已经全部失效**
- 不再是 `tQ / sQ`
- 不再是 `B_TOPK = 128`
- 也不再是“一个 `dKV` TMEM 大缓冲三阶段复用”

2. **当前 `dKV kernel` 既不紧 TMEM，也不紧 SMEM**
- TMEM: `288 / 512 cols`，使用率 **56.25%**
- SMEM: `155,776 / 232,448 B`，使用率 **67.02%**

3. **当前资源占用最值得关注的变化，是 per-CTA 输入 tile 明显缩小**
- `q / dO` 从“按整块 512 NoPE 宽度理解”变成了每 CTA 只放 `256`
- `q_rope` 从“整块 64”变成了每 CTA 只放 `32`
- 两 CTA cluster 共同覆盖完整逻辑宽度

4. **从片上存储角度看，当前 dKV-only 设计是安全的**
- TMEM 还剩 **112.00 KiB**
- SMEM 还剩 **74.88 KiB**
- 后续如果继续在 `dKV` kernel 里加状态，当前空间也明显比 `dQ` 侧宽松得多

> 注：本文完全按当前 `dkv_config.h` 静态布局推导；其中 `transac_bar_t` 仍按 `8 B / 个` 估值，2CTA 对完整逻辑宽度的覆盖关系则是根据 `TiledMMA_dKV / TiledMMA_dKV_RoPE` 的 tile 形状做出的推断。
