# SM100 Head128 反向双kernel dKV 资源占用分析

> 注意：这份分析文档基于旧版 `dkv_config.h` 假设撰写，尚未反映当前 `dKV` 已切换为 2CTA MMA 的新设计。
> 尤其是 `dV = S^T @ dO` 与 `dK = dS^T @ Q` 现在都按 2CTA 模式处理，输出在 `M` 维切半、`K = 128` 保持完整，因此本文中的 TMEM/布局结论已不再准确，需要后续整体重算。

基于 `FlashMLA/csrc/sm100/prefill/sparse/bwd/head128_2kernels/dkv_config.h` 的静态定义，以及 `FlashMLA/csrc/utils.py` 的计算结果，以下是关于 SM100 架构上 `Head128` 反向双-kernel 方案中 `dKV kernel` 的 Shared Memory 和 TMEM 资源占用分析。

假设配置：`D_QK = 576`（`D_V = 512`, `D_ROPE = 64`），`B_H = 128`，`B_TOPK = 128`。

## 1. 配置调整概要

`dKV kernel` 的设计目标和 `dQ kernel` 不同：它不再重建 softmax，也不再计算 `dQ`，而是直接消费中间 scratch `S / dS`，完成 `dV + dK` 的分阶段累加。

这份 `dkv_config.h` 的关键取舍是：

1. **不再在片上保留 K / V**
- `dV = S^T @ dO`
- `dK = dS^T @ Q`
- 从这两个公式出发，片上常驻只需要 `Q / dO / S / dS`

2. **Q 按前向同款的 `tQ / sQ` 口径描述**
- 逻辑上 `Q = [tQ, sQ]`
- `D_tQ = 384`
- `D_sQ = 192`
- 但在 `dKV kernel` 中，`Q` 整体常驻于 SMEM，`tQ / sQ` 主要用于 dK 分阶段计算的说明

3. **TMEM 只保留一个 `dKV` 缓冲并在不同 phase 复用**
- phase 1：`dKV` 承担 `dV[0:512]`
- phase 2：同一片 `dKV` 复用为 `dK_tQ[0:384]`
- phase 3：再复用前 192 列为 `dK_sQ[0:192]`

这套布局的好处是：SMEM 压力明显小于融合版，同时 TMEM 的使用方式非常直接，适合 dKV-only kernel。

---

## 2. TMEM (Tensor Memory) 占用分析

`dkv_config.h` 中的 TMEM 列划分为：

```cpp
struct tmem_cols {
    static constexpr int dKV = 0;
    static constexpr int kNumUsedCols = 512;
};
```

这里不是多个并行段，而是**单一 `dKV` 缓冲的时间复用**：

| 缓冲 | 描述 | 数据类型 | 峰值 Logical Shape | 起始列 | 结束列 | 占用列数 | 占用大小 |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `dKV` | 统一的 dKV TMEM 累加缓冲 | fp32 | `[128, 512]` | 0 | 512 | **512** | 256 KB |
| `dKV` phase 说明 | phase 1: dV，phase 2: dK_tQ，phase 3: dK_sQ |  |  |  |  |  |  |

### 结论

- **TMEM 峰值使用量**: 512 列 = 256 KB
- **SM100 TMEM 总容量**: 512 列 = 256 KB
- **使用率**: **100%**

虽然 `dK_tQ / dK_sQ` 两个阶段本身都小于 512 列，但因为统一缓冲 `dKV` 的峰值阶段是 `dV[128, 512]`，所以 dKV kernel 的 TMEM 峰值仍然是满占用。

---

## 3. Shared Memory (共享内存) 占用分析

### 3.1 `SharedMemoryPlan` 布局要点

`dkv_config.h` 的 `SharedMemoryPlan` 如下：

```cpp
struct alignas(128) SharedMemoryPlan {
    array_aligned<bf16, cosize_v<SmemLayoutQ>> q;
    array_aligned<bf16, cosize_v<SmemLayoutdO>> dO;
    struct {
        array_aligned<bf16, cosize_v<SmemLayoutdSTransposed>> s;
        array_aligned<bf16, cosize_v<SmemLayoutdSTransposed>> ds;
    } s_ds;
    char is_k_valid[B_TOPK / 8];
    ...
};
```

与 `dQ kernel` 相比，这版最显著的区别是：

1. 没有 `K`
2. 没有 `dQ`
3. 没有 `P / dP`
4. 也没有 `kv_peer`

因此，`dKV kernel` 的 SMEM 只承载真正参与 dKV 计算的 4 类输入：

- `Q`
- `dO`
- `S`
- `dS`

### 3.2 内存布局计算

#### A. 输入常驻区

1. `q`
- `q`: `[64, 576]` bf16 = `64 * 576 * 2` = **73,728 B**（72.0 KB）
- 逻辑上可拆为：
  - `tQ`: `[64, 384]` bf16 = **49,152 B**（48.0 KB）
  - `sQ`: `[64, 192]` bf16 = **24,576 B**（24.0 KB）

2. `dO`
- `dO`: `[64, 512]` bf16 = `64 * 512 * 2` = **65,536 B**（64.0 KB）

3. `s_ds`
- `s`: `[64, 128]` bf16 = `64 * 128 * 2` = **16,384 B**
- `ds`: `[64, 128]` bf16 = `64 * 128 * 2` = **16,384 B**
- 合计 = **32,768 B**（32.0 KB）

#### B. 辅助区

4. `is_k_valid[B_TOPK / 8]`
- `128 / 8 = 16 B`

5. barriers
- 共 `12` 个 `transac_bar_t`
- 按 `8B / 个` 估算 = **96 B**

6. `tmem_start_addr`
- `array_aligned<uint32_t, 1>` = **16 B**

7. 结构尾部对齐
- `alignas(128)` 下本次布局刚好对齐
- **tail padding = 0 B**

### 3.3 总占用量与使用率

| 组件 | 大小 (Bytes) | 大小 (KB) |
| :--- | ---: | ---: |
| `q` | 73,728 | 72.0 |
| `dO` | 65,536 | 64.0 |
| `s_ds` | 32,768 | 32.0 |
| `is_k_valid` | 16 | ~0.0 |
| barriers (估算) | 96 | ~0.1 |
| `tmem_start_addr` | 16 | ~0.0 |
| tail padding | 0 | 0.0 |
| **总计** | **172,160** | **168.12** |

- **SM100 共享内存上限**: 227 KB = 232,448 B
- **预估使用量**: 172,160 B = 168.12 KB
- **使用率**: **74.06%**
- **剩余裕量**: **60,288 B** = **58.88 KB**

### 结论

`dKV kernel` 的 SMEM 压力明显低于 `dQ kernel`：

- `dQ kernel` SMEM: `196,736 B`（192.12 KB）
- `dKV kernel` SMEM: `172,160 B`（168.12 KB）
- **减少**: `24,576 B`（24.0 KB）

这部分差值，基本就来自 `dKV kernel` 不再需要片上保留 `K`。

---

## 4. 总体结论

1. **dKV kernel 的片上瓶颈主要在 TMEM，不在 SMEM**
- TMEM 峰值仍是 `512 / 512 cols`
- 但 SMEM 只有 `168.12 KB`

2. **删除 K/V 片上 staging 是成立的**
- 从 `dV = S^T @ dO` 与 `dK = dS^T @ Q` 的依赖关系出发，`K / V` 并不是 dKV kernel 的必要常驻输入

3. **BH / BTOPK 都取 128 时仍然安全**
- TMEM：满占用但不超限
- SMEM：`74.06%`，仍有约 `58.88 KB` 裕量

4. **资源形态总结**
- TMEM：承担统一 `dKV` 缓冲，并在 `dV / dK_tQ / dK_sQ` 三个 phase 间复用
- SMEM：承担 `Q / dO / S / dS / 有效性掩码`

> 注：以上结论基于 `FlashMLA/csrc/utils.py --config csrc/sm100/prefill/sparse/bwd/head128_2kernels/dkv_config.h` 的静态估算；其中 barrier 仍按 `8 B / 个` 估值。
