# SM100 Head128 反向双kernel dQ 资源占用分析

基于当前 `FlashMLA/csrc/sm100/prefill/sparse/bwd/head128_2kernels/dq_config.h` 和 `FlashMLA/csrc/sm100/prefill/sparse/bwd/head128_2kernels/dq_phase.cuh` 的静态定义，以下重新整理 `Head128` 反向双-kernel 方案中 `dQ kernel` 的 Shared Memory 和 TMEM 资源占用。

需要先说明一点：

1. 当前 `python3 csrc/utils.py --config csrc/sm100/prefill/sparse/bwd/head128_2kernels/dq_config.h` 还没有完全跟上这版 `dq_config.h`。
2. 它仍按旧的 `dq_2kernels` profile 估算 `SMEM`，也没有把 `tmem_cols::q` 这一段算进 `TMEM`。
3. 因此本文不再直接复用脚本输出，而是按最新头文件和实际访问路径手工重算。

当前配置为：

- `D_QK = 576`
- `D_V = 512`
- `D_ROPE = 64`
- `D_tQ = 320`
- `D_sQ = 256`
- `B_H = 128`
- `B_TOPK = 64`

---

## 1. 当前配置概要

这版 `dq_config.h` 已经回到“`sQ` 放 SMEM、`tQ` 放 TMEM”的混合方案，不再是旧文档里那种“`TMEM` 只放 `dQ / P / dP`，`SMEM` 接近打满”的布局。

关键变化有四点：

1. `Q` 的 prologue staging 明确拆成两段用途
- `u.q_full` 先完整接住 `QNoPE + QRoPE`
- 其中 `tQ = 320` 列随后通过 `UTCCP` 搬到 `TMEM`
- `sQ = 256` 列则留在 `SMEM`，供 `P_sQ` 路径使用

2. `SharedMemoryPlan` 的 union 现在有三种视图
- `q_full`：Q 的完整 prologue landing
- `q_kv`：主循环期间复用成 `sq + local KV[2] + kv_peer`
- `dq`：收尾阶段复用成 dQ 的 `SMEM` staging

3. `dO` 仍然常驻 `SMEM`
- 它不在 union 内
- 但本地 `KV` 现在已经改成双 buffer，因此 `SMEM` 又重新变得很紧

4. `TMEM` 重新被打满
- `dQ / dQ_RoPE / P / dP` 之外，新增了 `q`
- `static_assert(tmem_cols::kNumUsedCols == 512)` 已经把这一点直接钉死

因此，当前版本更准确的资源画像应该是：

- `TMEM`：**512 / 512 cols，逻辑上打满**
- `SMEM`：因为本地 `KV` 变成 ping-pong 双 buffer，再次回到接近上限的区间

---

## 2. TMEM (Tensor Memory) 占用分析

当前 `dq_config.h` 中的 TMEM 列划分如下：

```cpp
struct tmem_cols {
    static constexpr int dQ = 0;
    static constexpr int dQ_RoPE = 256;
    static constexpr int P = 288;
    static constexpr int dP = 320;
    static constexpr int q = 352;
    static constexpr int kNumUsedCols = q + D_tQ / 2;
};

static_assert(tmem_cols::kNumUsedCols == 512,
              "dq kernel should fully use the 512 logical TMEM columns after staging tq.");
```

按当前代码语义，各段占用如下：

| 变量 | 描述 | 数据类型 | Logical Shape | 起始列 | 结束列 | 占用列数 | 占用大小 |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `dQ` | NoPE 部分 dQ 累加器 | fp32 | `[64, 512]` | 0 | 256 | **256** | **128.00 KiB** |
| `dQ_RoPE` | RoPE 部分 dQ 累加器 | fp32 | `[64, 64]` | 256 | 288 | **32** | **16.00 KiB** |
| `P` | logits / softmax 前缓冲 | fp32 | `[64, 64]` | 288 | 320 | **32** | **16.00 KiB** |
| `dP` | dP 缓冲 | fp32 | `[64, 64]` | 320 | 352 | **32** | **16.00 KiB** |
| `q` | `tQ` 的 TMEM operand-A staging | bf16 | `[128, 320]` | 352 | 512 | **160** | **80.00 KiB** |
| **总计** |  |  |  |  |  | **512** | **256.00 KiB** |

这里最容易看错的是最后一段 `q`：

1. 它不是旧脚本里缺失掉的“可忽略暂存”
2. 它对应 `TiledMMA_P_tQ` 的 TMEM A operand
3. 占用列数直接由 `D_tQ / 2 = 160` 给出，所以 `TMEM` 总列数正好补满到 `512`

### 结论

- **TMEM 使用量**: `512 / 512 cols` = `256.00 KiB / 256.00 KiB`
- **使用率**: **100.00%**
- **剩余裕量**: `0 cols`

和旧文档相比，最大的纠正点就是：

1. `TMEM` 并不是 `352 / 512 cols`
2. `tQ` 事实上已经重新回到了 `TMEM`
3. 当前 dQ kernel 的 `TMEM` 逻辑列布局已经满配

---

## 3. Shared Memory (共享内存) 占用分析

### 3.1 `SharedMemoryPlan` 布局要点

当前 `dq_config.h` 的核心布局如下：

```cpp
struct alignas(128) SharedMemoryPlan {
    union {
        array_aligned<bf16, cosize_v<SmemLayoutQ>> q_full;
        struct {
            array_aligned<bf16, cosize_v<SmemLayoutQTiles<NUM_sQ_TILES>>> sq;
            array_aligned<bf16, cosize_v<SmemLayoutKV>> kv[NUM_KV_BUFS];
            array_aligned<bf16, cosize_v<SmemLayoutKV>> kv_peer;
        } q_kv;
        array_aligned<bf16, cosize_v<SmemLayoutQ>> dq;
    } u;

    array_aligned<bf16, cosize_v<SmemLayoutdO>> dO;
    struct {
        array_aligned<bf16, cosize_v<SmemLayoutdSTransposed>> s;
        array_aligned<bf16, cosize_v<SmemLayoutdSTransposed>> ds;
    } s_ds;
    char is_k_valid[B_TOPK / 8];

    transac_bar_t bar_prologue_q_nope;
    transac_bar_t bar_prologue_q_rope;
    transac_bar_t bar_prologue_utccp;
    transac_bar_t bar_prologue_kv;
    transac_bar_t bar_prologue_dO;
    transac_bar_t bar_p_ready;
    transac_bar_t bar_dp_ready;
    transac_bar_t bar_s_ready;
    transac_bar_t bar_ds_ready;
    transac_bar_t bar_k_valid_free;
    transac_bar_t bar_k_valid_ready;
    transac_bar_t bar_kv_peer_nope_ready;
    transac_bar_t bar_kv_peer_rope_ready;
    transac_bar_t bar_dq_ready;

    array_aligned<uint32_t, 1> tmem_start_addr;
};
```

这版布局最重要的变化是：

1. `u.q_full` 只负责 prologue 阶段完整承接 `Q`
2. 主循环里，union 复用成 `u.q_kv = sq + local KV[2] + kv_peer`
3. 最终从 `TMEM` 回写 dQ 时，再复用成 `u.dq`
4. `dO`、`s / ds`、`is_k_valid` 和所有 barrier 都是常驻成员

也就是说，当前 `SMEM` 峰值已经不再来自“`Q + K + kv_peer + dO` 全都同时常驻”的旧布局，而是来自：

- `u.q_kv`
- 常驻 `dO`
- 常驻 `s_ds`
- 常驻同步元数据

### 3.2 内存布局计算

以下仍按当前仓库文档一贯的静态估算口径：

- `array_aligned<T, N>` 按 `16 B` 对齐
- `transac_bar_t` 继续按 **`8 B / 个`** 估算
- 结构体整体按 `alignas(128)` 收尾对齐

#### A. Union 部分

1. `u.q_full`
- `q_full`: `[64, 576]` bf16 = `64 * 576 * 2` = **73,728 B**（**72.00 KiB**）

2. `u.q_kv`
- `sq`: `[64, 256]` bf16 = `64 * 256 * 2` = **32,768 B**（**32.00 KiB**）
- `kv[2]`: `2 * [32, 576]` bf16 = `2 * 32 * 576 * 2` = **73,728 B**（**72.00 KiB**）
- `kv_peer`: `[32, 576]` bf16 = `32 * 576 * 2` = **36,864 B**（**36.00 KiB**）

因此：

- `u.q_kv = 32,768 + 73,728 + 36,864`
- `u.q_kv = **143,360 B**`（**140.00 KiB**）

3. `u.dq`
- `dq`: `[64, 576]` bf16 = `64 * 576 * 2` = **73,728 B**（**72.00 KiB**）

因此：

- **Union 最大占用** = `max(73,728, 143,360, 73,728)` = **143,360 B**（**140.00 KiB**）

#### B. 固定成员部分

1. `dO`
- `dO`: `[64, 512]` bf16 = `64 * 512 * 2` = **65,536 B**（**64.00 KiB**）

2. `s_ds`
- `s`: `[64, 64]` bf16 = `64 * 64 * 2` = **8,192 B**
- `ds`: `[64, 64]` bf16 = `64 * 64 * 2` = **8,192 B**
- 合计 = **16,384 B**（**16.00 KiB**）

3. `is_k_valid[B_TOPK / 8]`
- `64 / 8 = 8 B`

4. barriers
- 共 `14` 个 `transac_bar_t`
- 按 `8 B / 个` 估算 = **112 B**

5. `tmem_start_addr`
- `array_aligned<uint32_t, 1>` = **16 B**

6. 结构尾部对齐
- 按当前布局估算，**tail padding = 112 B**

### 3.3 总占用量与使用率

| 组件 | 大小 (Bytes) | 大小 (KiB) |
| :--- | ---: | ---: |
| Union (Max) | 143,360 | 140.00 |
| `dO` | 65,536 | 64.00 |
| `s_ds` | 16,384 | 16.00 |
| `is_k_valid` | 8 | ~0.01 |
| barriers (估算) | 112 | ~0.11 |
| `tmem_start_addr` | 16 | ~0.02 |
| tail padding | 112 | ~0.11 |
| **总计** | **225,536** | **220.25** |

- **SM100 共享内存上限**: `227 KiB = 232,448 B`
- **预估使用量**: `225,536 B = 220.25 KiB`
- **使用率**: **97.03%**
- **剩余裕量**: `6,912 B = 6.75 KiB`

### 结论

和单 buffer `kv` 那版相比，这次把本地 `KV` 改成双 buffer 后，`SMEM` 又明显抬高，重新接近上限：

- 当前方案 `SMEM`: `225,536 B`（**220.25 KiB**）
- SM100 上限: `232,448 B`（**227.00 KiB**）
- **剩余空间**: `6,912 B`（**6.75 KiB**）

也就是说，最新 dQ kernel 的压力画像已经变成：

1. `TMEM` 逻辑列被完全占满
2. `SMEM` 也重新回到“只差几 KiB 就接近超限”的状态

---

## 4. 与旧文档结论的差异

旧文档里有三条核心结论已经不再适用：

1. **“TMEM 只用了 `352 / 512 cols`”**
- 现在应改为 **`512 / 512 cols`**

2. **“当前 dQ kernel 的主瓶颈已经从 TMEM 转移到 SMEM”**
- 现在不能再这么写
- 最新版本是 `TMEM` 逻辑列打满，同时 `SMEM` 也只剩 **6.75 KiB** 裕量

3. **“`u.q_kv` 仍然是 `K local + kv_peer + Q` 的大块 staging”**
- 现在更准确的说法是 `sq + local KV[2] + kv_peer`
- `tQ` 已经被抽到 `TMEM`
- `q_full` 只在 prologue / UTCCP 阶段短暂使用

---

## 5. 总体结论

1. **最新 dQ 双-kernel 配置已经重新回到 `sQ + tQ` 分拆**
- `sQ = 256` 列留在 `SMEM`
- `tQ = 320` 列进入 `TMEM`

2. **TMEM 现在是满配**
- `512 / 512 cols`
- 使用率 **100.00%**

3. **SMEM 也重新变得很紧**
- `225,536 / 232,448 B`
- 使用率 **97.03%**

4. **当前资源占用的主特征是**
- `TMEM` 被 `dQ + dQ_RoPE + P + dP + q` 完整填满
- `SMEM` 则主要被 `u.q_kv` 里的本地 `KV` 双 buffer 拉高

5. **如果后续还要继续扩状态，需要同时警惕 `TMEM` 和 `SMEM`**
- `TMEM` 侧没有剩余 logical cols
- `SMEM` 侧也只剩 **6.75 KiB** 裕量

> 注：TMEM 部分由 `tmem_cols` 和 `static_assert(kNumUsedCols == 512)` 直接给出；SMEM 部分仍按当前仓库文档口径做静态估算，其中 `transac_bar_t` 按 `8 B / 个` 估值。
