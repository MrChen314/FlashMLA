# SM100 Head128 反向双kernel dQ 资源占用分析

基于 `FlashMLA/csrc/sm100/prefill/sparse/bwd/head128_2kernels/dq_config.h` 的当前静态定义，以及修正后的 `FlashMLA/csrc/utils.py` 计算结果，以下是关于 SM100 架构上 `Head128` 反向双-kernel 方案中 `dQ kernel` 的 Shared Memory 和 TMEM 资源占用分析。

当前配置为：`D_QK = 576`（`D_V = 512`, `D_ROPE = 64`），`B_H = 128`，`B_TOPK = 64`。

对应命令：

```bash
python3 csrc/utils.py --config csrc/sm100/prefill/sparse/bwd/head128_2kernels/dq_config.h
```

---

## 1. 当前配置概要

新的 `dq_config.h` 已经不再是之前文档里的 `tQ / sQ` 方案，而是回到了更接近融合版的片上布局，只是语义上仍然服务于 `dQ-only` kernel：

1. `Q` 保持按 `QNoPE / QRoPE` 拆分
- `q_nope = [64, 512]`
- `q_rope = [64, 64]`

2. `K` 侧除了本 CTA 的 `k_nope / k_rope`，还额外保留 `kv_peer`
- `k_nope = [32, 512]`
- `k_rope = [32, 64]`
- `kv_peer = [32, 576]`

3. `dO` 现在常驻 SMEM，不再与 `Q` 复用

4. `TMEM` 只保存 `dQ / dQ_RoPE / P / dP`
- 没有 `tQ`
- 也没有 `dKV`

5. `B_TOPK` 已经从旧文档里的 `128` 变成 `64`
- 当前 softmax tile 是 `64 x 64`
- `S / dS` 的共享内存缓冲也随之缩小

因此，这版设计的资源特征是：

- `TMEM` 不再打满
- `SMEM` 反而非常紧，已经接近 `227 KiB` 上限

---

## 2. TMEM (Tensor Memory) 占用分析

`dq_config.h` 中的 TMEM 列划分如下：

```cpp
struct tmem_cols {
    static constexpr int dQ = 0;
    static constexpr int dQ_RoPE = 256;
    static constexpr int P = 288;
    static constexpr int dP = 320;
    static constexpr int kNumUsedCols = 352;
};
```

按 `utils.py` 的统一口径，TMEM 每段占用如下：

| 变量 | 描述 | 数据类型 | Logical Shape | 起始列 | 结束列 | 占用列数 | 占用大小 |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `dQ` | NoPE 部分 dQ 累加器 | fp32 | `[64, 512]` | 0 | 256 | **256** | 128 KiB |
| `dQ_RoPE` | RoPE 部分 dQ 累加器 | fp32 | `[64, 64]` | 256 | 288 | **32** | 16 KiB |
| `P` | logits / softmax 前缓冲 | fp32 | `[64, 64]` | 288 | 320 | **32** | 16 KiB |
| `dP` | dP 缓冲 | fp32 | `[64, 64]` | 320 | 352 | **32** | 16 KiB |
| **总计** |  |  |  |  |  | **352** | **176 KiB** |

### 结论

- **TMEM 使用量**: `352 / 512 cols` = `176.00 KiB / 256.00 KiB`
- **使用率**: **68.75%**
- **剩余裕量**: `160 cols` = **80.00 KiB**

和旧版 `tQ` 填满剩余 TMEM 的思路不同，这个版本把 TMEM 明确收敛到 `dQ + softmax中间量`，因此 TMEM 已经不是瓶颈。

---

## 3. Shared Memory (共享内存) 占用分析

### 3.1 `SharedMemoryPlan` 布局要点

当前 `dq_config.h` 的核心布局如下：

```cpp
struct alignas(128) SharedMemoryPlan {
    union {
        struct {
            array_aligned<bf16, cosize_v<SmemLayoutKNoPE>> k_nope;
            array_aligned<bf16, cosize_v<SmemLayoutKRoPE>> k_rope;
            array_aligned<bf16, cosize_v<SmemLayoutKV>> kv_peer;
            array_aligned<bf16, cosize_v<SmemLayoutQNoPE>> q_nope;
            array_aligned<bf16, cosize_v<SmemLayoutQRoPE>> q_rope;
        } q_kv;
        array_aligned<bf16, cosize_v<SmemLayoutQ>> dq;
    } u;

    array_aligned<bf16, cosize_v<SmemLayoutdO>> dO;
    struct {
        array_aligned<bf16, cosize_v<SmemLayoutdSTransposed>> s;
        array_aligned<bf16, cosize_v<SmemLayoutdSTransposed>> ds;
    } s_ds;
    char is_k_valid[B_TOPK / 8];
    ...
};
```

这里最重要的变化有三点：

1. `u.q_kv` 同时容纳 `K local + kv_peer + Q`
2. `u.dq` 单独作为 dQ 的 SMEM staging，与 `q_kv` 做 union 复用
3. `dO` 不参与 union，而是完整常驻

也就是说，当前 dQ kernel 的峰值 SMEM 已经不再来自 `dq`，而是来自 `q_kv` 这一大块输入 staging。

### 3.2 内存布局计算

#### A. Union 部分

1. `u.q_kv`
- `k_nope`: `[32, 512]` bf16 = `32 * 512 * 2` = **32,768 B**
- `k_rope`: `[32, 64]` bf16 = `32 * 64 * 2` = **4,096 B**
- `kv_peer`: `[32, 576]` bf16 = `32 * 576 * 2` = **36,864 B**
- `q_nope`: `[64, 512]` bf16 = `64 * 512 * 2` = **65,536 B**
- `q_rope`: `[64, 64]` bf16 = `64 * 64 * 2` = **8,192 B**

因此：

- `u.q_kv = 32,768 + 4,096 + 36,864 + 65,536 + 8,192`
- `u.q_kv = **147,456 B**`（**144.00 KiB**）

2. `u.dq`
- `dq`: `[64, 576]` bf16 = `64 * 576 * 2` = **73,728 B**（72.00 KiB）

因此：

- **Union 最大占用** = `max(147,456, 73,728)` = **147,456 B**（144.00 KiB）

#### B. 固定成员部分

1. `dO`
- `dO`: `[64, 512]` bf16 = `64 * 512 * 2` = **65,536 B**（64.00 KiB）

2. `s_ds`
- `s`: `[64, 64]` bf16 = `64 * 64 * 2` = **8,192 B**
- `ds`: `[64, 64]` bf16 = `64 * 64 * 2` = **8,192 B**
- 合计 = **16,384 B**（16.00 KiB）

3. `is_k_valid[B_TOPK / 8]`
- `64 / 8 = 8 B`

4. barriers
- 共 `13` 个 `transac_bar_t`
- 按 `8 B / 个` 估算 = **104 B**

5. `tmem_start_addr`
- `array_aligned<uint32_t, 1>` = **16 B**

6. 结构尾部对齐
- `alignas(128)` 下本次布局刚好对齐
- **tail padding = 0 B**

### 3.3 总占用量与使用率

| 组件 | 大小 (Bytes) | 大小 (KiB) |
| :--- | ---: | ---: |
| Union (Max) | 147,456 | 144.00 |
| `dO` | 65,536 | 64.00 |
| `s_ds` | 16,384 | 16.00 |
| `is_k_valid` | 8 | ~0.01 |
| barriers (估算) | 104 | ~0.10 |
| `tmem_start_addr` | 16 | ~0.02 |
| tail padding | 0 | 0.00 |
| **总计** | **229,504** | **224.12** |

- **SM100 共享内存上限**: `227 KiB = 232,448 B`
- **预估使用量**: `229,504 B = 224.12 KiB`
- **使用率**: **98.73%**
- **剩余裕量**: `2,944 B = 2.88 KiB`

### 结论

这版 `dq_config.h` 在 TMEM 上很宽松，但在 SMEM 上已经非常贴边：

- 当前方案 SMEM: `229,504 B`（224.12 KiB）
- SM100 上限: `232,448 B`（227.00 KiB）
- **剩余空间只有**: `2,944 B`（2.88 KiB）

也就是说，当前 dQ kernel 的主瓶颈已经从 TMEM 转移到了 SMEM，后续如果再往 `SharedMemoryPlan` 中增加常驻片上状态，基本就会有超限风险。

---

## 4. `utils.py` 对应输出摘录

修正 profile 识别后，`utils.py` 的输出如下：

```text
Config: csrc/sm100/prefill/sparse/bwd/head128_2kernels/dq_config.h
Profile: dq_2kernels_kv_peer
Constants: D_QK=576, D_V=512, D_ROPE=64, B_H=128, B_TOPK=64

SMEM:
  k_nope        = (32 x 512 x 2)   =  32768 B  (32.00 KiB)
  k_rope        = (32 x 64 x 2)    =   4096 B  (4.00 KiB)
  kv_peer       = (32 x 576 x 2)   =  36864 B  (36.00 KiB)
  q_nope        = (64 x 512 x 2)   =  65536 B  (64.00 KiB)
  q_rope        = (64 x 64 x 2)    =   8192 B  (8.00 KiB)
  dq            = (64 x 576 x 2)   =  73728 B  (72.00 KiB)
  dO            = (64 x 512 x 2)   =  65536 B  (64.00 KiB)
  s             = (64 x 64 x 2)    =   8192 B  (8.00 KiB)
  ds            = (64 x 64 x 2)    =   8192 B  (8.00 KiB)
  union.q_kv    = 147456 B  (144.00 KiB)
  union.dq      =  73728 B  (72.00 KiB)
  union.u       = 147456 B  (144.00 KiB)
  s_ds          =  16384 B  (16.00 KiB)
  is_k_valid    =      8 B
  barriers(est) = 13 x 8 B =    104 B
  tmem_start    =     16 B  (array_aligned<uint32_t, 1>)
  tail padding  =      0 B
  total         = 229504 B  (224.12 KiB)
  limit check   = OK  (224.12 KiB / 227.00 KiB, 98.73%)

TMEM:
  dQ       = col[  0:256)  -> [64 x 512] fp32  = 256 cols  (128.00 KiB)
  dQ_RoPE  = col[256:288)  -> [64 x 64] fp32   =  32 cols  (16.00 KiB)
  P        = col[288:320)  -> [64 x 64] fp32   =  32 cols  (16.00 KiB)
  dP       = col[320:352)  -> [64 x 64] fp32   =  32 cols  (16.00 KiB)
  total        = 352 / 512 cols  (176.00 KiB / 256.00 KiB, 68.75%)
  limit check  = OK
```

---

## 5. 总体结论

1. **当前 dQ 双-kernel 配置已经不是旧的 `tQ / sQ` 方案**
- 文档中原先关于 `tQ` 占用 TMEM、`sQ` 常驻 SMEM、TMEM 打满 `512 cols` 的结论都不再适用

2. **TMEM 现在比较宽松**
- `352 / 512 cols`
- 使用率 **68.75%**

3. **SMEM 才是当前真正的硬件边界**
- `229,504 / 232,448 B`
- 使用率 **98.73%**

4. **当前布局的主要成本来自 `u.q_kv`**
- `K local`
- `kv_peer`
- `QNoPE / QRoPE`
- 再加上常驻 `dO`

5. **后续优化方向如果还要扩状态，优先看 SMEM 而不是 TMEM**
- 现在的 SMEM 裕量只有 **2.88 KiB**
- 再增加新的 staging buffer、额外 barrier 或更大的 tile，都需要非常谨慎

> 注：以上结论基于修正后的 `FlashMLA/csrc/utils.py --config csrc/sm100/prefill/sparse/bwd/head128_2kernels/dq_config.h` 静态估算；其中 barrier 仍按 `8 B / 个` 估值。
