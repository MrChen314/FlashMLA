# dKV 的 `done` Barrier 必须汇总到 CTA0

## 问题概述

在 SM100 sparse backward 的 `dKV` 双 CTA kernel 里，用来保护 TMEM 复用的 `done` barrier，必须由两个 CTA 一起汇报到 CTA0，不能各自汇报到自己的本地 barrier。

这个问题会出现在 `dKV` 路径，是因为：

- 2CTA MMA 只有 CTA0 发起
- 只有 CTA0 会在下一轮复用 TMEM 前等待 `done` barrier
- 但 TMEM 到 global memory 的 drain 是两个 CTA 一起完成的

所以，CTA0 在发起下一轮 MMA 之前，必须观察到两个 CTA 的 drain 都已经完成。

## 涉及代码

- `FlashMLA/csrc/sm100/prefill/sparse/bwd/head128_2kernels/dkv_phase.cuh`
- `FlashMLA/csrc/sm100/prefill/sparse/bwd/head128_2kernels/dkv_config.h`

相关 CUTLASS barrier API：

- `FlashMLA/csrc/cutlass/include/cutlass/arch/barrier.h:376`
- `FlashMLA/csrc/cutlass/include/cutlass/arch/barrier.h:382`
- `FlashMLA/csrc/cutlass/include/cutlass/arch/barrier.h:485`

## 现象

当 `topk = 128` 时，kernel 只跑一轮，TMEM 不会跨轮复用，所以通常可以正常运行。

当 `topk = 256` 或更大时，TMEM 会在多轮迭代之间复用。如果 CTA0 没有等到两个 CTA 都完成上一轮 drain，就有可能提前发起下一轮 MMA，覆盖仍在被另一个 CTA 读取的 TMEM 数据。

这类问题可能表现为：

- kernel 卡死
- TMEM 复用顺序错误
- 只有循环次数大于 1 时才出现问题

## 根因

最初的 `done` 信号写法是：

```cpp
plan.bar_dkv_nope_done.arrive(static_cast<uint32_t>(cta_idx));
plan.bar_dkv_rope_done.arrive(static_cast<uint32_t>(cta_idx));
```

这对当前 kernel 是错误的。

`ClusterBarrier::arrive(uint32_t cta_id, ...)` 的语义是“对 `cta_id` 对应 CTA 持有的远端 barrier 做 arrive”，而不是“把当前 CTA 的编号带上去”。

因此，上面的旧写法实际变成了：

- CTA0 把完成信号汇报给 CTA0 自己的 barrier
- CTA1 把完成信号汇报给 CTA1 自己的 barrier

但下一轮复用 TMEM 之前，只有 CTA0 会去 wait：

- `FlashMLA/csrc/sm100/prefill/sparse/bwd/head128_2kernels/dkv_phase.cuh:300`
- `FlashMLA/csrc/sm100/prefill/sparse/bwd/head128_2kernels/dkv_phase.cuh:318`

这就意味着 CTA0 根本看不到 CTA1 的 drain 完成信号。

## 正确模型

两个 CTA 都必须向 CTA0 的 `done` barrier 汇报完成：

```cpp
plan.bar_dkv_nope_done.arrive(static_cast<uint32_t>(0));
plan.bar_dkv_rope_done.arrive(static_cast<uint32_t>(0));
```

这样 CTA0 就成为唯一的完成信号消费者，但它能收集到两个 CTA 的 drain 完成状态。

## Barrier 计数为什么要改

既然现在两个 CTA 都往 CTA0 的 barrier 上 arrive，那么 barrier 的初始计数也必须覆盖整个 cluster 中所有参与 arrive 的线程。

当前初始化写法：

```cpp
plan.bar_dkv_nope_done.init(4 * kThreadsPerWarpgroup);
plan.bar_dkv_rope_done.init(2 * kThreadsPerWarpgroup);
```

原因如下：

- `bar_dkv_nope_done` 会收到两个 CTA 的 WG0 和 WG1 的 arrive
- `bar_dkv_rope_done` 会收到两个 CTA 的 WG1 的 arrive
- `kThreadsPerWarpgroup = 128`

所以：

- NoPE done 总计数 = `4 * 128 = 512`
- RoPE done 总计数 = `2 * 128 = 256`

## 修复位置

Barrier 声明：

- `FlashMLA/csrc/sm100/prefill/sparse/bwd/head128_2kernels/dkv_config.h:137`
- `FlashMLA/csrc/sm100/prefill/sparse/bwd/head128_2kernels/dkv_config.h:138`

Barrier 初始化：

- `FlashMLA/csrc/sm100/prefill/sparse/bwd/head128_2kernels/dkv_phase.cuh:90`
- `FlashMLA/csrc/sm100/prefill/sparse/bwd/head128_2kernels/dkv_phase.cuh:91`

Drain 完成后对 CTA0 汇报：

- `FlashMLA/csrc/sm100/prefill/sparse/bwd/head128_2kernels/dkv_phase.cuh:244`
- `FlashMLA/csrc/sm100/prefill/sparse/bwd/head128_2kernels/dkv_phase.cuh:263`
- `FlashMLA/csrc/sm100/prefill/sparse/bwd/head128_2kernels/dkv_phase.cuh:279`

CTA0 在复用 TMEM 之前等待：

- `FlashMLA/csrc/sm100/prefill/sparse/bwd/head128_2kernels/dkv_phase.cuh:300`
- `FlashMLA/csrc/sm100/prefill/sparse/bwd/head128_2kernels/dkv_phase.cuh:318`
- `FlashMLA/csrc/sm100/prefill/sparse/bwd/head128_2kernels/dkv_phase.cuh:331`

## 额外的编译坑

直接写 `arrive(0)` 在 NVCC 下会产生二义性。

原因是：

- 成员函数里有 `arrive(uint32_t cta_id, uint32_t pred = true)`
- 静态重载集合里还有指针版本
- 字面量 `0` 既可以被当成整数，也可以被当成空指针常量

因此应写成：

```cpp
arrive(static_cast<uint32_t>(0))
```

这样可以强制命中“对远端 CTA barrier 做 arrive”的那个重载。

## 经验总结

如果一个 cluster barrier 表示的是某一步“已经完成”，而下一步的唯一消费者只在 CTA0，那么所有生产该完成信号的 CTA 都应该对 CTA0 的 barrier 实例做 arrive。

只有在“每个 CTA 后续都会等待自己本地 barrier”的场景下，才应该使用 `arrive(cta_idx)`。
