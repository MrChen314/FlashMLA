# SM100 Sparse Prefill `head64` / `head128` Q-TMEM And P-Path Deep Dive

This note organizes a code-reading analysis of the two sparse prefill forward kernels:

- `csrc/sm100/prefill/sparse/fwd/head64`
- `csrc/sm100/prefill/sparse/fwd/head128`

The focus is on two questions:

1. Why can `head64` place the full `Q[64,576]` into TMEM using only 144 TMEM columns, while `head128` uses 192 TMEM columns for only part of `Q`?
2. What are the differences between `head64` and `head128` when computing `P = QK^T`, and when loading `P` for softmax?

Throughout this note:

- "logical Q dimension" means the bf16 feature dimension in the model sense
- "TMEM columns" means the TMEM address-space columns used by the kernel
- "TS" means A from TMEM, B from SMEM
- "SS" means A from SMEM, B from SMEM

## Executive Summary

The shortest useful summary is:

- `head64` uses a 1-CTA implicit dual-gemm view for `P`, so both `Q` and `K` are repacked into a dual-gemm layout before the QK MMA. Under that layout, the full logical `Q[64,576]` becomes only 144 TMEM columns.
- `head128` does not use that compression for `P`. It keeps `P` at its real width in a 2-CTA cluster MMA, so after reserving TMEM for `O` and `P`, only 192 TMEM columns remain for `Q`. Those 192 columns correspond to `tQ = 384` bf16 dimensions, while the remaining `sQ` stays in SMEM.
- As a result, `head64` computes `P` entirely through TS MMA, while `head128` computes `P` in two stages: `SS(sQ x sK)` then `TS(tQ x sK)`.
- `head64` also has a more complicated `load p` path because dual gemm leaves `P` as two physical pieces that must be loaded and reduced before softmax. `head128` can load final `P` directly from TMEM.

## Relevant Code Map

### `head64`

- TMEM layout and shared-memory plan:
  - `csrc/sm100/prefill/sparse/fwd/head64/config.h`
- Kernel body:
  - `csrc/sm100/prefill/sparse/fwd/head64/phase1.cuh`
- Shared helper for `load p` and reduction:
  - `csrc/sm100/prefill/sparse/common_subroutine.h`

### `head128`

- TMEM layout and shared-memory plan:
  - `csrc/sm100/prefill/sparse/fwd/head128/config.h`
- Kernel body:
  - `csrc/sm100/prefill/sparse/fwd/head128/phase1.cuh`

### Shared MMA atom definitions

- `csrc/kerutils/include/kerutils/device/sm100/gemm.cuh`

Important pieces there:

- `SM100_MMA_F16BF16_WS_TS_NOELECT`: 1-CTA WS TS MMA
- `SM100_MMA_F16BF16_2x1SM_TS_NOELECT`: 2-CTA TS MMA
- `SM100_MMA_F16BF16_2x1SM_SS_NOELECT`: 2-CTA SS MMA

## Section 1: Why `head64` Can Store Full `Q` In 144 TMEM Columns

## 1.1 The Static TMEM Allocation

In `head64`, the TMEM column layout is:

```cpp
namespace tmem_cols {
    //   0 ~ 256: output
    // 256 ~ 400: Q
    // 400 ~ 464: P
    constexpr int O = 0;
    constexpr int Q = 256;
    constexpr int Q_RoPE = 256 + 128;
    constexpr int P = 400;
}
```

Source:

- `csrc/sm100/prefill/sparse/fwd/head64/config.h:39-47`

So `Q` occupies TMEM columns `[256, 400)`, which is exactly 144 columns.

At the model level:

- `D_Q = 576`
- `D_V = 512`
- `D_Q - D_V = 64`

Source:

- `csrc/sm100/prefill/sparse/fwd/head64/config.h:26-27`

So logical `Q` is split into:

- NoPE part: 512 dims
- RoPE part: 64 dims

The question is therefore:

Why do logical `512 + 64 = 576` bf16 dimensions become TMEM `128 + 16 = 144` columns?

## 1.2 The Root Cause Is Implicit Dual GEMM

`head64` defines the `P` MMA as:

```cpp
using TiledMMA_P = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_TS_NOELECT<bf16, bf16, float, B_H, 128, UMMA::Major::K, UMMA::Major::K>{}
));
```

with a comment:

```cpp
// Here we use N = 128 = 2*B_TOPK since we're going to use implicit dual gemm
```

Source:

- `csrc/sm100/prefill/sparse/fwd/head64/config.h:141-143`

This is the key design choice.

The kernel's real top-k block width is:

- `B_TOPK = 64`

but the `P` MMA is deliberately constructed with:

- `N = 128 = 2 * B_TOPK`

That means the QK MMA is not laid out as a plain `64 x 64` attention block. Instead, it is built in a dual-gemm physical view. Once that happens, the `Q` and `K` operands are also viewed through a packed dual-gemm layout.

This is why `head64` can compress the full logical Q into fewer TMEM columns than `head128`.

## 1.3 The Dual-GEMM View Is Visible In The K Layouts

The corresponding `K` layouts in `head64` are:

```cpp
using SmemLayoutKNoPE_TiledMMA = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_TOPK*2>, Int<D_V/2>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));   // Re-view K-NoPE as B_TOPK*2 x D_V/2 for dual gemm

using SmemLayoutKRoPE_TiledMMA = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_TOPK*2>, Int<64/2>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));
```

Source:

- `csrc/sm100/prefill/sparse/fwd/head64/config.h:90-102`

This is an important clue:

- NoPE is re-viewed from logical `64 x 512` into dual-gemm `128 x 256`
- RoPE is re-viewed from logical `64 x 64` into dual-gemm `128 x 32`

So before TMEM column counting even starts, the Q/K operand seen by the MMA has already become:

- NoPE: 256 dual-gemm K units
- RoPE: 32 dual-gemm K units

Together that is:

- `256 + 32 = 288` bf16-sized K units in the dual-gemm view

TMEM columns are then counted in 32-bit units, so two bf16 elements share one TMEM column. Therefore:

- NoPE: `256 / 2 = 128` TMEM columns
- RoPE: `32 / 2 = 16` TMEM columns
- Total: `128 + 16 = 144` TMEM columns

This matches the static TMEM layout exactly.

## 1.4 The UTCCP Copy Path Confirms The 128 + 16 Split

The kernel binds the TMEM Q fragments like this:

```cpp
Tensor tQ_nope_part0 = ...
Tensor tQ_nope_part1 = ...
Tensor tQ_rope = ...

tQ_nope_part0.data().get() = tmem_cols::Q;
tQ_nope_part1.data().get() = tmem_cols::Q + 64;
tQ_rope.data().get() = tmem_cols::Q_RoPE;
```

Source:

- `csrc/sm100/prefill/sparse/fwd/head64/phase1.cuh:84-97`

That is already enough to see the final split:

- first NoPE half: 64 columns
- second NoPE half: 64 columns
- RoPE: 16 columns

The actual `SMEM -> TMEM` copy path makes the same split explicit.

### RoPE copy

```cpp
for (int subtile_idx = 0; subtile_idx < 2; ++subtile_idx) {
    SM100_UTCCP_128dp256bit_1cta::copy(
        sQ_rope_desc + (subtile_idx*32) / 16,
        tmem_cols::Q_RoPE + subtile_idx*8
    );
}
```

Source:

- `csrc/sm100/prefill/sparse/fwd/head64/phase1.cuh:436-449`

So RoPE uses:

- 2 subtiles
- each subtile advances TMEM by 8 columns
- total = `2 * 8 = 16` TMEM columns

The nearby comment says:

- UTCCP view: `128 rows x 32 cols`
- kernel's logical view: `64 rows x 64 cols`

So logical 64 RoPE dims become 16 TMEM columns in this dual-gemm layout.

### NoPE copy

```cpp
for (int tile_idx = 0; tile_idx < D_V/64/2; ++tile_idx) {
    for (int subtile_idx = 0; subtile_idx < 4; ++subtile_idx) {
        SM100_UTCCP_128dp256bit_1cta::copy(
            sQ_nope_desc + (tile_idx*(B_H*128*2) + subtile_idx*32) / 16,
            tmem_cols::Q + tile_idx*32 + subtile_idx*8
        );
    }
}
```

Source:

- `csrc/sm100/prefill/sparse/fwd/head64/phase1.cuh:455-467`

Since:

- `D_V = 512`
- `D_V/64/2 = 4`

there are 4 NoPE tiles.

Each tile contributes:

- 4 subtiles
- each subtile advances TMEM by 8 columns
- total = `4 * 8 = 32` columns per tile

So NoPE uses:

- `4 * 32 = 128` TMEM columns

Again this matches the final split exactly.

## 1.5 Why This Is Different From A Plain `dim / 2` Mapping

It is important not to mentally reduce `head64` to a plain "bf16 packed 2-to-1 into TMEM" rule.

If that were the whole story:

- logical 576 bf16 dims would become `576 / 2 = 288` TMEM columns

But the real answer is 144 columns.

The missing factor of 2 comes from implicit dual gemm:

- first, logical dims are folded into the dual-gemm K view
- then TMEM columns count 32-bit quantities

This extra factor is exactly why `head64` can fit full Q into TMEM while `head128` cannot.

## Section 2: Why `head128` Uses 192 TMEM Columns For Only Part Of `Q`

## 2.1 The Static TMEM Allocation

`head128` defines:

```cpp
static constexpr int D_tQ = 384, NUM_tQ_TILES = D_tQ / 64;
static constexpr int D_sQ = D_QK - D_tQ, NUM_sQ_TILES = D_sQ / 64;

struct tmem_cols {
    //   0 ~ 256: output
    // 256 ~ 320: P
    // 320 ~ 512: Q[D_QK-D_tQ:]
    static constexpr int o = 0;
    static constexpr int p = 256;
    static constexpr int q = 512 - D_tQ/2;
    static_assert(p+64 <= q);
};
```

Source:

- `csrc/sm100/prefill/sparse/fwd/head128/config.h:39-52`

For `D_QK = 576`:

- `D_tQ = 384`
- `D_sQ = 192`
- `q = 512 - 384/2 = 320`

So TMEM is divided as:

- `O`: 256 columns
- `P`: 64 columns
- `Q`: 192 columns

This already tells us the fundamental budget constraint:

- full `Q[576]` would require `576 / 2 = 288` TMEM columns
- but after reserving `O` and `P`, only 192 columns remain

So `head128` cannot store full Q in TMEM even before considering finer layout details.

## 2.2 `head128` Is A 2-CTA Cluster Kernel

Unlike `head64`, the `head128` kernel launches in clusters:

```cpp
cutlass::ClusterLaunchParams launch_params = {
    dim3(2*params.s_q, 1, 1),
    dim3(Kernel::NUM_THREADS, 1, 1),
    dim3(2, 1, 1),
    smem_size,
    params.stream
};
```

Source:

- `csrc/sm100/prefill/sparse/fwd/head128/phase1.cuh:768-777`

Inside the kernel:

```cpp
const int cta_idx = blockIdx.x % 2;
const int s_q_idx = blockIdx.x / 2;
```

Source:

- `csrc/sm100/prefill/sparse/fwd/head128/phase1.cuh:70-71`

And TMEM uses:

```cpp
cute::TMEM::Allocator2Sm().allocate(512, plan.tmem_start_addr.data());
```

Source:

- `csrc/sm100/prefill/sparse/fwd/head128/phase1.cuh:146-149`

So `head128` is built on 2-CTA cluster MMA primitives, not the 1-CTA dual-gemm trick used by `head64`.

This matters because the `P` MMA is not widened to `2 * B_TOPK` the way `head64` does it.

## 2.3 `head128` Uses Real-Width `P`, Not Dual-GEMM-Width `P`

The `P` MMA definitions are:

```cpp
using TiledMMA_P_tQ = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_TS_NOELECT<bf16, bf16, float, B_H, B_TOPK, UMMA::Major::K, UMMA::Major::K>{}
));

using TiledMMA_P_sQ = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS_NOELECT<bf16, bf16, float, B_H, B_TOPK, UMMA::Major::K, UMMA::Major::K>{}
));
```

Source:

- `csrc/sm100/prefill/sparse/fwd/head128/config.h:124-130`

The key point is:

- `N = B_TOPK`

not:

- `N = 2 * B_TOPK`

So `head128` does not use an implicit dual-gemm P layout. Because that extra compression is absent, TMEM column counting for `Q` is the straightforward bf16-to-32b mapping:

- 64 logical bf16 dims -> 32 TMEM columns
- 384 logical bf16 dims -> 192 TMEM columns

## 2.4 The UTCCP Path Confirms That `192` Columns Mean `384` Q Dims

The TMEM Q fragment is:

```cpp
Tensor tQr = tiled_mma_P_tQ.get_slice(_0{}).make_fragment_A(
    partition_shape_A(tiled_mma_P_tQ, Shape<Int<B_H/2>, Int<D_tQ>>{})
);
tQr.data().get() = tmem_cols::q;
```

Source:

- `csrc/sm100/prefill/sparse/fwd/head128/phase1.cuh:101-108`

The `SMEM -> TMEM` copy path is:

```cpp
UMMA::SmemDescriptor sQ_desc = UMMA::make_umma_desc<UMMA::Major::K>(
    make_tensor(
        make_smem_ptr(plan.u.q_full.data() + (B_H/2)*D_sQ),
        tile_to_shape(
            UMMA::Layout_K_SW128_Atom<bf16>{},
            Shape<Int<B_H/2>, Int<64>>{}
        )
    )
);

for (int tile_idx = 0; tile_idx < NUM_tQ_TILES; ++tile_idx) {
    for (int subtile_idx = 0; subtile_idx < 8; ++subtile_idx) {
        SM100_UTCCP_2x64dp128bitlw0213_2cta::copy(
            sQ_desc + tile_idx*((B_H/2)*128/16) + subtile_idx*(16/16),
            tmem_cols::q + tile_idx*32 + subtile_idx*4
        );
    }
}
```

Source:

- `csrc/sm100/prefill/sparse/fwd/head128/phase1.cuh:555-579`

Important observations:

- the source pointer starts at `plan.u.q_full.data() + (B_H/2)*D_sQ`
- so the copied region is the suffix of Q, not the full Q
- this copied suffix is exactly `tQ`

Since:

- `NUM_tQ_TILES = D_tQ / 64 = 6`

and each tile contributes:

- 8 subtiles
- each subtile advances TMEM by 4 columns
- total = `8 * 4 = 32` columns per tile

therefore:

- `6 * 32 = 192` TMEM columns

So `192` TMEM columns in `head128` correspond exactly to:

- `6` Q tiles
- each tile is `64` logical bf16 dimensions
- total = `384` logical Q dimensions

That is why the kernel stores only `tQ`, not full Q.

## 2.5 Where The Rest Of Q Lives

In `head128`, shared memory is organized so that full Q is first loaded into `q_full`, then split logically into:

- `sq`: shared-memory-resident Q prefix
- `tQ`: TMEM-resident Q suffix

The shared-memory plan is:

```cpp
union {
    array_aligned<bf16, cosize_v<SmemLayoutQTiles<D_Q/64>>> q_full;
    struct {
        array_aligned<bf16, cosize_v<SmemLayoutQTiles<NUM_sQ_TILES>>> sq;
        array_aligned<bf16, cosize_v<SmemLayoutV>> v;
        array_aligned<bf16, cosize_v<SmemLayoutKTiles<D_K/64>>> k;
    } s;
    array_aligned<bf16, cosize_v<SmemLayoutO>> o;
} u;
```

Source:

- `csrc/sm100/prefill/sparse/fwd/head128/config.h:97-105`

This says:

- full Q is initially in `q_full`
- `sq` is the prefix that stays in SMEM
- after `tQ` has been copied into TMEM, the tail of `q_full` can be reused by `v`

The kernel even waits on the Q UTCCP completion before reusing the memory for V:

- `csrc/sm100/prefill/sparse/fwd/head128/phase1.cuh:513-516`

So `head128` is not "forgetting" the remaining Q. It intentionally keeps the `sQ` part in SMEM and feeds it into a separate SS MMA path.

## Section 3: How `head64` Computes `P`

## 3.1 The P MMA Is Entirely TS

In `head64`, the relevant fragments are:

```cpp
Tensor tP = partition_fragment_C(tiled_mma_P, Shape<Int<B_H>, _128>{});
Tensor tQ_nope_part0 = ...
Tensor tQ_nope_part1 = ...
Tensor tQ_rope = ...
```

Source:

- `csrc/sm100/prefill/sparse/fwd/head64/phase1.cuh:83-91`

The actual QK loop is:

```cpp
if constexpr (HAVE_ROPE) {
    plan.bar_kv_rope_ready.wait(k&1);
    ku::tcgen05_after_thread_sync();
    ku::utcmma_ts(tiled_mma_P, tQ_rope, sK_rope, tP, true);
    ku::umma_arrive_noelect(plan.bar_qk_rope_done);
}

Tensor sK_nope_divided = flat_divide(sK_nope, Tile<Int<B_TOPK*2>, Int<D_V/4>>{})(_, _, _0{}, _);
for (int kv_nope_part_idx = 0; kv_nope_part_idx < 2; ++kv_nope_part_idx) {
    plan.bar_kv_nope_ready[cur_buf][kv_nope_part_idx].arrive_and_expect_tx(B_TOPK*D_V/2*sizeof(bf16));
    plan.bar_kv_nope_ready[cur_buf][kv_nope_part_idx].wait((k/NUM_BUFS)&1);
    ku::tcgen05_after_thread_sync();

    bool clear_accum = (!HAVE_ROPE) && kv_nope_part_idx == 0;
    ku::utcmma_ts(tiled_mma_P, kv_nope_part_idx ? tQ_nope_part1 : tQ_nope_part0, sK_nope_divided(_, _, kv_nope_part_idx), tP, clear_accum);
}
ku::umma_arrive_noelect(plan.bar_qk_nope_done[cur_buf]);
```

Source:

- `csrc/sm100/prefill/sparse/fwd/head64/phase1.cuh:485-508`

So the full `P = QK^T` path is:

1. `Q_rope @ K_rope^T` by TS MMA, clearing accumulators if RoPE exists
2. `Q_nope_part0 @ K_nope_part0^T` by TS MMA
3. `Q_nope_part1 @ K_nope_part1^T` by TS MMA

Everything is TS:

- A operand from TMEM
- B operand from SMEM
- C accumulator/result in TMEM

There is no `sQ` branch and no SS QK MMA in `head64`.

## 3.2 Why `P` Becomes Two Physical Pieces

Because `head64` widens the QK MMA to `N = 128 = 2 * B_TOPK`, the physical `P` stored in TMEM is not yet the final logical attention-score matrix.

This is documented directly in the common helper:

```cpp
Initially, since dual gemm is used, we have two P pieces in Tensor Memory,
one occupying rows 0 ~ 63 while the other occupying rows 64 ~ 127.
We'd like to have them reduced into one single P piece
```

Source:

- `csrc/sm100/prefill/sparse/common_subroutine.h:47-66`

So `head64` pays for its compact full-Q TMEM residency by producing a non-final `P` layout that later has to be reduced.

## Section 4: How `head128` Computes `P`

## 4.1 The P MMA Is Split Into SS And TS

`head128` uses two distinct MMA atoms:

- `TiledMMA_P_sQ`: SS
- `TiledMMA_P_tQ`: TS

Source:

- `csrc/sm100/prefill/sparse/fwd/head128/config.h:124-130`

The K producer loads K in two parts:

```cpp
load_part_ki(plan.bar_k_part0_ready[cur_buf], 0, D_sQ/64);
...
load_part_ki(plan.bar_k_part1_ready[cur_buf], D_sQ/64, D_K/64);
```

Source:

- `csrc/sm100/prefill/sparse/fwd/head128/phase1.cuh:467-503`

Then the QK loop is:

```cpp
Tensor sQl = make_tensor(make_smem_ptr(plan.u.s.sq.data()), SmemLayoutQTiles<NUM_sQ_TILES>{});
Tensor sKl = make_tensor(make_smem_ptr(plan.u.s.k.data()), SmemLayoutKTiles<NUM_sQ_TILES>{});
Tensor sKr = make_tensor(make_smem_ptr(plan.u.s.k.data()+64*D_sQ), SmemLayoutKTiles<NUM_tQ_TILES>{});

plan.bar_k_part0_ready[cur_buf].arrive_and_expect_tx(B_TOPK*D_sQ*sizeof(bf16));
plan.bar_k_part0_ready[cur_buf].wait((k/NUM_BUFS)&1);
...
ku::utcmma_ss(tiled_mma_P_sQ, sQl, sKl, tP, true);
ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_qk_part_done[cur_buf], 1|2);

plan.bar_k_part1_ready[cur_buf].arrive_and_expect_tx(B_TOPK*(D_K-D_sQ)*sizeof(bf16));
plan.bar_k_part1_ready[cur_buf].wait((k/NUM_BUFS)&1);
...
ku::utcmma_ts(tiled_mma_P_tQ, tQr, sKr, tP, false);
ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_qk_done[cur_buf], 1|2);
```

Source:

- `csrc/sm100/prefill/sparse/fwd/head128/phase1.cuh:586-607`

So `head128` computes `P` in two stages:

1. `sQ x sK(part0)` with SS MMA and `clear_accum = true`
2. `tQ x sK(part1)` with TS MMA and `clear_accum = false`

This is the direct consequence of only part of Q fitting in TMEM.

## 4.2 Why `head128` Does Not Need P Reduction

Because `head128` does not widen `P` to `2 * B_TOPK`, its `P` accumulator in TMEM already matches the logical attention tile for that CTA's rows.

So unlike `head64`, it does not produce "dual-gemm twin pieces" that later need to be added together.

This becomes very visible in the `load p` step.

## Section 5: The `load p` Difference

## 5.1 `head64`: Load, Mask, Exchange, Reduce

In `head64`, the softmax consumer path does:

```cpp
retrieve_mask_and_reduce_p<
    NUM_ELEMS_PER_THREAD,
    tmem_cols::P,
    NamedBarriers::wg0_warp02_sync,
    NamedBarriers::wg0_warp13_sync,
    false
>(
    plan.is_k_valid[k%NUM_BUFS],
    warp_idx, lane_idx,
    [&]() {plan.bar_p_free.arrive();},
    plan.p_exchange_buf,
    p
);
```

Source:

- `csrc/sm100/prefill/sparse/fwd/head64/phase1.cuh:177-191`

The helper does:

```cpp
if (local_warp_idx < 2) {
    ku::tmem_ld_32dp32bNx<NUM_ELEMS_PER_THREAD>(TMEM_COL_START, p);
    ku::tmem_ld_32dp32bNx<NUM_ELEMS_PER_THREAD>(TMEM_COL_START + NUM_ELEMS_PER_THREAD, p_peer);
} else {
    ku::tmem_ld_32dp32bNx<NUM_ELEMS_PER_THREAD>(TMEM_COL_START, p_peer);
    ku::tmem_ld_32dp32bNx<NUM_ELEMS_PER_THREAD>(TMEM_COL_START + NUM_ELEMS_PER_THREAD, p);
}

// mask invalid tokens

// store p_peer into p_exchange_buf
// synchronize warp 0<->2 and 1<->3
// load peer fragment from p_exchange_buf
// add it into p
```

Source:

- `csrc/sm100/prefill/sparse/common_subroutine.h:87-136`

So `head64`'s `load p` includes all of the following:

1. load one physical `P` piece into `p`
2. load the peer physical `P` piece into `p_peer`
3. mask invalid tokens
4. exchange fragments through `p_exchange_buf`
5. reduce the two physical pieces into final logical `P`

Only after that does it proceed to rowwise max, online softmax, and `S` generation.

This complexity is not incidental. It is the price of using implicit dual gemm to fit full Q into TMEM.

## 5.2 `head128`: Direct Load Of Final P

In `head128`, after `bar_qk_done`, the scale/exp warpgroup simply does:

```cpp
float2 p[(B_TOPK/2)/2];
ku::tmem_ld_32dp32bNx<B_TOPK/2>(tmem_cols::p, p);
cutlass::arch::fence_view_async_tmem_load();
ku::tcgen05_before_thread_sync();
plan.bar_p_free[k%NUM_BUFS].arrive(0u);
```

Source:

- `csrc/sm100/prefill/sparse/fwd/head128/phase1.cuh:179-184`

Then it:

1. masks invalid tokens
2. computes row max / rescaling / exp / sumexp
3. writes `S` into SMEM

Source:

- `csrc/sm100/prefill/sparse/fwd/head128/phase1.cuh:186-307`

There is no peer exchange buffer and no reduction of dual-gemm fragments.

That is because `head128`'s `P` in TMEM is already in final logical form for the CTA's rows.

## 5.3 Operationally, The Difference Is:

- `head64`: `load p` means "materialize logical P from dual-gemm physical P"
- `head128`: `load p` means "read final P directly from TMEM"

This is the single most important consumer-side difference between the two kernels.

## Section 6: Side-By-Side Comparison

| Topic | `head64` | `head128` |
| --- | --- | --- |
| Launch structure | 1 CTA per query block | 2-CTA cluster per query block |
| TMEM allocator | `Allocator1Sm` | `Allocator2Sm` |
| Q in TMEM | Full `Q[64,576]` | Only `tQ = 384` dims |
| Why Q fits / does not fit | Implicit dual gemm compresses operand view | No dual-gemm compression for `P`; only plain bf16-to-32b packing |
| Q TMEM usage | `128` NoPE + `16` RoPE = `144` columns | `384 / 2 = 192` columns |
| QK MMA form | TS only | SS for `sQ`, then TS for `tQ` |
| P physical layout | Dual-gemm twin pieces | Final logical tile |
| `load p` | load + peer exchange + reduction | direct TMEM load |
| Main reason `load p` is complex | `P` is widened physically during production | `P` is not widened physically |

## Section 7: Final Answers To The Two Original Questions

## Question 1

Why can `head64` put full `Q` into 144 TMEM columns, while `head128` uses 192 TMEM columns for only part of `Q`?

Answer:

- `head64` does not store Q in a plain logical-dimension layout. It stores Q in the operand layout required by a 1-CTA implicit dual-gemm `P` MMA with `N = 2 * B_TOPK`. Under that layout, logical `576` dims become dual-gemm `288` bf16 K units, which become `144` TMEM columns.
- `head128` does not use that dual-gemm compression. It uses a 2-CTA cluster MMA with real `N = B_TOPK`, so Q follows the ordinary bf16-to-32b packing rule: `384` dims become `192` TMEM columns. After reserving TMEM for `O` and `P`, only those 192 columns are available, so only `tQ` can go into TMEM.

## Question 2

What is the difference between `head64` and `head128` in `P` MMA and in `load p`?

Answer:

- `head64` computes `P` entirely through TS MMA:
  - `Q_rope @ K_rope^T`
  - `Q_nope_part0 @ K_nope_part0^T`
  - `Q_nope_part1 @ K_nope_part1^T`
  All three accumulate into a dual-gemm-shaped physical `P`.
- `head128` computes `P` in two stages:
  - `SS`: `sQ @ sK(part0)^T`
  - `TS`: `tQ @ sK(part1)^T`
  The result is already the final logical `P`.
- Therefore, `head64` must reduce physical `P` pieces during `load p`, while `head128` can directly load final `P` from TMEM.

## Section 8: Code References

### `head64`

- TMEM layout:
  - `csrc/sm100/prefill/sparse/fwd/head64/config.h:39-47`
- Dual-gemm `P` MMA shape:
  - `csrc/sm100/prefill/sparse/fwd/head64/config.h:141-143`
- Dual-gemm K layouts:
  - `csrc/sm100/prefill/sparse/fwd/head64/config.h:90-102`
- TMEM fragment bindings for Q:
  - `csrc/sm100/prefill/sparse/fwd/head64/phase1.cuh:83-97`
- Q `SMEM -> TMEM` UTCCP:
  - `csrc/sm100/prefill/sparse/fwd/head64/phase1.cuh:436-467`
- QK MMA:
  - `csrc/sm100/prefill/sparse/fwd/head64/phase1.cuh:485-508`
- `load p` callsite:
  - `csrc/sm100/prefill/sparse/fwd/head64/phase1.cuh:177-191`
- `load p` helper:
  - `csrc/sm100/prefill/sparse/common_subroutine.h:47-143`

### `head128`

- TMEM layout and `tQ/sQ` split:
  - `csrc/sm100/prefill/sparse/fwd/head128/config.h:39-52`
- shared-memory reuse plan:
  - `csrc/sm100/prefill/sparse/fwd/head128/config.h:97-105`
- Q fragment bound to TMEM:
  - `csrc/sm100/prefill/sparse/fwd/head128/phase1.cuh:101-108`
- 2-CTA launch:
  - `csrc/sm100/prefill/sparse/fwd/head128/phase1.cuh:70-71`
  - `csrc/sm100/prefill/sparse/fwd/head128/phase1.cuh:768-777`
- `tQ` `SMEM -> TMEM` UTCCP:
  - `csrc/sm100/prefill/sparse/fwd/head128/phase1.cuh:555-579`
- K producer split:
  - `csrc/sm100/prefill/sparse/fwd/head128/phase1.cuh:467-503`
- QK MMA:
  - `csrc/sm100/prefill/sparse/fwd/head128/phase1.cuh:586-607`
- direct `load p`:
  - `csrc/sm100/prefill/sparse/fwd/head128/phase1.cuh:179-184`

### Shared MMA atom definitions

- 1-CTA WS TS MMA:
  - `csrc/kerutils/include/kerutils/device/sm100/gemm.cuh:16-108`
- 2-CTA TS MMA:
  - `csrc/kerutils/include/kerutils/device/sm100/gemm.cuh:211-312`
- 2-CTA SS MMA:
  - `csrc/kerutils/include/kerutils/device/sm100/gemm.cuh:316-418`

## Section 9: A Practical Mental Model

When reading these kernels later, the cleanest mental model is:

- `head64` spends complexity on `P` layout so that it can keep all of `Q` in TMEM
- `head128` spends complexity on split Q paths so that it can keep `P` simple

Equivalently:

- `head64`: compact `Q`, complicated `P`
- `head128`: split `Q`, straightforward `P`

That tradeoff explains almost every visible difference between the two implementations.
