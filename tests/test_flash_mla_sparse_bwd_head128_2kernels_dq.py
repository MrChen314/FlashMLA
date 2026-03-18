import pytest
import torch
import kernelkit as kk

import flash_mla


def calc_diff(a: torch.Tensor, b: torch.Tensor):
    abs_diff = torch.abs(a - b)
    max_diff = abs_diff.max().item()
    rel_diff = (abs_diff / (1e-4 + torch.abs(a))).mean().item()
    return max_diff, rel_diff


def ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, is_casual=True, q_start_index_s=0):
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)
    b, sq, h, dim_q = q.shape
    b, sk, g, _ = kv.shape

    assert kv.shape[-1] == 576, "you should assign dim otherwise"
    dim = 512
    k = kv
    v = kv[..., :dim]

    b, _, _, dim_v = v.shape
    g_index = g
    h_index = h // g
    compressed_casual_mask = torch.arange(q_start_index_s, q_start_index_s + sq, dtype=torch.int32, device="cuda").view(-1, 1) >= torch.arange(
        1 - 1, sk * 1, 1, dtype=torch.int32, device="cuda"
    ).view(1, -1)

    mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
    mask = mask[..., :-1]
    mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
    mask[:, :, : 1 - 1, 0] = True
    mask = mask.view(b, g_index, 1, sq, sk)

    q = q.view(b, sq, g, -1, dim_q)
    score = torch.einsum("bmghd,bngd->bghmn", q, k)
    sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
    p = score.softmax(dim=-1)
    p = p.view(b, g_index, h_index, -1, sq, sk)
    p = p.view(b, g, -1, sq, sk)
    o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
    o = o.reshape(b, sq, h, dim_v)
    return o.to(torch.bfloat16)


def ref_sparse_mla_bwd_interface(q, kv, o, do, indices, lse, sm_scale=None, is_casual=True, q_start_index_s=0):
    q = q.detach().clone()
    kv = kv.detach().clone()
    q.requires_grad = True
    kv.requires_grad = True
    o = ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale, is_casual, q_start_index_s)
    o.backward(do)
    return q.grad


def ref_sparse_mla_sd_interface(q, kv, do, indices, sm_scale=None, d_v=512, q_start_index_s=0, q_chunk_size=32):
    q = q.float()
    kv = kv.float()
    do = do.float()

    b, sq, h, dim_q = q.shape
    _, skv, g, dim_kv = kv.shape
    topk = indices.shape[-1]

    assert h % g == 0, "h_q must be divisible by h_kv"
    assert dim_kv >= d_v, "kv head dim must be >= value dim"

    grouped_heads = h // g
    sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale

    q = q.view(b, sq, g, grouped_heads, dim_q)
    do = do.view(b, sq, g, grouped_heads, d_v)

    batch_idx = torch.arange(b, device=q.device).view(b, 1, 1, 1)
    group_idx = torch.arange(g, device=q.device).view(1, 1, g, 1)

    s_chunks = []
    ds_chunks = []
    for q_begin in range(0, sq, q_chunk_size):
        q_end = min(q_begin + q_chunk_size, sq)
        q_chunk = q[:, q_begin:q_end]
        do_chunk = do[:, q_begin:q_end]
        indices_chunk = indices[:, q_begin:q_end]

        safe_indices = indices_chunk.long().clamp(min=0, max=max(skv - 1, 0))
        gathered_kv = kv[batch_idx, safe_indices, group_idx]
        gathered_v = gathered_kv[..., :d_v]

        logits = torch.einsum("bnghd,bngtd->bnght", q_chunk, gathered_kv).mul(sm_scale)
        dp = torch.einsum("bnghd,bngtd->bnght", do_chunk, gathered_v)

        causal_limit = torch.arange(
            q_start_index_s + q_begin,
            q_start_index_s + q_end,
            dtype=indices.dtype,
            device=indices.device,
        ).view(1, q_end - q_begin, 1, 1)
        valid_mask = (indices_chunk >= 0) & (indices_chunk < skv) & (indices_chunk <= causal_limit)
        valid_mask = valid_mask.unsqueeze(3).expand(-1, -1, -1, grouped_heads, -1)

        logits = logits.masked_fill(~valid_mask, float("-inf"))
        lse = torch.logsumexp(logits, dim=-1)
        lonely_q_mask = lse == float("-inf")
        lse_for_prob = lse.clone()
        lse_for_prob[lonely_q_mask] = float("+inf")

        s_chunk = torch.exp(logits - lse_for_prob.unsqueeze(-1))
        o_chunk = torch.einsum("bnght,bngtd->bnghd", s_chunk.type(gathered_v.dtype), gathered_v)
        delta = (o_chunk * do_chunk).sum(dim=-1, keepdim=True)
        ds_chunk = s_chunk * (dp - delta) * sm_scale

        s_chunks.append(s_chunk.reshape(b, q_end - q_begin, h, topk).to(torch.bfloat16))
        ds_chunks.append(ds_chunk.reshape(b, q_end - q_begin, h, topk).to(torch.bfloat16))

    return torch.cat(s_chunks, dim=1), torch.cat(ds_chunks, dim=1)


def test_sparse_mla_bwd_head128_2kernels_dq(
    B=1,
    S=64,
    SKV=256,
    H=128,
    HKV=1,
    DQKV=576,
    DV=512,
    topk=64,
    sm_scale=None,
    dtype=torch.bfloat16,
    check_correctness=True,
    q_start_index_s=64,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    cc_major, _ = torch.cuda.get_device_capability()
    if cc_major != 10:
        pytest.skip("head128_2kernels dq path currently targets SM100 only")

    sm_scale = DQKV ** -0.5 if sm_scale is None else sm_scale

    q = torch.randn((B, S, H, DQKV), dtype=dtype, device="cuda").requires_grad_(True)
    kv = torch.randn((B, SKV, HKV, DQKV), dtype=dtype, device="cuda").requires_grad_(True)
    do = torch.randn((B, S, H, DV), dtype=dtype, device="cuda")

    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                max_kv_i = min(SKV, max(1, q_start_index_s + t))
                i_i = torch.randperm(max_kv_i)[:topk]
                indices[b, t, h, : len(i_i)] = i_i

    flash_out, _, flash_lse, _ = flash_mla.flash_mla_sparse_fwd(
        q.squeeze(0).contiguous(), kv.squeeze(0).contiguous(), indices.squeeze(0).contiguous(),
        sm_scale=sm_scale, q_start_index_s=q_start_index_s
    )

    q_3d = q.squeeze(0)
    kv_3d = kv.squeeze(0)
    do_3d = do.squeeze(0)
    indices_3d = indices.squeeze(0)
    flash_dq, flash_s, flash_ds = flash_mla.flash_mla_sparse_bwd_head128_2kernels_dq(
        q_3d, kv_3d, flash_out, do_3d, indices_3d, flash_lse,
        sm_scale=sm_scale,
        q_start_index_s=q_start_index_s,
    )
    torch.cuda.synchronize()

    if check_correctness:
        ref_dq = ref_sparse_mla_bwd_interface(
            q, kv, None, do, indices, None, sm_scale=sm_scale, q_start_index_s=q_start_index_s
        )
        ref_s, ref_ds = ref_sparse_mla_sd_interface(
            q, kv, do, indices, sm_scale=sm_scale, d_v=DV, q_start_index_s=q_start_index_s
        )
        ref_dq_3d = ref_dq.squeeze(0)
        ref_s_3d = ref_s.squeeze(0)
        ref_ds_3d = ref_ds.squeeze(0)
        flash_dq_max_diff, flash_dq_rel_diff = calc_diff(flash_dq, ref_dq_3d)
        flash_s_max_diff, flash_s_rel_diff = calc_diff(flash_s, ref_s_3d)
        flash_ds_max_diff, flash_ds_rel_diff = calc_diff(flash_ds, ref_ds_3d)
        print(f"[ref vs flash] dQ  max_diff={flash_dq_max_diff:.6f}, rel_diff={flash_dq_rel_diff:.6f}")
        print(f"[ref vs flash] s   max_diff={flash_s_max_diff:.6f}, rel_diff={flash_s_rel_diff:.6f}")
        print(f"[ref vs flash] ds  max_diff={flash_ds_max_diff:.6f}, rel_diff={flash_ds_rel_diff:.6f}")

        assert kk.check_is_allclose(
            "dq",
            flash_dq.float(),
            ref_dq_3d.float(),
            abs_tol=1e-3,
            rel_tol=8.01 / 128,
            cos_diff_tol=7e-6,
        )
        assert kk.check_is_allclose(
            "s",
            flash_s.float(),
            ref_s_3d.float(),
            abs_tol=1e-3,
            rel_tol=8.01 / 128,
            cos_diff_tol=7e-6,
        )
        assert kk.check_is_allclose(
            "ds",
            flash_ds.float(),
            ref_ds_3d.float(),
            abs_tol=1e-3,
            rel_tol=8.01 / 128,
            cos_diff_tol=7e-6,
        )

    per_token_flop = 2 * sum(
        [
            H * DQKV * topk,
            H * DV * topk,
            H * DQKV * topk,
        ]
    )

    def fn():
        return flash_mla.flash_mla_sparse_bwd_head128_2kernels_dq(
            q_3d, kv_3d, flash_out, do_3d, indices_3d, flash_lse,
            sm_scale=sm_scale,
            q_start_index_s=q_start_index_s,
        )

    bench_result = kk.bench_kineto(fn, num_tests=100)
    if kk.is_using_profiling_tools():
        bwd_time_s = 1
    else:
        kernel_names = bench_result.get_kernel_names()
        has_bwd_kernel = any("dq_phase_kernel" in name for name in kernel_names)
        has_preprocess_kernel = any("preprocess_delta" in name for name in kernel_names)

        if has_bwd_kernel and has_preprocess_kernel:
            bwd_time_s = bench_result.get_e2e_time("preprocess_delta", "dq_phase_kernel")
        elif has_bwd_kernel:
            bwd_time_s = bench_result.get_kernel_time("dq_phase_kernel")
        elif len(kernel_names) > 0:
            bwd_time_s = bench_result.get_e2e_time(kernel_names[0], kernel_names[-1])
        else:
            raise RuntimeError("No CUDA kernels were captured by bench_kineto for flash_mla_sparse_bwd_head128_2kernels_dq")

    ms = bwd_time_s * 1e3
    print(f"Average time: {ms:.3f} ms")
    print(f"bwd io bandwidth = ", (B * S * (DQKV + DV) * topk * 2) / (ms * 1e-3) / 1e12)
    print(f"bwd tflops = ", per_token_flop * S / (ms * 1e-3) / 1e12)


if __name__ == "__main__":
    test_sparse_mla_bwd_head128_2kernels_dq(
        B=1,
        S=4096,
        SKV=8192,
        H=128,
        HKV=1,
        DQKV=576,
        DV=512,
        topk=2048,
        sm_scale=576 ** -0.5,
        dtype=torch.bfloat16,
        check_correctness=True,
        q_start_index_s=2048,
    )
