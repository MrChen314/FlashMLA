import pytest
import torch
import kernelkit as kk

import flash_mla


def ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, q_start_index_s=0):
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)
    b, sq, h, dim_q = q.shape
    b, sk, g, _ = kv.shape

    dim = 512
    k = kv
    v = kv[..., :dim]

    compressed_casual_mask = torch.arange(
        q_start_index_s, q_start_index_s + sq, dtype=torch.int32, device="cuda"
    ).view(-1, 1) >= torch.arange(0, sk, 1, dtype=torch.int32, device="cuda").view(1, -1)

    mask = q.new_zeros(b, g, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
    mask = mask[..., :-1]
    mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
    mask = mask.view(b, g, 1, sq, sk)

    q = q.view(b, sq, g, -1, dim_q)
    score = torch.einsum("bmghd,bngd->bghmn", q, k)
    sm_scale = dim_q ** -0.5 if sm_scale is None else sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
    p = score.softmax(dim=-1)
    p = p.view(b, g, h // g, -1, sq, sk)
    p = p.view(b, g, -1, sq, sk)
    o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
    return o.reshape(b, sq, h, dim).to(torch.bfloat16)


def ref_sparse_mla_bwd_interface(q, kv, do, indices, sm_scale=None, q_start_index_s=0):
    q = q.detach().clone()
    kv = kv.detach().clone()
    q.requires_grad = True
    kv.requires_grad = True
    o = ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale, q_start_index_s)
    o.backward(do)
    return q.grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sparse_mla_bwd_head128_2kernels_dq_accuracy():
    cc_major, _ = torch.cuda.get_device_capability()
    if cc_major != 10:
        pytest.skip("head128_2kernels dq path currently targets SM100 only")

    torch.manual_seed(0)

    b = 1
    s = 64
    skv = 256
    h = 128
    hkv = 1
    dqkv = 576
    dv = 512
    topk = 64
    q_start_index_s = 64
    sm_scale = dqkv ** -0.5

    q = torch.randn((b, s, h, dqkv), dtype=torch.bfloat16, device="cuda").requires_grad_(True)
    kv = torch.randn((b, skv, hkv, dqkv), dtype=torch.bfloat16, device="cuda").requires_grad_(True)
    do = torch.randn((b, s, h, dv), dtype=torch.bfloat16, device="cuda")

    indices = torch.full((b, s, hkv, topk), skv, dtype=torch.int32, device="cuda")
    for t in range(s):
        max_kv_i = min(skv, q_start_index_s + t)
        indices[0, t, 0] = torch.randperm(max_kv_i, device="cuda")[:topk]

    flash_out, _, flash_lse, _ = flash_mla.flash_mla_sparse_fwd(
        q.squeeze(0).contiguous(),
        kv.squeeze(0).contiguous(),
        indices.squeeze(0).contiguous(),
        sm_scale=sm_scale,
        q_start_index_s=q_start_index_s,
    )

    flash_dq = flash_mla.flash_mla_sparse_bwd_head128_2kernels_dq(
        q.squeeze(0).contiguous(),
        kv.squeeze(0).contiguous(),
        flash_out,
        do.squeeze(0).contiguous(),
        indices.squeeze(0).contiguous(),
        flash_lse,
        sm_scale=sm_scale,
        q_start_index_s=q_start_index_s,
    )
    ref_dq = ref_sparse_mla_bwd_interface(q, kv, do, indices, sm_scale=sm_scale, q_start_index_s=q_start_index_s).squeeze(0)

    assert kk.check_is_allclose(
        "dq",
        flash_dq.float(),
        ref_dq.float(),
        abs_tol=1e-3,
        rel_tol=8.01 / 128,
        cos_diff_tol=7e-6,
    )
