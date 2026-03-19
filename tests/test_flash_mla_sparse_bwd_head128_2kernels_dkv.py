import pytest
import torch
import kernelkit as kk

import flash_mla


def calc_diff(a: torch.Tensor, b: torch.Tensor):
    abs_diff = torch.abs(a - b)
    max_diff = abs_diff.max().item()
    rel_diff = (abs_diff / (1e-4 + torch.abs(a))).mean().item()
    return max_diff, rel_diff


def ref_sparse_mla_dkv_interface(
    q: torch.Tensor,
    dO: torch.Tensor,
    indices: torch.Tensor,
    s: torch.Tensor,
    ds: torch.Tensor,
    s_kv: int,
    q_start_index_s: int = 0,
    topk_length: torch.Tensor | None = None,
) -> torch.Tensor:
    q_fp32 = q.float()
    dO_fp32 = dO.float()
    s_fp32 = s.float()
    ds_fp32 = ds.float()

    s_q, h_q, d_qk = q.shape
    h_kv = indices.shape[1]
    topk = indices.shape[2]
    d_v = dO.shape[-1]

    ref_dkv = torch.zeros((s_kv, h_kv, d_qk), dtype=torch.float32, device=q.device)
    if topk_length is None:
        topk_length = torch.full((s_q,), topk, dtype=torch.int32, device=q.device)

    for q_idx in range(s_q):
        max_kv_i = q_start_index_s + q_idx
        valid_topk = min(int(topk_length[q_idx].item()), topk)
        for topk_idx in range(valid_topk):
            kv_idx = int(indices[q_idx, 0, topk_idx].item())
            if kv_idx < 0 or kv_idx >= s_kv or kv_idx > max_kv_i:
                continue

            s_vec = s_fp32[q_idx, :, topk_idx].unsqueeze(1)
            ds_vec = ds_fp32[q_idx, :, topk_idx].unsqueeze(1)
            ref_dkv[kv_idx, 0, :d_v] += (s_vec * dO_fp32[q_idx]).sum(dim=0)
            ref_dkv[kv_idx, 0, :d_v] += (ds_vec * q_fp32[q_idx, :, :d_v]).sum(dim=0)
            ref_dkv[kv_idx, 0, d_v:] += (ds_vec * q_fp32[q_idx, :, d_v:]).sum(dim=0)

    return ref_dkv


def make_random_indices(
    s_q: int,
    s_kv: int,
    topk: int,
    q_start_index_s: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_length = torch.randint(
        low=max(1, topk // 4),
        high=topk + 1,
        size=(s_q,),
        dtype=torch.int32,
        device="cuda",
    )
    indices = torch.full((s_q, 1, topk), s_kv, dtype=torch.int32, device="cuda")

    for q_idx in range(s_q):
        max_valid_kv = min(s_kv, q_start_index_s + q_idx + 1)
        if max_valid_kv <= 0:
            continue
        valid_topk = min(int(topk_length[q_idx].item()), max_valid_kv)
        if valid_topk <= 0:
            continue
        indices[q_idx, 0, :valid_topk] = torch.randperm(max_valid_kv, device="cuda", dtype=torch.int64)[:valid_topk].to(torch.int32)

    return indices, topk_length


def test_sparse_mla_bwd_head128_2kernels_dkv(
    S=96,
    SKV=256,
    H=128,
    DQKV=576,
    DV=512,
    topk=64,
    dtype=torch.bfloat16,
    check_correctness=True,
    q_start_index_s=48,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    cc_major, _ = torch.cuda.get_device_capability()
    if cc_major != 10:
        pytest.skip("head128_2kernels dkv path currently targets SM100 only")

    torch.manual_seed(0)

    q = (torch.randn((S, H, DQKV), dtype=torch.float32, device="cuda") * 0.1).to(dtype)
    dO = (torch.randn((S, H, DV), dtype=torch.float32, device="cuda") * 0.1).to(dtype)
    s = (torch.randn((S, H, topk), dtype=torch.float32, device="cuda") * 0.1).to(dtype)
    ds = (torch.randn((S, H, topk), dtype=torch.float32, device="cuda") * 0.1).to(dtype)
    indices, topk_length = make_random_indices(S, SKV, topk, q_start_index_s)

    flash_dkv = flash_mla.flash_mla_sparse_bwd_head128_2kernels_dkv(
        q,
        dO,
        indices,
        s,
        ds,
        s_kv=SKV,
        d_v=DV,
        topk_length=topk_length,
        q_start_index_s=q_start_index_s,
    )
    torch.cuda.synchronize()

    if check_correctness:
        ref_dkv = ref_sparse_mla_dkv_interface(
            q,
            dO,
            indices,
            s,
            ds,
            s_kv=SKV,
            q_start_index_s=q_start_index_s,
            topk_length=topk_length,
        )
        flash_dkv_max_diff, flash_dkv_rel_diff = calc_diff(flash_dkv, ref_dkv.bfloat16())
        print(f"[ref vs flash] dKV max_diff={flash_dkv_max_diff:.6f}, rel_diff={flash_dkv_rel_diff:.6f}")

        assert kk.check_is_allclose(
            "dkv",
            flash_dkv,
            ref_dkv.bfloat16(),
            abs_tol=1e-3,
            rel_tol=8.01 / 128,
            cos_diff_tol=7e-6,
        )

    per_token_flop = 2 * H * topk * (DV + DQKV)

    def fn():
        return flash_mla.flash_mla_sparse_bwd_head128_2kernels_dkv(
            q,
            dO,
            indices,
            s,
            ds,
            s_kv=SKV,
            d_v=DV,
            topk_length=topk_length,
            q_start_index_s=q_start_index_s,
        )

    bench_result = kk.bench_kineto(fn, num_tests=100)
    if kk.is_using_profiling_tools():
        bwd_time_s = 1
    else:
        kernel_names = bench_result.get_kernel_names()
        has_bwd_kernel = any("dkv_phase_kernel" in name for name in kernel_names)

        if has_bwd_kernel:
            bwd_time_s = bench_result.get_kernel_time("dkv_phase_kernel")
        elif len(kernel_names) > 0:
            bwd_time_s = bench_result.get_e2e_time(kernel_names[0], kernel_names[-1])
        else:
            raise RuntimeError("No CUDA kernels were captured by bench_kineto for flash_mla_sparse_bwd_head128_2kernels_dkv")

    ms = bwd_time_s * 1e3
    print(f"Average time: {ms:.3f} ms")
    print(f"bwd io bandwidth = ", (S * topk * (DQKV + DV) * 2) / (ms * 1e-3) / 1e12)
    print(f"bwd tflops = ", per_token_flop * S / (ms * 1e-3) / 1e12)


if __name__ == "__main__":
    test_sparse_mla_bwd_head128_2kernels_dkv(
        S=4096,
        SKV=8192,
        H=128,
        DQKV=576,
        DV=512,
        topk=2048,
        dtype=torch.bfloat16,
        check_correctness=True,
        q_start_index_s=2048,
    )
