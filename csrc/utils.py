#!/usr/bin/env python3

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "sm100/prefill/sparse/bwd/head128/config.h"
BASE_HEAD128_CONFIG_PATH = Path(__file__).resolve().parent / "sm100/prefill/sparse/bwd/head128/config.h"

BF16_BYTES = 2
FP32_BYTES = 4
ARRAY_ALIGNED_BYTES = 16
SHARED_PLAN_ALIGNMENT = 128
DEFAULT_BARRIER_BYTES = 8
BARRIER_ALIGNMENT = 8
TMEM_ROWS = 128
SM100_SMEM_LIMIT_BYTES = 227 * 1024
SM100_TMEM_LIMIT_COLS = 512
SM100_TMEM_LIMIT_BYTES = SM100_TMEM_LIMIT_COLS * TMEM_ROWS * FP32_BYTES


@dataclass(frozen=True)
class ConfigConstants:
    d_qk: int
    d_v: int
    b_h: int
    b_topk: int

    @property
    def d_rope(self) -> int:
        return self.d_qk - self.d_v


@dataclass(frozen=True)
class TmemSegment:
    name: str
    start_col: int
    cols: int
    rows: int
    logical_cols: int
    elem_bytes: int
    dtype_label: str

    @property
    def bytes(self) -> int:
        return self.cols * TMEM_ROWS * FP32_BYTES

    @property
    def end_col(self) -> int:
        return self.start_col + self.cols


def round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def kib_str(num_bytes: int) -> str:
    return f"{num_bytes / 1024:.2f} KiB"


def pct_str(numerator: int, denominator: int) -> str:
    return f"{100.0 * numerator / denominator:.2f}%"


def read_int(name: str, text: str) -> int | None:
    match = re.search(rf"static constexpr int {name} = ([0-9]+)(?:,|;)", text)
    if not match:
        return None
    return int(match.group(1))


def parse_config_constants(config_path: Path) -> ConfigConstants:
    text = config_path.read_text()
    base_text = BASE_HEAD128_CONFIG_PATH.read_text()

    def read_with_fallback(name: str) -> int:
        value = read_int(name, text)
        if value is not None:
            return value
        value = read_int(name, base_text)
        if value is None:
            raise ValueError(f"Cannot find `{name}` in {config_path} or {BASE_HEAD128_CONFIG_PATH}")
        return value

    return ConfigConstants(
        d_qk=read_with_fallback("D_QK"),
        d_v=read_with_fallback("D_V"),
        b_h=read_with_fallback("B_H"),
        b_topk=read_with_fallback("B_TOPK"),
    )


def tensor_bytes(rows: int, cols: int, elem_bytes: int = BF16_BYTES) -> int:
    return rows * cols * elem_bytes


def array_aligned_bytes(raw_bytes: int, alignment: int = ARRAY_ALIGNED_BYTES) -> int:
    return round_up(raw_bytes, alignment)


def config_path_relative(path: Path) -> str:
    repo_root = Path(__file__).resolve().parent.parent
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def detect_profile(config_path: Path, text: str) -> str:
    if config_path.name == "dkv_config.h":
        return "dkv_2kernels"
    if re.search(r"\bstatic constexpr int D_tQ\b", text) and re.search(r"\bstatic constexpr int tQ\b", text):
        return "dq_2kernels_tq_sq"
    if "q_nope_tmem" in text or "q_nope_smem" in text:
        return "dq_2kernels_q_tmem"
    if "kv_peer" in text and "bar_kv_peer_ready" in text:
        return "dq_2kernels_kv_peer"
    if config_path.name == "dq_config.h" or "q_or_do" in text:
        return "dq_2kernels"
    return "fused"


def parse_barrier_count(text: str) -> int:
    return len(re.findall(r"\btransac_bar_t\b", text))


def read_int_with_fallback(name: str, text: str, base_text: str) -> int | None:
    value = read_int(name, text)
    if value is not None:
        return value
    return read_int(name, base_text)


def tmem_shape_map(constants: ConfigConstants, text: str, base_text: str) -> dict[str, tuple[int, int, int, str]]:
    q_nope_tmem_cols = read_int_with_fallback("D_Q_NOPE_TMEM", text, base_text)
    d_tq = read_int_with_fallback("D_tQ", text, base_text)
    d_sq = read_int_with_fallback("D_sQ", text, base_text)
    if d_sq is None and d_tq is not None:
        d_sq = constants.d_qk - d_tq
    dq_tq_sq_single_buffer = (
        d_tq is not None
        and re.search(r"\bstatic constexpr int tQ\b", text)
        and re.search(r"\bstatic constexpr int dQ\b", text)
        and not re.search(r"\bstatic constexpr int dQ_RoPE\b", text)
    )
    return {
        "dQ": (
            constants.b_h // 2,
            constants.d_qk if dq_tq_sq_single_buffer else constants.d_v,
            FP32_BYTES,
            "fp32",
        ),
        "dQ_RoPE": (constants.b_h // 2, constants.d_rope, FP32_BYTES, "fp32"),
        "P": (constants.b_h // 2, constants.b_topk, FP32_BYTES, "fp32"),
        "dP": (constants.b_h // 2, constants.b_topk, FP32_BYTES, "fp32"),
        "dKV": (constants.b_topk, 256, FP32_BYTES, "fp32"),
        "dKV_RoPE": (constants.b_topk, constants.d_rope, FP32_BYTES, "fp32"),
        **(
            {"dV": (constants.b_topk, constants.d_v, FP32_BYTES, "fp32")}
            if re.search(r"\bstatic constexpr int dV\b", text)
            else {}
        ),
        **(
            {"dK_tQ": (constants.b_topk, d_tq, FP32_BYTES, "fp32")}
            if d_tq is not None and re.search(r"\bstatic constexpr int dK_tQ\b", text)
            else {}
        ),
        **(
            {"dK_sQ": (constants.b_topk, d_sq, FP32_BYTES, "fp32")}
            if d_sq is not None and re.search(r"\bstatic constexpr int dK_sQ\b", text)
            else {}
        ),
        **(
            {"tQ": (constants.b_h // 2, d_tq, BF16_BYTES, "bf16-packed")}
            if d_tq is not None
            else {}
        ),
        **(
            {"q_nope_tmem": (constants.b_h // 2, q_nope_tmem_cols, BF16_BYTES, "bf16-packed")}
            if q_nope_tmem_cols is not None
            else {}
        ),
    }


def tmem_cols_for_shape(rows: int, cols: int, elem_bytes: int) -> int:
    value = rows * cols * elem_bytes
    if value % (TMEM_ROWS * FP32_BYTES) != 0:
        raise ValueError(f"TMEM shape [{rows}, {cols}] does not map cleanly to {TMEM_ROWS} rows")
    return value // (TMEM_ROWS * FP32_BYTES)


def parse_tmem_segments(text: str, constants: ConfigConstants) -> list[TmemSegment]:
    base_text = BASE_HEAD128_CONFIG_PATH.read_text()
    starts = {
        name: int(value)
        for name, value in re.findall(r"static constexpr int (\w+) = ([0-9]+);", text)
        if name in tmem_shape_map(constants, text, base_text)
    }
    shapes = tmem_shape_map(constants, text, base_text)
    segments = []
    for name, start_col in sorted(starts.items(), key=lambda item: item[1]):
        rows, logical_cols, elem_bytes, dtype_label = shapes[name]
        segments.append(
            TmemSegment(
                name=name,
                start_col=start_col,
                cols=tmem_cols_for_shape(rows, logical_cols, elem_bytes),
                rows=rows,
                logical_cols=logical_cols,
                elem_bytes=elem_bytes,
                dtype_label=dtype_label,
            )
        )
    return segments


def build_tmem_report(text: str, constants: ConfigConstants) -> tuple[list[str], int]:
    segments = parse_tmem_segments(text, constants)
    if not segments:
        return ["TMEM: no `tmem_cols` segments detected"], 0

    total_cols = max(segment.end_col for segment in segments)
    total_bytes = total_cols * TMEM_ROWS * FP32_BYTES
    lines = [
        "TMEM:",
        *[
            f"  {segment.name:<8} = col[{segment.start_col:>3}:{segment.end_col:>3})"
            f"  -> [{segment.rows} x {segment.logical_cols}] {segment.dtype_label}"
            f"  = {segment.cols:>3} cols  ({kib_str(segment.bytes)})"
            for segment in segments
        ],
        f"  total        = {total_cols} / {SM100_TMEM_LIMIT_COLS} cols"
        f"  ({kib_str(total_bytes)} / {kib_str(SM100_TMEM_LIMIT_BYTES)}, {pct_str(total_cols, SM100_TMEM_LIMIT_COLS)})",
        f"  limit check  = {'OK' if total_cols <= SM100_TMEM_LIMIT_COLS else 'EXCEEDED'}",
    ]
    return lines, total_cols


def build_dkv_2kernels_tmem_report(
    text: str,
    constants: ConfigConstants,
) -> tuple[list[str], int]:
    if not re.search(r"\bstatic constexpr int dKV\b", text):
        raise ValueError("`dKV` is required for the dKV TMEM report")

    dkv_cols = tmem_cols_for_shape(constants.b_topk, constants.d_v, FP32_BYTES)
    total_bytes = dkv_cols * TMEM_ROWS * FP32_BYTES
    lines = [
        "TMEM:",
        f"  dKV      = col[  0:{dkv_cols:>3})  -> [128 x 512] fp32  = {dkv_cols:>3} cols  ({kib_str(total_bytes)})",
        "  phase use     = dV full tile -> reuse for dK_tQ -> reuse front part for dK_sQ",
        f"  total        = {dkv_cols} / {SM100_TMEM_LIMIT_COLS} cols"
        f"  ({kib_str(total_bytes)} / {kib_str(SM100_TMEM_LIMIT_BYTES)}, {pct_str(dkv_cols, SM100_TMEM_LIMIT_COLS)})",
        f"  limit check  = {'OK' if dkv_cols <= SM100_TMEM_LIMIT_COLS else 'EXCEEDED'}",
    ]
    return lines, dkv_cols


def build_fused_smem_report(constants: ConfigConstants, barrier_count: int, barrier_bytes: int) -> tuple[list[str], int]:
    q_rows = constants.b_h // 2
    k_rows = constants.b_topk // 2

    k_nope = array_aligned_bytes(tensor_bytes(k_rows, constants.d_v))
    k_rope = array_aligned_bytes(tensor_bytes(k_rows, constants.d_rope))
    kv_peer = array_aligned_bytes(tensor_bytes(k_rows, constants.d_qk))
    q_nope = array_aligned_bytes(tensor_bytes(q_rows, constants.d_v))
    q_rope = array_aligned_bytes(tensor_bytes(q_rows, constants.d_rope))
    dq = array_aligned_bytes(tensor_bytes(q_rows, constants.d_qk))
    d_o = array_aligned_bytes(tensor_bytes(q_rows, constants.d_v))
    s = array_aligned_bytes(tensor_bytes(q_rows, constants.b_topk))
    ds = array_aligned_bytes(tensor_bytes(q_rows, constants.b_topk))
    is_k_valid = constants.b_topk // 8
    barrier_total = barrier_count * barrier_bytes
    tmem_start_addr = array_aligned_bytes(4)

    union_q_kv = k_nope + k_rope + kv_peer + q_nope + q_rope
    union_u = max(union_q_kv, dq)

    offset_after_payload = union_u + d_o + s + ds + is_k_valid
    offset_after_barriers = round_up(offset_after_payload, BARRIER_ALIGNMENT) + barrier_total
    offset_after_tmem = round_up(offset_after_barriers, ARRAY_ALIGNED_BYTES) + tmem_start_addr
    tail_padding = round_up(offset_after_tmem, SHARED_PLAN_ALIGNMENT) - offset_after_tmem
    total = offset_after_tmem + tail_padding

    lines = [
        "SMEM:",
        f"  k_nope        = ({k_rows} x {constants.d_v} x 2)   = {k_nope:>6} B  ({kib_str(k_nope)})",
        f"  k_rope        = ({k_rows} x {constants.d_rope} x 2) = {k_rope:>6} B  ({kib_str(k_rope)})",
        f"  kv_peer       = ({k_rows} x {constants.d_qk} x 2) = {kv_peer:>6} B  ({kib_str(kv_peer)})",
        f"  q_nope        = ({q_rows} x {constants.d_v} x 2)   = {q_nope:>6} B  ({kib_str(q_nope)})",
        f"  q_rope        = ({q_rows} x {constants.d_rope} x 2) = {q_rope:>6} B  ({kib_str(q_rope)})",
        f"  dq            = ({q_rows} x {constants.d_qk} x 2) = {dq:>6} B  ({kib_str(dq)})",
        f"  dO            = ({q_rows} x {constants.d_v} x 2)   = {d_o:>6} B  ({kib_str(d_o)})",
        f"  s             = ({q_rows} x {constants.b_topk} x 2) = {s:>6} B  ({kib_str(s)})",
        f"  ds            = ({q_rows} x {constants.b_topk} x 2) = {ds:>6} B  ({kib_str(ds)})",
        f"  union.q_kv    = {union_q_kv:>6} B  ({kib_str(union_q_kv)})",
        f"  union.dq      = {dq:>6} B  ({kib_str(dq)})",
        f"  union.u       = {union_u:>6} B  ({kib_str(union_u)})",
        f"  s_ds          = {s + ds:>6} B  ({kib_str(s + ds)})",
        f"  is_k_valid    = {is_k_valid:>6} B",
        f"  barriers(est) = {barrier_count} x {barrier_bytes} B = {barrier_total:>6} B",
        f"  tmem_start    = {tmem_start_addr:>6} B  (array_aligned<uint32_t, 1>)",
        f"  tail padding  = {tail_padding:>6} B",
        f"  total         = {total:>6} B  ({kib_str(total)})",
        f"  limit check   = {'OK' if total <= SM100_SMEM_LIMIT_BYTES else 'EXCEEDED'}"
        f"  ({kib_str(total)} / {kib_str(SM100_SMEM_LIMIT_BYTES)}, {pct_str(total, SM100_SMEM_LIMIT_BYTES)})",
    ]
    return lines, total


def build_dq_2kernels_smem_report(constants: ConfigConstants, barrier_count: int, barrier_bytes: int) -> tuple[list[str], int]:
    q_rows = constants.b_h // 2
    k_rows = constants.b_topk // 2

    k_nope = array_aligned_bytes(tensor_bytes(k_rows, constants.d_v))
    k_rope = array_aligned_bytes(tensor_bytes(k_rows, constants.d_rope))
    q_nope = array_aligned_bytes(tensor_bytes(q_rows, constants.d_v))
    q_rope = array_aligned_bytes(tensor_bytes(q_rows, constants.d_rope))
    d_o = array_aligned_bytes(tensor_bytes(q_rows, constants.d_v))
    dq = array_aligned_bytes(tensor_bytes(q_rows, constants.d_qk))
    s = array_aligned_bytes(tensor_bytes(q_rows, constants.b_topk))
    ds = array_aligned_bytes(tensor_bytes(q_rows, constants.b_topk))
    is_k_valid = constants.b_topk // 8
    barrier_total = barrier_count * barrier_bytes
    tmem_start_addr = array_aligned_bytes(4)

    q_or_do = max(q_nope + q_rope, d_o)
    union_q_kv = k_nope + k_rope + q_or_do
    union_u = max(union_q_kv, dq)

    offset_after_payload = union_u + s + ds + is_k_valid
    offset_after_barriers = round_up(offset_after_payload, BARRIER_ALIGNMENT) + barrier_total
    offset_after_tmem = round_up(offset_after_barriers, ARRAY_ALIGNED_BYTES) + tmem_start_addr
    tail_padding = round_up(offset_after_tmem, SHARED_PLAN_ALIGNMENT) - offset_after_tmem
    total = offset_after_tmem + tail_padding

    lines = [
        "SMEM:",
        f"  k_nope        = ({k_rows} x {constants.d_v} x 2)   = {k_nope:>6} B  ({kib_str(k_nope)})",
        f"  k_rope        = ({k_rows} x {constants.d_rope} x 2) = {k_rope:>6} B  ({kib_str(k_rope)})",
        f"  q_nope        = ({q_rows} x {constants.d_v} x 2)   = {q_nope:>6} B  ({kib_str(q_nope)})",
        f"  q_rope        = ({q_rows} x {constants.d_rope} x 2) = {q_rope:>6} B  ({kib_str(q_rope)})",
        f"  dO            = ({q_rows} x {constants.d_v} x 2)   = {d_o:>6} B  ({kib_str(d_o)})",
        f"  q_or_do       = max(q={q_nope + q_rope}, dO={d_o}) = {q_or_do:>6} B  ({kib_str(q_or_do)})",
        f"  dq            = ({q_rows} x {constants.d_qk} x 2) = {dq:>6} B  ({kib_str(dq)})",
        f"  s             = ({q_rows} x {constants.b_topk} x 2) = {s:>6} B  ({kib_str(s)})",
        f"  ds            = ({q_rows} x {constants.b_topk} x 2) = {ds:>6} B  ({kib_str(ds)})",
        f"  union.q_kv    = {union_q_kv:>6} B  ({kib_str(union_q_kv)})",
        f"  union.dq      = {dq:>6} B  ({kib_str(dq)})",
        f"  union.u       = {union_u:>6} B  ({kib_str(union_u)})",
        f"  s_ds          = {s + ds:>6} B  ({kib_str(s + ds)})",
        f"  is_k_valid    = {is_k_valid:>6} B",
        f"  barriers(est) = {barrier_count} x {barrier_bytes} B = {barrier_total:>6} B",
        f"  tmem_start    = {tmem_start_addr:>6} B  (array_aligned<uint32_t, 1>)",
        f"  tail padding  = {tail_padding:>6} B",
        f"  total         = {total:>6} B  ({kib_str(total)})",
        f"  limit check   = {'OK' if total <= SM100_SMEM_LIMIT_BYTES else 'EXCEEDED'}"
        f"  ({kib_str(total)} / {kib_str(SM100_SMEM_LIMIT_BYTES)}, {pct_str(total, SM100_SMEM_LIMIT_BYTES)})",
    ]
    return lines, total


def build_dq_2kernels_q_tmem_smem_report(
    text: str,
    constants: ConfigConstants,
    barrier_count: int,
    barrier_bytes: int,
) -> tuple[list[str], int]:
    base_text = BASE_HEAD128_CONFIG_PATH.read_text()
    q_nope_tmem_cols = read_int_with_fallback("D_Q_NOPE_TMEM", text, base_text)
    if q_nope_tmem_cols is None:
        raise ValueError("`D_Q_NOPE_TMEM` is required for the q-in-TMEM dQ profile")

    q_rows = constants.b_h // 2
    k_rows = constants.b_topk // 2
    q_nope_smem_cols = constants.d_v - q_nope_tmem_cols
    if q_nope_smem_cols < 0:
        raise ValueError("TMEM-backed Q NoPE cols exceed D_V")

    k_nope = array_aligned_bytes(tensor_bytes(k_rows, constants.d_v))
    k_rope = array_aligned_bytes(tensor_bytes(k_rows, constants.d_rope))
    q_nope_tmem = tensor_bytes(q_rows, q_nope_tmem_cols)
    q_nope_smem = array_aligned_bytes(tensor_bytes(q_rows, q_nope_smem_cols))
    q_rope = array_aligned_bytes(tensor_bytes(q_rows, constants.d_rope))
    d_o = array_aligned_bytes(tensor_bytes(q_rows, constants.d_v))
    dq = array_aligned_bytes(tensor_bytes(q_rows, constants.d_qk))
    s = array_aligned_bytes(tensor_bytes(q_rows, constants.b_topk))
    ds = array_aligned_bytes(tensor_bytes(q_rows, constants.b_topk))
    is_k_valid = constants.b_topk // 8
    barrier_total = barrier_count * barrier_bytes
    tmem_start_addr = array_aligned_bytes(4)

    union_q_kv = k_nope + k_rope + q_nope_smem + q_rope
    union_u = max(union_q_kv, dq)

    offset_after_payload = union_u + d_o + s + ds + is_k_valid
    offset_after_barriers = round_up(offset_after_payload, BARRIER_ALIGNMENT) + barrier_total
    offset_after_tmem = round_up(offset_after_barriers, ARRAY_ALIGNED_BYTES) + tmem_start_addr
    tail_padding = round_up(offset_after_tmem, SHARED_PLAN_ALIGNMENT) - offset_after_tmem
    total = offset_after_tmem + tail_padding

    lines = [
        "SMEM:",
        f"  k_nope        = ({k_rows} x {constants.d_v} x 2)   = {k_nope:>6} B  ({kib_str(k_nope)})",
        f"  k_rope        = ({k_rows} x {constants.d_rope} x 2) = {k_rope:>6} B  ({kib_str(k_rope)})",
        f"  q_nope_tmem   = ({q_rows} x {q_nope_tmem_cols} x 2) = {q_nope_tmem:>6} B  ({kib_str(q_nope_tmem)})  [TMEM]",
        f"  q_nope_smem   = ({q_rows} x {q_nope_smem_cols} x 2) = {q_nope_smem:>6} B  ({kib_str(q_nope_smem)})",
        f"  q_rope        = ({q_rows} x {constants.d_rope} x 2) = {q_rope:>6} B  ({kib_str(q_rope)})",
        f"  dO            = ({q_rows} x {constants.d_v} x 2)   = {d_o:>6} B  ({kib_str(d_o)})",
        f"  dq            = ({q_rows} x {constants.d_qk} x 2) = {dq:>6} B  ({kib_str(dq)})",
        f"  s             = ({q_rows} x {constants.b_topk} x 2) = {s:>6} B  ({kib_str(s)})",
        f"  ds            = ({q_rows} x {constants.b_topk} x 2) = {ds:>6} B  ({kib_str(ds)})",
        f"  union.q_kv    = {union_q_kv:>6} B  ({kib_str(union_q_kv)})",
        f"  union.dq      = {dq:>6} B  ({kib_str(dq)})",
        f"  union.u       = {union_u:>6} B  ({kib_str(union_u)})",
        f"  dO resident   = {d_o:>6} B  ({kib_str(d_o)})",
        f"  s_ds          = {s + ds:>6} B  ({kib_str(s + ds)})",
        f"  is_k_valid    = {is_k_valid:>6} B",
        f"  barriers(est) = {barrier_count} x {barrier_bytes} B = {barrier_total:>6} B",
        f"  tmem_start    = {tmem_start_addr:>6} B  (array_aligned<uint32_t, 1>)",
        f"  tail padding  = {tail_padding:>6} B",
        f"  total         = {total:>6} B  ({kib_str(total)})",
        f"  limit check   = {'OK' if total <= SM100_SMEM_LIMIT_BYTES else 'EXCEEDED'}"
        f"  ({kib_str(total)} / {kib_str(SM100_SMEM_LIMIT_BYTES)}, {pct_str(total, SM100_SMEM_LIMIT_BYTES)})",
    ]
    return lines, total


def build_dq_2kernels_tq_sq_smem_report(
    text: str,
    constants: ConfigConstants,
    barrier_count: int,
    barrier_bytes: int,
) -> tuple[list[str], int]:
    base_text = BASE_HEAD128_CONFIG_PATH.read_text()
    d_tq = read_int_with_fallback("D_tQ", text, base_text)
    if d_tq is None:
        raise ValueError("`D_tQ` is required for the tQ/sQ dQ profile")

    d_sq = read_int_with_fallback("D_sQ", text, base_text)
    if d_sq is None:
        d_sq = constants.d_qk - d_tq

    q_rows = constants.b_h // 2
    k_rows = constants.b_topk // 2

    k = array_aligned_bytes(tensor_bytes(k_rows, constants.d_qk))
    t_q = tensor_bytes(q_rows, d_tq)
    s_q = array_aligned_bytes(tensor_bytes(q_rows, d_sq))
    d_o = array_aligned_bytes(tensor_bytes(q_rows, constants.d_v))
    dq = array_aligned_bytes(tensor_bytes(q_rows, constants.d_qk))
    s = array_aligned_bytes(tensor_bytes(q_rows, constants.b_topk))
    ds = array_aligned_bytes(tensor_bytes(q_rows, constants.b_topk))
    is_k_valid = constants.b_topk // 8
    barrier_total = barrier_count * barrier_bytes
    tmem_start_addr = array_aligned_bytes(4)

    union_q_kv = k + s_q
    union_u = max(union_q_kv, dq)

    offset_after_payload = union_u + d_o + s + ds + is_k_valid
    offset_after_barriers = round_up(offset_after_payload, BARRIER_ALIGNMENT) + barrier_total
    offset_after_tmem = round_up(offset_after_barriers, ARRAY_ALIGNED_BYTES) + tmem_start_addr
    tail_padding = round_up(offset_after_tmem, SHARED_PLAN_ALIGNMENT) - offset_after_tmem
    total = offset_after_tmem + tail_padding

    lines = [
        "SMEM:",
        f"  k             = ({k_rows} x {constants.d_qk} x 2) = {k:>6} B  ({kib_str(k)})",
        f"  tQ            = ({q_rows} x {d_tq} x 2) = {t_q:>6} B  ({kib_str(t_q)})  [TMEM]",
        f"  sQ            = ({q_rows} x {d_sq} x 2) = {s_q:>6} B  ({kib_str(s_q)})",
        f"  dO            = ({q_rows} x {constants.d_v} x 2)   = {d_o:>6} B  ({kib_str(d_o)})",
        f"  dq            = ({q_rows} x {constants.d_qk} x 2) = {dq:>6} B  ({kib_str(dq)})",
        f"  s             = ({q_rows} x {constants.b_topk} x 2) = {s:>6} B  ({kib_str(s)})",
        f"  ds            = ({q_rows} x {constants.b_topk} x 2) = {ds:>6} B  ({kib_str(ds)})",
        f"  union.q_kv    = {union_q_kv:>6} B  ({kib_str(union_q_kv)})",
        f"  union.dq      = {dq:>6} B  ({kib_str(dq)})",
        f"  union.u       = {union_u:>6} B  ({kib_str(union_u)})",
        f"  dO resident   = {d_o:>6} B  ({kib_str(d_o)})",
        f"  s_ds          = {s + ds:>6} B  ({kib_str(s + ds)})",
        f"  is_k_valid    = {is_k_valid:>6} B",
        f"  barriers(est) = {barrier_count} x {barrier_bytes} B = {barrier_total:>6} B",
        f"  tmem_start    = {tmem_start_addr:>6} B  (array_aligned<uint32_t, 1>)",
        f"  tail padding  = {tail_padding:>6} B",
        f"  total         = {total:>6} B  ({kib_str(total)})",
        f"  limit check   = {'OK' if total <= SM100_SMEM_LIMIT_BYTES else 'EXCEEDED'}"
        f"  ({kib_str(total)} / {kib_str(SM100_SMEM_LIMIT_BYTES)}, {pct_str(total, SM100_SMEM_LIMIT_BYTES)})",
    ]
    return lines, total


def build_dkv_2kernels_smem_report(
    text: str,
    constants: ConfigConstants,
    barrier_count: int,
    barrier_bytes: int,
) -> tuple[list[str], int]:
    base_text = BASE_HEAD128_CONFIG_PATH.read_text()
    d_tq = read_int_with_fallback("D_tQ", text, base_text)
    if d_tq is None:
        raise ValueError("`D_tQ` is required for the dKV profile")

    d_sq = read_int_with_fallback("D_sQ", text, base_text)
    if d_sq is None:
        d_sq = constants.d_qk - d_tq

    q_rows = constants.b_h // 2
    q = array_aligned_bytes(tensor_bytes(q_rows, constants.d_qk))
    d_o = array_aligned_bytes(tensor_bytes(q_rows, constants.d_v))
    s = array_aligned_bytes(tensor_bytes(q_rows, constants.b_topk))
    ds = array_aligned_bytes(tensor_bytes(q_rows, constants.b_topk))
    is_k_valid = constants.b_topk // 8
    barrier_total = barrier_count * barrier_bytes
    tmem_start_addr = array_aligned_bytes(4)

    total_before_barriers = q + d_o + s + ds + is_k_valid
    offset_after_barriers = round_up(total_before_barriers, BARRIER_ALIGNMENT) + barrier_total
    offset_after_tmem = round_up(offset_after_barriers, ARRAY_ALIGNED_BYTES) + tmem_start_addr
    tail_padding = round_up(offset_after_tmem, SHARED_PLAN_ALIGNMENT) - offset_after_tmem
    total = offset_after_tmem + tail_padding

    lines = [
        "SMEM:",
        f"  q             = ({q_rows} x {constants.d_qk} x 2) = {q:>6} B  ({kib_str(q)})",
        f"    tQ          = ({q_rows} x {d_tq} x 2) = {tensor_bytes(q_rows, d_tq):>6} B  ({kib_str(tensor_bytes(q_rows, d_tq))})",
        f"    sQ          = ({q_rows} x {d_sq} x 2) = {tensor_bytes(q_rows, d_sq):>6} B  ({kib_str(tensor_bytes(q_rows, d_sq))})",
        f"  dO            = ({q_rows} x {constants.d_v} x 2)   = {d_o:>6} B  ({kib_str(d_o)})",
        f"  s             = ({q_rows} x {constants.b_topk} x 2) = {s:>6} B  ({kib_str(s)})",
        f"  ds            = ({q_rows} x {constants.b_topk} x 2) = {ds:>6} B  ({kib_str(ds)})",
        f"  s_ds          = {s + ds:>6} B  ({kib_str(s + ds)})",
        f"  is_k_valid    = {is_k_valid:>6} B",
        f"  barriers(est) = {barrier_count} x {barrier_bytes} B = {barrier_total:>6} B",
        f"  tmem_start    = {tmem_start_addr:>6} B  (array_aligned<uint32_t, 1>)",
        f"  tail padding  = {tail_padding:>6} B",
        f"  total         = {total:>6} B  ({kib_str(total)})",
        f"  limit check   = {'OK' if total <= SM100_SMEM_LIMIT_BYTES else 'EXCEEDED'}"
        f"  ({kib_str(total)} / {kib_str(SM100_SMEM_LIMIT_BYTES)}, {pct_str(total, SM100_SMEM_LIMIT_BYTES)})",
    ]
    return lines, total


def build_report(config_path: Path, barrier_bytes: int) -> str:
    text = config_path.read_text()
    constants = parse_config_constants(config_path)
    profile = detect_profile(config_path, text)
    barrier_count = parse_barrier_count(text)

    if profile == "dkv_2kernels":
        smem_lines, _ = build_dkv_2kernels_smem_report(text, constants, barrier_count, barrier_bytes)
        tmem_lines, _ = build_dkv_2kernels_tmem_report(text, constants)
    elif profile == "dq_2kernels_kv_peer":
        smem_lines, _ = build_fused_smem_report(constants, barrier_count, barrier_bytes)
        tmem_lines, _ = build_tmem_report(text, constants)
    elif profile == "dq_2kernels_tq_sq":
        smem_lines, _ = build_dq_2kernels_tq_sq_smem_report(text, constants, barrier_count, barrier_bytes)
        tmem_lines, _ = build_tmem_report(text, constants)
    elif profile == "dq_2kernels_q_tmem":
        smem_lines, _ = build_dq_2kernels_q_tmem_smem_report(text, constants, barrier_count, barrier_bytes)
        tmem_lines, _ = build_tmem_report(text, constants)
    elif profile == "dq_2kernels":
        smem_lines, _ = build_dq_2kernels_smem_report(constants, barrier_count, barrier_bytes)
        tmem_lines, _ = build_tmem_report(text, constants)
    else:
        smem_lines, _ = build_fused_smem_report(constants, barrier_count, barrier_bytes)
        tmem_lines, _ = build_tmem_report(text, constants)

    lines = [
        f"Config: {config_path_relative(config_path)}",
        f"Profile: {profile}",
        f"Constants: D_QK={constants.d_qk}, D_V={constants.d_v}, D_ROPE={constants.d_rope}, B_H={constants.b_h}, B_TOPK={constants.b_topk}",
        "",
        *smem_lines,
        "",
        *tmem_lines,
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate SM100 head128 backward SMEM/TMEM usage")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Target config header (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--barrier-bytes",
        type=int,
        default=DEFAULT_BARRIER_BYTES,
        help=f"Estimated bytes per transac_bar_t (default: {DEFAULT_BARRIER_BYTES})",
    )
    args = parser.parse_args()

    print(build_report(args.config.resolve(), barrier_bytes=args.barrier_bytes))


if __name__ == "__main__":
    main()
