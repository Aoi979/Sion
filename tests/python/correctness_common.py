from __future__ import annotations

from dataclasses import dataclass
import math
import sys

import torch


def require_cuda() -> None:
    if not torch.cuda.is_available():
        print("skip: CUDA is not available", file=sys.stderr)
        raise SystemExit(77)


@dataclass
class ErrorStats:
    max_abs: float
    mean_abs: float
    rms: float
    nan_count: int
    inf_count: int


def error_stats(actual: torch.Tensor, expected: torch.Tensor) -> ErrorStats:
    torch._check(actual.shape == expected.shape, lambda: "tensor shape mismatch")
    actual_f = actual.detach().float()
    expected_f = expected.detach().float()
    diff = (actual_f - expected_f).abs()
    return ErrorStats(
        max_abs=diff.max().item(),
        mean_abs=diff.mean().item(),
        rms=torch.sqrt((diff * diff).mean()).item(),
        nan_count=torch.isnan(actual_f).sum().item(),
        inf_count=torch.isinf(actual_f).sum().item(),
    )


def assert_close_with_stats(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float = 0.0,
) -> ErrorStats:
    torch.cuda.synchronize()
    stats = error_stats(actual, expected)
    print(
        f"{name}: max_abs={stats.max_abs:.6g} mean_abs={stats.mean_abs:.6g} "
        f"rms={stats.rms:.6g} nan={stats.nan_count} inf={stats.inf_count}"
    )
    if stats.nan_count or stats.inf_count:
        raise AssertionError(f"{name}: output contains NaN/Inf")

    actual_f = actual.detach().float()
    expected_f = expected.detach().float()
    allowed = atol + rtol * expected_f.abs()
    violation = ((actual_f - expected_f).abs() - allowed).max().item()
    if math.isfinite(violation) and violation > 0.0:
        raise AssertionError(
            f"{name}: max_abs={stats.max_abs:.6g} exceeds atol={atol} rtol={rtol}"
        )
    return stats


def clear_cuda_cache() -> None:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
