from __future__ import annotations

import torch
import torch.nn.functional as F

from . import _C as _C

__all__ = [
    "gemm",
    "sgemm",
    "hgemm",
    "flash_attention",
    "ops",
]

ops = torch.ops.sion


def _check_same_device(a: torch.Tensor, b: torch.Tensor, a_name: str, b_name: str):
    torch._check(a.device == b.device, lambda: f"{a_name} and {b_name} must be on the same device")


def _check_gemm_contract(a: torch.Tensor, b: torch.Tensor, *, nt: bool = False):
    torch._check(a.dim() == 2, lambda: "A must be 2D")
    torch._check(b.dim() == 2, lambda: "B must be 2D")
    _check_same_device(a, b, "A", "B")
    torch._check(a.dtype == b.dtype, lambda: "A and B must have the same dtype")
    torch._check(
        a.dtype in (torch.float16, torch.float32),
        lambda: "Sion GEMM supports float16 and float32 tensors",
    )
    if nt:
        torch._check(b.shape[1] == a.shape[1], lambda: "B.shape[1] must match A.shape[1]")
        return (a.shape[0], b.shape[0])
    torch._check(b.shape[0] == a.shape[1], lambda: "B.shape[0] must match A.shape[1]")
    return (a.shape[0], b.shape[1])


@torch.library.register_fake("sion::gemm")
def _fake_gemm(a, b, alpha=1.0, beta=0.0):
    shape = _check_gemm_contract(a, b)
    return torch.empty(shape, device=a.device, dtype=a.dtype)


@torch.library.register_fake("sion::sgemm")
def _fake_sgemm(a, b, alpha=1.0, beta=0.0):
    shape = _check_gemm_contract(a, b)
    torch._check(a.dtype == torch.float32, lambda: "sion::sgemm requires float32 tensors")
    return torch.empty(shape, device=a.device, dtype=a.dtype)


@torch.library.register_fake("sion::hgemm")
def _fake_hgemm(a, b, alpha=1.0, beta=0.0):
    shape = _check_gemm_contract(a, b)
    torch._check(a.dtype == torch.float16, lambda: "sion::hgemm requires float16 tensors")
    return torch.empty(shape, device=a.device, dtype=a.dtype)


@torch.library.register_fake("sion::hgemm_nt")
def _fake_hgemm_nt(a, b, alpha=1.0, beta=0.0):
    shape = _check_gemm_contract(a, b, nt=True)
    torch._check(a.dtype == torch.float16, lambda: "sion::hgemm_nt requires float16 tensors")
    return torch.empty(shape, device=a.device, dtype=a.dtype)


@torch.library.register_fake("sion::flash_attention")
def _fake_flash_attention(query, key, value):
    torch._check(query.dim() == 4, lambda: "query must be 4D")
    torch._check(key.dim() == 4, lambda: "key must be 4D")
    torch._check(value.dim() == 4, lambda: "value must be 4D")
    _check_same_device(query, key, "query", "key")
    _check_same_device(query, value, "query", "value")
    torch._check(query.shape == key.shape, lambda: "query and key must have the same shape")
    torch._check(query.shape == value.shape, lambda: "query and value must have the same shape")
    torch._check(query.dtype == torch.float16, lambda: "sion::flash_attention requires float16 tensors")
    torch._check(key.dtype == torch.float16, lambda: "sion::flash_attention requires float16 tensors")
    torch._check(value.dtype == torch.float16, lambda: "sion::flash_attention requires float16 tensors")
    return torch.empty_like(query)


def _setup_gemm_context(ctx, inputs, output):
    a, b, alpha, beta = inputs
    ctx.save_for_backward(a, b)
    ctx.alpha = alpha


def _gemm_backward(ctx, grad):
    a, b = ctx.saved_tensors
    alpha = ctx.alpha
    grad_a = grad_b = None
    if ctx.needs_input_grad[0]:
        grad_a = torch.matmul(grad, b.transpose(0, 1)) * alpha
    if ctx.needs_input_grad[1]:
        grad_b = torch.matmul(a.transpose(0, 1), grad) * alpha
    return grad_a, grad_b, None, None


def _setup_gemm_nt_context(ctx, inputs, output):
    a, b, alpha, beta = inputs
    ctx.save_for_backward(a, b)
    ctx.alpha = alpha


def _gemm_nt_backward(ctx, grad):
    a, b = ctx.saved_tensors
    alpha = ctx.alpha
    grad_a = grad_b = None
    if ctx.needs_input_grad[0]:
        grad_a = torch.matmul(grad, b) * alpha
    if ctx.needs_input_grad[1]:
        grad_b = torch.matmul(grad.transpose(0, 1), a) * alpha
    return grad_a, grad_b, None, None


def _setup_flash_attention_context(ctx, inputs, output):
    query, key, value = inputs
    ctx.save_for_backward(query, key, value)


def _flash_attention_backward(ctx, grad):
    query, key, value = ctx.saved_tensors
    needs = ctx.needs_input_grad
    with torch.enable_grad():
        query_ref = query.detach().requires_grad_(True)
        key_ref = key.detach().requires_grad_(True)
        value_ref = value.detach().requires_grad_(True)
        out = F.scaled_dot_product_attention(
            query_ref,
            key_ref,
            value_ref,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        grads = torch.autograd.grad(
            out,
            (query_ref, key_ref, value_ref),
            grad,
            allow_unused=True,
        )
    return tuple(g if need else None for g, need in zip(grads, needs))


torch.library.register_autograd(
    "sion::gemm", _gemm_backward, setup_context=_setup_gemm_context
)
torch.library.register_autograd(
    "sion::sgemm", _gemm_backward, setup_context=_setup_gemm_context
)
torch.library.register_autograd(
    "sion::hgemm", _gemm_backward, setup_context=_setup_gemm_context
)
torch.library.register_autograd(
    "sion::hgemm_nt", _gemm_nt_backward, setup_context=_setup_gemm_nt_context
)
torch.library.register_autograd(
    "sion::flash_attention",
    _flash_attention_backward,
    setup_context=_setup_flash_attention_context,
)


def gemm(a: torch.Tensor, b: torch.Tensor, alpha: float = 1.0, beta: float = 0.0):
    return ops.gemm(a, b, float(alpha), float(beta))


def sgemm(a: torch.Tensor, b: torch.Tensor, alpha: float = 1.0, beta: float = 0.0):
    return ops.sgemm(a, b, float(alpha), float(beta))


def hgemm(a: torch.Tensor, b: torch.Tensor, alpha: float = 1.0, beta: float = 0.0):
    return ops.hgemm(a, b, float(alpha), float(beta))


def flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
):
    return ops.flash_attention(query, key, value)
