import sys


try:
    import torch
    import sion
    from correctness_common import assert_close_with_stats, clear_cuda_cache, require_cuda
except ModuleNotFoundError as exc:
    print(f"skip: missing Python dependency: {exc}", file=sys.stderr)
    raise SystemExit(77)


require_cuda()


def hgemm_ref(a: torch.Tensor, b: torch.Tensor, alpha: float, beta: float):
    c = torch.zeros((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float16)
    return torch.addmm(c, a, b, beta=beta, alpha=alpha)


def run_hgemm_case(
    name: str,
    m: int,
    k: int,
    n: int,
    *,
    alpha: float = 1.0,
    beta: float = 0.0,
    atol: float = 6.0,
    rtol: float = 2e-2,
    nt: bool = False,
):
    torch.manual_seed(0)
    a = torch.rand((m, k), device="cuda", dtype=torch.float16)
    b = torch.rand((k, n), device="cuda", dtype=torch.float16)
    ref = hgemm_ref(a, b, alpha, beta)

    if nt:
        out = torch.ops.sion.hgemm_nt(a, b.transpose(0, 1).contiguous(), alpha, beta)
    else:
        out = sion.hgemm(a, b, alpha, beta)

    assert_close_with_stats(name, out, ref, atol=atol, rtol=rtol)
    del a, b, ref, out
    clear_cuda_cache()


# Ports of the old C++ correctness cases.
run_hgemm_case("hgemm_basic0_cpp_port", 2048, 2048, 2048)
run_hgemm_case("hgemm_nt_basic0_cpp_port", 2048, 2048, 2048, nt=True)

# Python-only dispatcher/autograd coverage.
torch.manual_seed(0)
A = torch.randn((128, 64), device="cuda", dtype=torch.float16) * 0.1
B = torch.randn((64, 128), device="cuda", dtype=torch.float16) * 0.1
REF = (A.float() @ B.float()).half()

OUT = sion.hgemm(A, B, 1.0, 0.0)
assert_close_with_stats("sion.hgemm.smoke", OUT, REF, atol=3e-2, rtol=3e-2)

OUT_OPS = torch.ops.sion.hgemm(A, B, 1.0, 0.0)
assert_close_with_stats("torch.ops.sion.hgemm.smoke", OUT_OPS, REF, atol=3e-2, rtol=3e-2)

A_grad = A.detach().clone().requires_grad_(True)
B_grad = B.detach().clone().requires_grad_(True)
loss = sion.hgemm(A_grad, B_grad).float().sum()
loss.backward()
assert A_grad.grad is not None
assert B_grad.grad is not None
assert A_grad.grad.shape == A_grad.shape
assert B_grad.grad.shape == B_grad.shape

print("python_hgemm_test passed")
