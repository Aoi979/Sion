import sys


try:
    import torch
    import sion
    from correctness_common import assert_close_with_stats, clear_cuda_cache, require_cuda
except ModuleNotFoundError as exc:
    print(f"skip: missing Python dependency: {exc}", file=sys.stderr)
    raise SystemExit(77)


require_cuda()


def sgemm_ref(a: torch.Tensor, b: torch.Tensor, alpha: float, beta: float):
    c = torch.zeros((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float32)
    return torch.addmm(c, a.float(), b.float(), beta=beta, alpha=alpha)


def run_sgemm_case(
    name: str,
    m: int,
    k: int,
    n: int,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    atol: float = 1e-3,
    rtol: float = 1e-5,
    check_dispatcher: bool = False,
):
    torch.manual_seed(0)
    a = torch.rand((m, k), device="cuda", dtype=torch.float32)
    b = torch.rand((k, n), device="cuda", dtype=torch.float32)
    ref = sgemm_ref(a, b, alpha, beta)

    out = sion.sgemm(a, b, alpha, beta)
    assert_close_with_stats(name, out, ref, atol=atol, rtol=rtol)

    if check_dispatcher:
        out_ops = torch.ops.sion.gemm(a, b, alpha, beta)
        assert_close_with_stats(f"{name}.torch_ops_gemm", out_ops, ref, atol=atol, rtol=rtol)

    del a, b, ref, out
    clear_cuda_cache()


# Ports of the old C++ correctness cases.
run_sgemm_case("sgemm_basic0_cpp_port", 2048, 2048, 2048)
run_sgemm_case("sgemm_basic1_cpp_port", 2048, 1024, 1536)

# Python-only dispatcher/autograd/compile coverage.
torch.manual_seed(0)
A = torch.randn((128, 64), device="cuda", dtype=torch.float32)
B = torch.randn((64, 128), device="cuda", dtype=torch.float32)
REF = A @ B

OUT = sion.sgemm(A, B, 1.0, 0.0)
assert_close_with_stats("sion.sgemm.smoke", OUT, REF, atol=1e-3, rtol=1e-5)

OUT_OPS = torch.ops.sion.gemm(A, B, 1.0, 0.0)
assert_close_with_stats("torch.ops.sion.gemm.smoke", OUT_OPS, REF, atol=1e-3, rtol=1e-5)

A_grad = A.detach().clone().requires_grad_(True)
B_grad = B.detach().clone().requires_grad_(True)
loss = sion.sgemm(A_grad, B_grad).sum()
loss.backward()
assert A_grad.grad is not None
assert B_grad.grad is not None
assert A_grad.grad.shape == A_grad.shape
assert B_grad.grad.shape == B_grad.shape

compiled = torch.compile(lambda x, y: sion.gemm(x, y), fullgraph=True)
out_compiled = compiled(A, B)
assert_close_with_stats("torch.compile(sion.gemm)", out_compiled, REF, atol=1e-3, rtol=1e-5)

print("python_sgemm_test passed")
