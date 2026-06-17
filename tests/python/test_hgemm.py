import sys


try:
    import torch
    import sion
except ModuleNotFoundError as exc:
    print(f"skip: missing Python dependency: {exc}", file=sys.stderr)
    raise SystemExit(77)


if not torch.cuda.is_available():
    print("skip: CUDA is not available", file=sys.stderr)
    raise SystemExit(77)


def assert_close(name, actual, expected, atol=3e-2, rtol=3e-2):
    max_abs = (actual.float() - expected.float()).abs().max().item()
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as exc:
        raise AssertionError(f"{name} max_abs={max_abs}") from exc


torch.manual_seed(0)
M, K, N = 128, 64, 128
A = torch.randn((M, K), device="cuda", dtype=torch.float16) * 0.1
B = torch.randn((K, N), device="cuda", dtype=torch.float16) * 0.1
ref = (A.float() @ B.float()).half()

out = sion.hgemm(A, B, 1.0, 0.0, kernel_name="cute_hgemm_128x128_nn")
torch.cuda.synchronize()
assert_close("cute_hgemm_128x128_nn", out, ref)

out_sm80 = sion.hgemm(
    A, B, 1.0, 0.0, kernel_name="sm80_hgemm_128x128x64_fp32acc"
)
torch.cuda.synchronize()
assert_close("sm80_hgemm_128x128x64_fp32acc", out_sm80, ref)

B_t = B.t().contiguous()
out_nt = sion.hgemm_nt(A, B_t, 1.0, 0.0)
torch.cuda.synchronize()
assert_close("cute_hgemm_128x128_nt", out_nt, ref)

print("python_hgemm_test passed")
