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


def assert_close(name, actual, expected, atol=1e-3, rtol=1e-3):
    max_abs = (actual - expected).abs().max().item()
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as exc:
        raise AssertionError(f"{name} max_abs={max_abs}") from exc


torch.manual_seed(0)
M, K, N = 128, 64, 128
A = torch.randn((M, K), device="cuda", dtype=torch.float32)
B = torch.randn((K, N), device="cuda", dtype=torch.float32)
ref = A @ B

out = sion.sgemm(A, B, 1.0, 0.0, kernel_name="cute_sgemm_64x64_nn")
torch.cuda.synchronize()
assert_close("cute_sgemm_64x64_nn", out, ref)

out_sm80 = sion.sgemm(
    A, B, 1.0, 0.0, kernel_name="sm80_sgemm_128x128x8_stage5"
)
torch.cuda.synchronize()
assert_close("sm80_sgemm_128x128x8_stage5", out_sm80, ref)

print("python_sgemm_test passed")
