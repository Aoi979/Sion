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


torch.manual_seed(0)
B, H, N, D = 1, 1, 64, 64
Q = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16) * 0.1
K = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16) * 0.1
V = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16) * 0.1

out = sion.flash_attention(Q, K, V)
ref = torch.nn.functional.scaled_dot_product_attention(
    Q, K, V, attn_mask=None, dropout_p=0.0
)
torch.cuda.synchronize()

max_abs = (out.float() - ref.float()).abs().max().item()
try:
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
except AssertionError as exc:
    raise AssertionError(f"flash_attention max_abs={max_abs}") from exc

print("python_flash_attention_test passed")
