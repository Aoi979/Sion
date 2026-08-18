import sys


try:
    import torch
    import torch.nn.functional as F
    import cuda_ops
    from correctness_common import assert_close_with_stats, clear_cuda_cache, require_cuda
except ModuleNotFoundError as exc:
    print(f"skip: missing Python dependency: {exc}", file=sys.stderr)
    raise SystemExit(77)


require_cuda()


def run_flash_attention_case(
    name: str,
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    *,
    scale: float = 1.0,
    atol: float = 3e-2,
    rtol: float = 3e-2,
    check_dispatcher: bool = False,
):
    torch.manual_seed(0)
    q = scale * torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
    k = scale * torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
    v = scale * torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)

    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0)
    out = cuda_ops.flash_attention(q, k, v)
    assert_close_with_stats(name, out, ref, atol=atol, rtol=rtol)

    if check_dispatcher:
        out_ops = torch.ops.cuda_ops.flash_attention(q, k, v)
        assert_close_with_stats(f"{name}.torch_ops", out_ops, ref, atol=atol, rtol=rtol)

    del q, k, v, ref, out
    clear_cuda_cache()


# Port of the old C++ correctness case.
run_flash_attention_case(
    "flash_attention_basic_cpp_port",
    batch=16,
    heads=16,
    seq_len=1280,
    head_dim=128,
    atol=3e-2,
    rtol=3e-2,
)

# Python-only dispatcher/autograd coverage.
run_flash_attention_case(
    "flash_attention_smoke",
    batch=1,
    heads=1,
    seq_len=128,
    head_dim=64,
    scale=0.1,
    atol=2e-2,
    rtol=2e-2,
    check_dispatcher=True,
)

torch.manual_seed(0)
Q = (torch.randn((1, 1, 128, 64), device="cuda", dtype=torch.float16) * 0.1).requires_grad_(True)
K = (torch.randn((1, 1, 128, 64), device="cuda", dtype=torch.float16) * 0.1).requires_grad_(True)
V = (torch.randn((1, 1, 128, 64), device="cuda", dtype=torch.float16) * 0.1).requires_grad_(True)
loss = cuda_ops.flash_attention(Q, K, V).float().sum()
loss.backward()
assert Q.grad is not None
assert K.grad is not None
assert V.grad is not None

print("python_flash_attention_test passed")
