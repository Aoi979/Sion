import torch
import sion

print(sion.__doc__)

batch_size = 8
seq_len = 1280
num_heads = 8
head_dim = 128

dtype = torch.float16

# query, key, value
Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=dtype)
K = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=dtype)
V = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=dtype)


fa_out = sion.flash_attention(Q, K, V)

torch_out = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=0.0)

print("Max absolute difference:", (fa_out - torch_out).abs().max().item())
