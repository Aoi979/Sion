import torch
import sion

print(sion.__doc__)

M, K, N = 433, 466, 477

device = 'cuda'
A = torch.randn(M, K, dtype=torch.float32, device=device)
B = torch.randn(K, N, dtype=torch.float32, device=device)

alpha = 0.4
beta = 0.3  

C = sion.sgemm(A, B, alpha, beta)

print("A @ B using libpysion.sgemm on CUDA:")
print(C)

C_ref = alpha * torch.matmul(A, B)  # beta*C0 = 0
print("\nReference torch.matmul result on CUDA:")
print(C_ref)

max_error = (C - C_ref).abs().max()
print(f"\nMax absolute error: {max_error}")
