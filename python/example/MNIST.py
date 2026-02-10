import torch
import sion  


batch_size = 16
input_dim = 28 * 28
hidden_dim = 128
output_dim = 10

# 随机输入
x = torch.randn(batch_size, input_dim, device='cuda')

# 随机权重
W1 = torch.randn(input_dim, hidden_dim, device='cuda')
b1 = torch.randn(hidden_dim, device='cuda')
W2 = torch.randn(hidden_dim, output_dim, device='cuda')
b2 = torch.randn(output_dim, device='cuda')

# -----------------------------
# 前向传播
# -----------------------------
# 第一层: x -> hidden
hidden = sion.sgemm(x, W1, alpha=1.0, beta=0.0)
hidden += b1  # 偏置
hidden = torch.relu(hidden)

# 第二层: hidden -> logits
logits = sion.sgemm(hidden, W2, alpha=1.0, beta=0.0)
logits += b2

# softmax 可选
probs = torch.softmax(logits, dim=1)

# -----------------------------
# 打印输出
# -----------------------------
print("logits shape:", logits.shape)
print("probs shape:", probs.shape)
print("probs[0]:", probs[0])
