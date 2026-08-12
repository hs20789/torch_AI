import torch
import torch.nn as nn

d_model = 2
d_ff = 4

x = torch.tensor([[-1.0, 2.0]])

W1 = torch.tensor([
    [1.0, -1.0],
    [-1.0, 1.0],
    [0.5, 0.5],
    [-0.5, -0.5],
])
b1 = torch.tensor([0.0, 0.0, 0.0, 0.0])
layer1_out = torch.matmul(x, W1.t()) + b1
print(f"첫 번째 선형 층을 적용한 후: {layer1_out}")

# ReLU 적용
relu_out = torch.relu(layer1_out)
print(f"ReLU를 적용한 후: {relu_out}")
# 음수는 0이 되므로 비선형 연산이다.

W2 = torch.tensor([
    [1.0, -1.0, 0.5, -0.5],
    [-1.0, 1.0, 0.5, -0.5],
])
b2 = torch.tensor([0.0, 0.0])
final_out = torch.matmul(relu_out, W2.t()) + b2
print(f"최종 출력: {final_out}")


print('\n\n', '=='*30, '\n\n')
print(f"W1.shape: {W1.shape}, b1.shape: {b1.shape}")
print(f"W2.shape: {W2.shape}, b2.shape: {b2.shape}")