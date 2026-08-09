# %%
import torch

def create_sliding_windows(data, window_size, shift=1):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)

    windows = data.unfold(0, window_size, shift)
    features = windows[:, :-1]
    targets = windows[:, -1:]

    return features, targets

# Example Usage
data = torch.arange(10)

features, targets = create_sliding_windows(
    data,
    window_size=5,
    shift=1,
)

for x, y in zip(features, targets):
    print(f"특성: {x.numpy()}, 타겟: {y.numpy()}")

# %%
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(features, targets)
batch_size = 2
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
)

for batch_features, batch_targets in dataloader:
    print(f"배치 특성 크기: {batch_features.shape}")
    print(f"특성:\n{batch_features}\n")
    print(f"타깃:\n{batch_targets}\n")

# %%
x = torch.tensor([1, 2, 3])
print(x)

x = x.unsqueeze(1)
print(x)