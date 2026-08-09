import torch
import torch.nn as nn
import torch.optim as optim
from linear import SimpleLinearModel

from pathlib import Path


def train_model():
    model = SimpleLinearModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # fmt: off
    xs = torch.tensor([[-1.0], [0.0], [1.0], [2.0], [3.0], [4.0]],
                      dtype=torch.float32)
    ys = torch.tensor([[-3.0], [-1.0], [1.0], [3.0], [5.0], [7.0]],
                      dtype=torch.float32)
    # fmt: on

    for _ in range(500):
        optimizer.zero_grad()
        outputs = model(xs)
        loss = criterion(outputs, ys)
        loss.backward()
        optimizer.step()

    return model


model = train_model()
model_path = Path(__file__).resolve().parent.parent / "model.pth"
torch.save(model.state_dict(), model_path)
