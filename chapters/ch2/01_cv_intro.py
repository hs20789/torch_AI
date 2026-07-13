# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# %%
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# %%
class FashionClassifier(nn.Module):
    def __init__(self):
        super(FashionClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = FashionClassifier()

# %%
# 손실 함수와 옵티마이저 정의
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())


# 모델 훈련 함수
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # 예측
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# %%
# 훈련 실행
epochs = 3

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_loader, model, loss_function, optimizer)

print("Done!")


# %%
# 모델 테스트 함수
def test(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_function(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


# %%
# 모델 평가
test(test_loader, model)


# %%
# 정확도 계산 함수
def get_accuracy(pred, labels):
    _, predictions = torch.max(pred, 1)
    correct = (predictions == labels).float().sum()
    accuracy = correct / labels.shape[0]
    return accuracy


# 모델 훈련 함수
def train_v2(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0

    for batch, (X, y) in enumerate(dataloader):
        # 예측
        pred = model(X)
        loss = loss_fn(pred, y)
        total_loss += loss.item()

        accuracy = get_accuracy(pred, y)
        total_accuracy += accuracy.item()

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            current = batch * len(X)
            avg_loss = total_loss / (batch + 1) * 100
            avg_accuracy = total_accuracy / (batch + 1) * 100
            print(
                f"배치 {batch}, 손실: {avg_loss:>7f}, 정확도: {avg_accuracy:>0.2f}% [{current:>5d}/{size:>5d}]"
            )


# %%
# 훈련 실행
epochs = 3
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_v2(train_loader, model, loss_function, optimizer)
print("Done!")
# %%
# 모델 평가
test(test_loader, model)


# %%
import matplotlib.pyplot as plt


def predict_single_image(image, label, model):
    model.eval()

    image = image.unsqueeze(0)
    with torch.no_grad():
        prediction = model(image)
        print(prediction)
        predicted_label = prediction.argmax(1).item()

    # 이미지와 예측 결과를 출력
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"Predicted: {predicted_label}, Actual: {label}")
    plt.show()
    return predicted_label


# %%
image, label = test_dataset[10]

# 선택한 이미지의 레이블을 예측
predicted_label = predict_single_image(image, label, model)
print(f"모델의 예측 {predicted_label}, 실제 레이블: {label}")

# %% 조기 종료


# 모델 훈련 함수
def train_v3(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0

    for batch, (X, y) in enumerate(dataloader):
        # 예측
        pred = model(X)
        loss = loss_fn(pred, y)
        total_loss += loss.item()

        accuracy = get_accuracy(pred, y)
        total_accuracy += accuracy.item()

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            current = batch * len(X)
            avg_loss = total_loss / (batch + 1) * 100
            avg_accuracy = total_accuracy / (batch + 1) * 100
            print(
                f"배치 {batch}, 손실: {avg_loss:>7f}, 정확도: {avg_accuracy:>0.2f}% [{current:>5d}/{size:>5d}]"
            )

        if avg_accuracy > 95.0:
            print("95% 정확도에 도달했으므로 훈련을 중지합니다.")
            return True


# %%
# 훈련 실행
epochs = 30
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_v3(train_loader, model, loss_function, optimizer)
print("Done!")
# %%
# 모델 평가
test(test_loader, model)

# %%
