# %%
import urllib.request
import zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
val_url = "https://storage.googleapis.com/learning-datasets/validation-horse-or-human.zip"
train_url = "https://storage.googleapis.com/learning-datasets/horse-or-human.zip"

training_dir = "horse-or-human/training/"
validation_dir = "horse-or-human/validation/"

urllib.request.urlretrieve(train_url, "horse-or-human.zip")
urllib.request.urlretrieve(val_url, "validation-horse-or-human.zip")

with zipfile.ZipFile("horse-or-human.zip", "r") as zip_ref:
    zip_ref.extractall(training_dir)

with zipfile.ZipFile("validation-horse-or-human.zip", "r") as zip_ref:
    zip_ref.extractall(validation_dir)

print("Data downloaded and extracted successfully.")


# %%
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# %%
# 변환을 정의
transform = transforms.Compose(
    [
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

# 데이터셋 로드
train_dataset = datasets.ImageFolder(root=training_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=validation_dir, transform=transform)

# 데이터 로더
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# %%
# 말-사람 데이터셋을 위한 CNN 구조

class HorsesHumansCNN(nn.Module):
    def __init__(self):
        super(HorsesHumansCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 18 * 18, 512)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 18 * 18)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


# %%
from torchsummary import summary

model = HorsesHumansCNN()
model.to(device)
summary(model, (3, 150, 150), batch_size=640)

# %%
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_model(num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(images).view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device).float()
                outputs = model(images).view(-1)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(f"Training Accuracy: {100 * correct / total}%")

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()
                outputs = model(images).view(-1)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(f"Validation Accuracy: {100 * correct / total}%")

        print("-" * 50, '\n')

train_model(5)
# %%
