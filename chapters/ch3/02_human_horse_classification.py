# %%
import urllib.request
import zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
url = "https://storage.googleapis.com/learning-datasets/validation-horse-or-human.zip"

file_name = "validation-horse-or-human.zip"
training_dir = "horse-or-human/training/"
validation_dir = "horse-or-human/validation/"
urllib.request.urlretrieve(url, file_name)

with zipfile.ZipFile(file_name, "r") as zip_ref:
    zip_ref.extractall(validation_dir)
    zip_ref.extractall(training_dir)


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


