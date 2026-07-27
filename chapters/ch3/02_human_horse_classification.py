# %%
import urllib.request
import zipfile

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
