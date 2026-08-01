# %%
import torch
from torch.utils.data import Dataset, DataLoader


# %%

# 합성 데이터 생성
torch.manual_seed(0)
x = torch.arange(0, 100, dtype=torch.float32)
y = 2 * x - 1

class CustomDataset(Dataset):
    def __init__(self, x, y):
        """
        x와 y 값으로 데이터셋을 초기화한다.
        매개변수:
            x (torch.Tensor): 입력 데이터
            y (torch.Tensor): 출력 데이터
        """
        self.x = x
        self.y = y

    def __len__(self):
        """
        데이터셋의 길이를 반환한다.
        """
        return len(self.x)

    def __getitem__(self, idx):
        """
        주어진 인덱스에 해당하는 데이터 샘플을 반환한다.
        매개변수:
            idx (int): 데이터 샘플의 인덱스
        반환값:
            tuple: (x[idx], y[idx]) 형태의 데이터 샘플
        """
        return self.x[idx], self.y[idx]


# %%

# CustomDataset의 인스턴스 생성
dataset = CustomDataset(x, y)

data_loader = DataLoader(dataset, batch_size=10, shuffle=True)

for batch_idx, (inputs, labels) in enumerate(data_loader):
    print(f"배치 {batch_idx + 1}")
    print(f"입력: {inputs}")
    print(f"레이블: {labels}")
    print("-" * 30, '\n')
    # 데모를 위해 첫 번째 배치 후 중지
    if batch_idx == 1:
        break

# %%

# 훈련 성능 향상을 위해 ETL 병렬화하기
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

# 변환 정의
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

# CIFAR10 데이터셋 로드
dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)

# %%
data_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
