# %%
with open("../data/sing.txt", "r") as f:
    data = f.read()
print(data)

# %%
from helper import (
    tokenize,
    create_word_dictionary,
    text_to_sequence,
    pad_sequence,
    split_sequences,
    one_hot_encode_with_checks,
)
# %%
tokens = tokenize(data)
word_index = create_word_dictionary(tokens)
total_words = len(word_index)
print(f"Tokens: {tokens}\nTotal Words: {total_words}\nWord Index: {word_index}")

# %%
corpus = data.lower().split("\n")
print(f"Corpus: {corpus}")
input_sequences = []
for line in corpus:
    token_list = text_to_sequence(line, word_index)
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequence(input_sequences, max_sequence_len)

print(f"Input Sequences: {input_sequences}\n\
Max Sequence Length: {max_sequence_len}\n\
Input Sequences Lengths: {input_sequences}")

# %%
xs, labels = split_sequences(input_sequences)
print(f"xs: {xs}\nlabels: {labels}")
vocab_size = len(word_index)
ys = []
for y in labels:
  ys.append(one_hot_encode_with_checks(y[0], vocab_size))
print(f"ys[0]: {ys[0]}\nys Length: {len(ys)}\nys[0] Length: {len(ys[0])}")


# %%
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# %%
class LSTMPredictor(nn.Module):
    def __init__(self, total_words, embedding_dim=8, hidden_dim=None):
        super().__init__()

        # hidden_dim이 지정되지 않으면 텐서플로 버전처럼 max_sequence_len-1을 사용한다.
        if hidden_dim is None:
            hidden_dim = max_sequence_len - 1

        self.embedding = nn.Embedding(total_words, embedding_dim)

        # 양방향 LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True,
        )

        # 최종 밀집 층 (양방향 LSTM의 출력 고려)
        self.fc = nn.Linear(hidden_dim * 2, total_words)

        # 소프트맥스 활성화 함수
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # 마지막 시퀀스의 출력만 사용
        x = self.fc(x)
        x = self.softmax(x)
        return x
# %%
total_words = len(word_index)
model = LSTMPredictor(total_words).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())


# Convert data to PyTorch tensors
# Assuming xs and ys are numpy arrays
xs_tensor = torch.LongTensor(xs)
ys_tensor = torch.FloatTensor(ys)


# Lists to store metrics
train_losses = []
train_accuracies = []

# Training loop with accuracy tracking
num_epochs = 15000
model.train()

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(xs_tensor)
    loss = criterion(outputs, ys_tensor)

    # Calculate accuracy
    _, predicted = torch.max(outputs.data, 1)
    _, targets = torch.max(ys_tensor, 1)

    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    accuracy = 100 * correct / total

    # Store metrics
    train_losses.append(loss.item())
    train_accuracies.append(accuracy)

    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Loss: {loss.item():.4f}, '
              f'Accuracy: {accuracy:.2f}%')

# %%
import matplotlib.pyplot as plt
# Plot training metrics
plt.figure(figsize=(12, 4))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies)
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')

plt.tight_layout()
plt.show()

# Print final metrics
print(f'\nFinal Results:')
print(f'Loss: {train_losses[-1]:.4f}')
print(f'Accuracy: {train_accuracies[-1]:.2f}%')


# %%
seed_text = "In the town of Athy"

words = seed_text.lower().strip().split()
number_sequence = [word_index.get(word, 0) for word in words]
padded_sequence = pad_sequence([number_sequence], max_sequence_len)
input_tensor = torch.LongTensor(padded_sequence).to(device)

with torch.no_grad():
    output = model(input_tensor)

# 예측된 (가장 높은 확률을 가진) 단어 인덱스를 구한다.
predicted_index = torch.argmax(output[0]).item()
print(f"Predicted Index: {predicted_index}\nPredicted Word: {list(word_index.keys())[list(word_index.values()).index(predicted_index)]}")

# %%
def generate_sequence(model, initial_text, word_index, sequence_length, num_words=10):

    # 모델을 평가 모드로 전화
    model.eval()

    # 초기 텍스트
    current_text = initial_text
    generated_sequence = initial_text

    # 인덱스를 단어로 바꾸기 위해 역 딕셔너리를 생성
    reverse_dict = {v: k for k, v in word_index.items()}

    print(f"초기 텍스트: {initial_text}")

    for i in range(num_words):
        # 텍스트를 소문자로 바꾸고 단어로 분할
        words = current_text.lower().strip().split()

        # 이가 초과되면 마지막 sequence_length 개 단어를 선택
        if len(words) > sequence_length:
            words = words[-sequence_length:]

        # 단어 딕셔너리를 사용해 단어를 숫자로 변환, 0은 알지 못하는 단어이다.
        number_sequence = [word_index.get(word, 0) for word in words]
        # 시퀀스를 패딩한다.
        padded_sequence = pad_sequence([number_sequence], sequence_length)

        # 텐서로 변환
        input_tensor = torch.LongTensor(padded_sequence).to(device)

        # 예측
        with torch.no_grad():
            output = model(input_tensor)

        # 예측된 단어 인덱스를 구한다.
        predicted_index = torch.argmax(output[0]).item()
        predicted_word = reverse_dict.get(predicted_index, "UNK")

        # 예측된 단어를 시퀀스에 추가
        generated_sequence += " " + predicted_word

        # 다음 예측을 위해 현재 텍스트를 업데이트
        current_text = generated_sequence

        # 진행 과정을 출력
        print(f"생성된 단어 {i+1}: {predicted_word} (인덱스: {predicted_index})")

        # 각 단계마다 상위 다섯 개 예측을 출력
        _, top_indices = torch.topk(output[0], 5)
        print(f"\n단계 {i+1}에서의 상위 5개 예측:")
        for idx in top_indices:
            word = reverse_dict.get(idx.item(), "UNK")
            prob = output[0][idx].item()
            print(f"단어: {word}, 확률: {prob:.4f}")
        print("\n" + "-"*50 + "\n")

    return generated_sequence


# %%
# 사용 예

initial_text = "HeonSu"
generated_sequence = generate_sequence(
    model=model,
    initial_text=initial_text,
    word_index=word_index,
    sequence_length=max_sequence_len,
    num_words=10,
)

print(f"최종 생성된 시퀀스: {generated_sequence}")