# %%
import urllib.request
import zipfile
import torch
import torch.optim as optim

# %%
# GloVe 임베딩 다운로드
url = "https://nlp.stanford.edu/data/glove.6B.zip"
urllib.request.urlretrieve(url, "../data/glove.6B.zip")

# 압축 해제
with zipfile.ZipFile("../data/glove.6B.zip", "r") as zip_ref:
    zip_ref.extractall("../data/")

# %%
import numpy as np

glove_embeddings = dict()

f = open("../data/glove.6B.100d.txt")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype="float32")
    glove_embeddings[word] = coefs
f.close()

# %%
glove_embeddings["samsung"]

# %%

"""
이 임베딩을 신경망 모델의 임베딩 층에 로드하여 임베딩을 처음부터 학습하는 대신
사전 훈련된 임베딩으로 사용할 수 있다.
"""
import torch.nn as nn


class TextClassificationModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=100,
        hidden_dim=16,
        dropout_rate=0.25,
        pretrained_embeddings=None,
        freeze_embeddings=True,
        lstm_layers=2,
    ):
        super().__init__()
        # Embddding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 사전 훈련된 임베딩 로드
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
        )

        # 전역 풀링
        self.global_pooling = nn.AdaptiveAvgPool1d(1)

        # 완전 연결층
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # 드롭아웃
        self.dropout = nn.Dropout(dropout_rate)

        # 활성화 함수
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out.transpose(1, 2)
        pooled = self.global_pooling(lstm_out)
        pooled = pooled.squeeze(-1)
        x = self.relu(self.fc1(pooled))
        x = self.sigmoid(self.fc2(x))
        return x


# %%
from collections import Counter
from typing import List, Dict, Tuple, Union
import re


# Assuming the tokenize function is defined elsewhere
def tokenize(text):
    # Tokenization logic, removing HTML and stopwords as discussed earlier
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text()
    tokens = cleaned_text.lower().split()
    filtered_tokens = [token for token in tokens if token not in stopwords]
    return filtered_tokens


def tokenize_glove_style(text):
    """
    Tokenize text to match GloVe's preprocessing
    """
    # Remove HTML
    text = BeautifulSoup(text, "html.parser").get_text()

    # Convert to lowercase
    text = text.lower()

    # Replace numbers with '0'
    text = re.sub(r"\d", "0", text)

    # Split on whitespace and punctuation
    # GloVe keeps punctuation as separate tokens
    text = re.sub(r"([.,!?()])", r" \1 ", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.split()


def build_vocab(sentences, max_vocab_size=10000):
    counter = Counter()
    for text in sentences:
        counter.update(tokenize(text))

    # Take only the top max_vocab_size-1 most frequent words (leave room for special tokens)
    most_common = counter.most_common(max_vocab_size - 2)  # -2 for  and

    # Create vocabulary with indices starting from 2
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(most_common)}
    vocab[""] = 0  # Add padding token
    vocab[""] = 1  # Add unknown token
    return vocab


# When building vocabulary:
def build_vocab_glove(sentences, max_vocab_size=10000):
    counter = Counter()
    for text in sentences:
        counter.update(tokenize_glove_style(text))

    # Take most frequent words
    most_common = counter.most_common(max_vocab_size - 2)

    # Create vocabulary with indices starting from 2
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(most_common)}
    vocab[""] = 0
    vocab[""] = 1

    return vocab


def texts_to_sequences(sentences, word_index):
    sequences = []
    for sentence in sentences:
        sequence = []
        for word in tokenize_glove_style(sentence):
            # Use unknown token (1) for words not in vocabulary
            sequence.append(word_index.get(word, 1))
        sequences.append(sequence)
    return sequences


def pad_sequences(sequences, max_len):
    padded_sequences = []
    for seq in sequences:
        if len(seq) > max_len:
            padded_seq = seq[:max_len]
        else:
            padded_seq = seq + [0] * (max_len - len(seq))
        padded_sequences.append(padded_seq)
    return padded_sequences


def word_frequency(sentences, word_dict):
    frequency = {word: 0 for word in word_dict}

    for sentence in sentences:
        words = sentence.lower().split()
        for word in words:
            if word in frequency:
                frequency[word] += 1

    return frequency


def word_frequency_glove(sentences, vocab=None):
    """
    Count word frequencies in sentences and return sorted results
    Args:
        sentences: List of sentences
        vocab: Optional vocabulary to filter words (if None, count all words)
    Returns:
        List of tuples (word, count) sorted by count in descending order
    """
    counter = Counter()

    # Count frequencies using the same tokenization
    for sentence in sentences:
        tokens = tokenize_glove_style(sentence)
        counter.update(tokens)

    # If vocab is provided, only keep words in vocab
    if vocab is not None:
        counter = Counter(
            {word: count for word, count in counter.items() if word in vocab}
        )

    # Sort by frequency (descending) and then alphabetically for ties
    sorted_words = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    return sorted_words


# %%
import json


from bs4 import BeautifulSoup
import string

# fmt: off
stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
             "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
             "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
             "he", "hed", "hes", "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how",
             "hows", "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "it", "its", "itself",
             "lets", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought",
             "our", "ours", "ourselves", "out", "over", "own", "same", "she", "shed", "shell", "shes", "should",
             "so", "some", "such", "than", "that", "thats", "the", "their", "theirs", "them", "themselves", "then",
             "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "we", "wed", "well", "were", "weve", "were",
             "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos", "whom", "why",
             "whys", "with", "would", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself",
             "yourselves"]
# fmt: on

table = str.maketrans("", "", string.punctuation)
# %%
# %%
from pathlib import Path
import kagglehub

dataset_path = Path("../data/sarcasm")
downloaded_path = kagglehub.dataset_download(
    "rmisra/news-headlines-dataset-for-sarcasm-detection",
    output_dir=str(dataset_path),
)

# KaggleHub가 반환한 실제 다운로드 폴더
download_dir = Path(downloaded_path)

source_path = download_dir / "Sarcasm_Headlines_Dataset_v2.json"
target_path = download_dir / "sarcasm.json"

if source_path.exists():
    source_path.rename(target_path)
    print("파일 이름 변경 완료:", target_path.resolve())
elif target_path.exists():
    print("이미 이름이 변경되어 있습니다:", target_path.resolve())
else:
    raise FileNotFoundError(f"원본 파일을 찾을 수 없습니다: {source_path}")

sarcasm_path = download_dir / "sarcasm.json"

datastore = []
sentences = []
labels = []
urls = []

with sarcasm_path.open("r", encoding="utf-8") as f:
    for line in f:
        datastore.append(json.loads(line))

for item in datastore:
    sentence = item["headline"].lower()
    sentence = sentence.replace(",", " , ")
    sentence = sentence.replace(".", " . ")
    sentence = sentence.replace("-", " - ")
    sentence = sentence.replace("/", " / ")
    soup = BeautifulSoup(sentence)
    sentence = soup.get_text()
    words = sentence.split()
    filtered_sentence = ""
    for word in words:
        word = word.translate(table)
        if word not in stopwords:
            filtered_sentence = filtered_sentence + word + " "
    sentences.append(filtered_sentence)
    labels.append(item["is_sarcastic"])
    urls.append(item["article_link"])

# %%
vocab_size = 8000
max_length = 80
training_size = 20000


training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

# Updated for vocab size limitation
word_index = build_vocab_glove(training_sentences, max_vocab_size=vocab_size)
training_sequences = texts_to_sequences(training_sentences, word_index)
training_padded = pad_sequences(training_sequences, max_len=max_length)

testing_sequences = texts_to_sequences(testing_sentences, word_index)
testing_padded = pad_sequences(testing_sequences, max_len=max_length)

# word_freq = word_frequency(training_sentences, word_index)
# print(word_freq)

# Example usage:
word_freq = word_frequency_glove(sentences, word_index)

# %%


def create_model(vocab, device="cuda", embedding_dim=50):
    # 1. load_pretrained_embeddings에는 vocab '딕셔너리'를 그대로 전달
    pretrained_embeddings = load_pretrained_embeddings(vocab, embedding_dim)

    # 2. 모델에는 행렬의 실제 행 개수(len(pretrained_embeddings) == len(vocab) + 1)를 전달
    model = TextClassificationModel(
        vocab_size=len(pretrained_embeddings),
        embedding_dim=embedding_dim,
        hidden_dim=16,
        pretrained_embeddings=pretrained_embeddings,
        freeze_embeddings=True,
    ).to(device)

    return model


def load_pretrained_embeddings(vocab, embedding_dim=100):
    """
    Load GloVe embeddings for words in vocabulary
    """

    embeddings_dict = {}
    glove_file = f"../data/glove.6B.{embedding_dim}d.txt"

    # Read GloVe embeddings
    print(f"Loading GloVe embeddings from {glove_file}...")
    with open(glove_file, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings_dict[word] = vector

    # Initialize embedding matrix
    embedding_matrix = np.random.uniform(-0.25, 0.25, (len(vocab) + 1, embedding_dim))

    # Special tokens
    embedding_matrix[0] = np.zeros(embedding_dim)  #

    # Fill with pretrained embeddings
    found_words = 0
    for word, idx in vocab.items():
        if word in embeddings_dict:
            embedding_matrix[idx] = embeddings_dict[word]
            found_words += 1

    print(f"Found embeddings for {found_words}/{len(vocab)} words")
    return torch.FloatTensor(embedding_matrix)


# Create model with GloVe embeddings
model = create_model(
    vocab=word_index,
    device="cuda" if torch.cuda.is_available() else "cpu",
    embedding_dim=100,  # Can be 50, 100, 200, or 300
)


# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(
    model.parameters(), lr=0.00003, betas=(0.9, 0.999), amsgrad=False
)

# %%
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Assuming your TextClassificationModel is already defined

# Convert your preprocessed data to PyTorch tensors
training_padded = torch.tensor(training_padded, dtype=torch.long)
testing_padded = torch.tensor(testing_padded, dtype=torch.long)
training_labels = torch.tensor(training_labels, dtype=torch.float32)
testing_labels = torch.tensor(testing_labels, dtype=torch.float32)

# Create DataLoader objects
batch_size = 32
train_dataset = TensorDataset(training_padded, training_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(testing_padded, testing_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# Training loop
num_epochs = 300
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_total += targets.size(0)
        train_correct += ((outputs.squeeze() > 0.5) == targets).sum().item()

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)

            val_loss += loss.item()
            val_total += targets.size(0)
            val_correct += ((outputs.squeeze() > 0.5) == targets).sum().item()

    print(f"Epoch {epoch + 1}/{num_epochs}:")
    train_loss_history.append(train_loss / len(train_loader))
    train_acc_history.append(train_correct / train_total)
    val_loss_history.append(val_loss / len(test_loader))
    val_acc_history.append(val_correct / val_total)
    print(
        f"Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_correct / train_total:.4f}"
    )
    print(
        f"Val Loss: {val_loss / len(test_loader):.4f}, Val Acc: {val_correct / val_total:.4f}"
    )

# After training, you can save the model
torch.save(model.state_dict(), "../data/text_classification_model.pth")

# %%


import matplotlib.pyplot as plt
import numpy as np


def plot_training_metrics(train_loss, train_acc, val_loss, val_acc):
    """
    Plot training and validation metrics on two subplots.

    Args:
        train_loss: Array of training loss values
        train_acc: Array of training accuracy values
        val_loss: Array of validation loss values
        val_acc: Array of validation accuracy values
    """
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot epochs on x-axis
    epochs = range(1, len(train_loss) + 1)

    # First subplot: Loss
    ax1.plot(epochs, train_loss, "b-", label="Training Loss")
    ax1.plot(epochs, val_loss, "r-", label="Validation Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Second subplot: Accuracy
    ax2.plot(epochs, train_acc, "b-", label="Training Accuracy")
    ax2.plot(epochs, val_acc, "r-", label="Validation Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    # Add accuracy percentage labels
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0%}".format(y)))

    # Adjust layout to prevent overlap
    plt.tight_layout()

    return fig


# Usage example:
plot_training_metrics(
    train_loss_history, train_acc_history, val_loss_history, val_acc_history
)
plt.show()

# %%


def predict_sentences(model, sentences, vocab, max_len, device="cuda", threshold=0.5):
    """
    Make predictions for new sentences and interpret results
    """
    # Preprocess
    sequences = texts_to_sequences(sentences, vocab)
    padded = pad_sequences(sequences, max_len)
    # print(padded)

    # Convert to tensor
    input_ids = torch.tensor(padded, dtype=torch.long).to(device)

    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        print(outputs)
        probabilities = outputs.squeeze().cpu().numpy()
        predictions = (probabilities >= threshold).astype(int)

    # Print results
    for sentence, prob, pred in zip(sentences, probabilities, predictions):
        print(f"\nText: {sentence}")
        print(f"Probability: {prob:.4f}")
        print(f"Classification: {'Sarcastic' if pred == 1 else 'Not Sarcastic'}")
        print("-" * 80)


test_sentences = [
    "It Was, For, Uh, Medical Reasons, Says Doctor To Boris Johnson, Explaining Why They Had To Give Him Haircut",
    "It's a beautiful sunny day",
    "I lived in Ireland, so in high school they made me learn to speak and write in Gaelic",
    "Census Foot Soldiers Swarm Neighborhoods, Kick Down Doors To Tally Household Sizes",
]

# Example usage:
model = model.to(device)
predict_sentences(
    model=model,
    sentences=test_sentences,
    vocab=word_index,
    max_len=max_length,
    threshold=0.5,  # Adjust this threshold if needed
)


# %%
my_sentences = [
    "You just Idiot, you don't know how to use a computer",
    "Hmm... I think this is a great idea!",
    " Thanks a lot. Now I have to walk to the supermarket and carry the groceries.",
    "Sorry I didn't hear my phone ring, I was too busy today not like you, I was working hard and you were just sitting there doing nothing.",
    "Thank you. This is exactly what I needed.",
    "Today is very sunny and warm, perfect for a walk in the park.",
    "Fuck You!!! Itiot!!!!",
    "I can't believe you did that. You're such a genius.",
]


model = model.to(device)
predict_sentences(
    model=model,
    sentences=my_sentences,
    vocab=word_index,
    max_len=max_length,
    threshold=0.5,  # Adjust this threshold if needed
)
