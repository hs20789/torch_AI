# %%
import os 
import urllib.request
import tarfile
# %%
def download_and_extract(url, destination):
    if not os.path.exists(destination):
        os.makedirs(destination)
    file_path = os.path.join(destination, "aclImdb_v1.tar.gz")

    if not os.path.exists(file_path):
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, file_path)
        print("Download completed!")

    if "aclImdb" not in os.listdir(destination):
        print("Extracting dataset...")
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=destination)
        print("Extraction completed!")

# 데이터셋 url
dataset_url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
download_and_extract(dataset_url, "../data")

# %%
from collections import Counter

# 간단한 토크나이저
def tokenize(text):
    return text.lower().split()

# build vocabulary
def build_vocab(path):
    counter = Counter()
    for folder in ["pos", "neg"]:
        folder_path = os.path.join(path, folder)
        for filename in os.listdir(folder_path):
            with open(os.path.join(folder_path, filename), 'r',
                                    encoding='utf-8') as file:
                counter.update(tokenize(file.read()))
    return {word: i+1 for i, word in enumerate(counter)}

vocab = build_vocab("../data/aclImdb/train")

# %%
def text_to_sequence(text, vocab):
    return [vocab.get(word, 0) for word in tokenize(text)]

def pad_sequence(sequences, maxlen):
    return [seq + [0] * (maxlen - len(seq))
            if len(seq) < maxlen else seq[:maxlen] for seq in sequences]

# example
text = "This is an example."
seq = text_to_sequence(text, vocab)
print(seq)

padded_seq = pad_sequence([seq], maxlen=50)
print(padded_seq)
# %%
def build_vocab(path):
    counter = Counter()
    for folder in ["pos", "neg"]:
        folder_path = os.path.join(path, folder)
        for filename in os.listdir(folder_path):
            with open(os.path.join(folder_path, filename), 'r',
                                    encoding='utf-8') as file:
                counter.update(tokenize(file.read()))
    sorted_words = sorted(counter.items(), key=lambda x: x[1], reverse=True)

    vocab = {word: idx+1 for idx, (word, _) in enumerate(sorted_words)}
    vocab['<pad>'] = 0
    return vocab

vocab = build_vocab("../data/aclImdb/train")
print(f"Vocabulary size: {len(vocab)}")
print(f"Sample vocabulary: {list(vocab.items())[:20]}")

# %%
from bs4 import BeautifulSoup

stopwords = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "/><br"
    # ... (add more stopwords as needed)
])

def tokenize(text):
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text()
    return [word.lower() for word in cleaned_text.split() 
            if word.lower() not in stopwords]


print(f"Sample vocabulary: {list(vocab.items())[:20]}")
# %%