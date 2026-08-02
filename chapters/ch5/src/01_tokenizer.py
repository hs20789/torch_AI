# %%
import torch

sentences = [
    "Today is a sunny day",
    "Today is a rainy day",
    "Is it sunny today?",
    "I reallly enjoyed walking in the snow today",
]

test_data = [
    "Today is a snowy day",
    "Will it be rainy tomorrow?",
]

# %%
# 사용자 정의 토크나이저
def tokenize(text):
    return text.lower().split()

def build_vocab(sentences):
    vocab = {}
    for sentence in sentences:
        tokens = tokenize(sentence)
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab) + 1

    return vocab

vocab = build_vocab(sentences)
print("어휘 사전 인덱스:", vocab)

def text_to_sequence(text, vocab):
    return [vocab.get(token, 0) for token in tokenize(text)]

text_to_sequence(sentences[3], vocab)

# %% 
# 허깅 페이스에 있는 사전 훈련된 토크나이저

from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
encoded_inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

tokens = [tokenizer.convert_ids_to_tokens(ids)
          for ids in encoded_inputs["input_ids"]]

word_index = tokenizer.get_vocab()

print("토큰:", tokens)
print("토큰 ID:", encoded_inputs["input_ids"])
print("단어 인덱스:", dict(list(word_index.items())[:10]))


# %%
print(type(encoded_inputs["input_ids"]))
print(type(encoded_inputs))