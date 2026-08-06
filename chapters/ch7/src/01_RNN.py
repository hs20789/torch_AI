# %%
import urllib.request
import zipfile

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

f = open("../data/glove.6B.50d.txt")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype="float32")
    glove_embeddings[word] = coefs
f.close()

# %%
glove_embeddings["samsung"]
