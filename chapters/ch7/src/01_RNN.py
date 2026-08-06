# %%
import urllib.request
import zipfile

# %%
# GloVe 임베딩 다운로드
url = "https://nlp.stanford.edu/data/glove.6B.zip"
urllib.request.urlretrieve(url, "../data/glove.6B.zip")

# 압축 해제
with zipfile.ZipFile("../data/glove.6B.zip", "r") as zip_ref:
    zip_ref.extractall()

# %%
import numpy as np

glove_embeddings = dict()

with open("../data/")