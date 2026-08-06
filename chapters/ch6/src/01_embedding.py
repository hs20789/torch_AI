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

# %%
import json
from bs4 import BeautifulSoup

datastore = []    
sentences = []
labels = []
urls = []

with sarcasm_path.open('r', encoding='utf-8') as f:
    for line in f:
        datastore.append(json.loads(line))

for item in datastore:
    sentences.append(item['headline'].lower())
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

training_size = 24000
training_sentences = sentences[:training_size]
training_labels = labels[:training_size]
testing_sentences = sentences[training_size:]
testing_labels = labels[training_size:]
