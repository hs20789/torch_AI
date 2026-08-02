# %%
import kagglehub
from pathlib import Path

dataset_path = Path("../data/sarcasm")

# Download latest version
path = kagglehub.dataset_download(
    "rmisra/news-headlines-dataset-for-sarcasm-detection",
    output_dir=str(dataset_path),)

print("Path to dataset files:", path)

# %%
import json

with open("../data/sarcasm/sarcasm.json", 'r') as f:
    )