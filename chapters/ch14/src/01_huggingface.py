# HuggingFace
from transformers import pipeline

classifier = pipeline("sentiment-analysis")  # 기본값: distilbert-...-sst-2-english

# Test the model
text = "I so happy to see you! I love you!"
result = classifier(text)
print('\n', '==' *30, '\n\n', result)