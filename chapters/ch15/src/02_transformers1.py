# Using Hugging Face Transformers Library
import os
from transformers import pipeline

# 기본 감성 분석
classifier = pipeline("sentiment-analysis")
result = classifier("I love working with transformers!")
print(result)
