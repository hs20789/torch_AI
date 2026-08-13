import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import evaluate
import numpy as np
from torch.utils.data import DataLoader


dataset = load_dataset("stanfordnlp/imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
max_length = 512
num_virtual_tokens = 20

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length - num_virtual_tokens,
    )

train_size = 5000
np.random.seed(42)
train_indices = np.random.choice(
    len(dataset["train"]),
    size=train_size,
    replace=False,
)
test_indices = np.random.choice(
    len(dataset["test"]),
    size=train_size,
    replace=False,
)

tokenized_train = dataset["train"].map(tokenize_function, batched=True)
tokenized_test = dataset["test"].map(tokenize_function, batched=True)

tokenized_train = tokenized_train.select(train_indices)
tokenized_test = tokenized_test.select(test_indices)

tokenized_train.set_format(
    type="torch",
    columns=["input_ids",
             "attention_mask",
             "label",
             ],
)
tokenized_test.set_format(
    type="torch",
    columns=["input_ids",
             "attention_mask",
             "label",
             ],
)

train_dataloader = DataLoader(tokenized_train, batch_size=16, shuffle=True)
eval_dataloader = DataLoader(tokenized_test, batch_size=8)

