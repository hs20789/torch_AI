# %%
import torch
from transformers import BertTokenizer, BertModel


# %%
texts = [
    "I love my dog.",
    "The Manatee became a doctor.",
]


def text_to_embeddings(texts):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()

    # fmt: off
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    # fmt: on

    # Generate embeddings
    with torch.no_grad():  # No need to calculate gradients
        outputs = model(**encoded)
        embeddings = outputs.last_hidden_state

    return embeddings, encoded["input_ids"]


# Get embeddings
embeddings, token_ids = text_to_embeddings(texts)

# Print information about the embeddings
print(f"Input texts: {texts}")
print(f"Encodings: {token_ids}")
print(f"\nEmbedding tensor shape: {embeddings.shape}")
print("Shape explanation:")
print(f"- Number of sentences: {embeddings.shape[0]}")
print(f"- Words per sentence: {embeddings.shape[1]}")
print(f"- Embedding dimensions: {embeddings.shape[2]}")

print("*" * 30)
print(f"Embedding: {embeddings}")


# %%

