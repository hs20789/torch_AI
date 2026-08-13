from transformers import AutoTokenizer

text = "The ultramarathoner prequalified for the "\
"immunohistochemistry conference in neuroscience."

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize(text)
print(f"토큰: {tokens}")

ids = tokenizer.encode(tokens)
print(f"정수 인코딩: {ids}")

decoded = tokenizer.decode(ids)
print(f"디코딩: {decoded}")