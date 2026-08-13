def print_step(text):
    print("\n" + "==" * 20)
    print(text)

print_step("1. 설정 및 필요 패키지 불러오기")

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


print_step("2. 데이터 로드 및 검사")
dataset = load_dataset("imdb")
print(f"훈련 세트 크기: {len(dataset['train'])})")
print(f"테스트 세트 크기: {len(dataset['test'])})")


print_step("3.모델과 토크나이저 초기화")

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print_step("4. 데이터 전처리")

def preprocess_function(examples):
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
    )
    result["labels"] = examples["label"]
    return result

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

print_step("5. 데이터 콜레이터 만들기")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print_step("6. 메트릭 정의")
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

print_step("7. 훈련 설정")
from pathlib import Path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent

training_args = TrainingArguments(
    output_dir=str(parent_dir / "results"),
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir=str(parent_dir / "logs"),
    logging_steps=50,
    eval_strategy="epoch",
    save_total_limit=2,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    gradient_accumulation_steps=4,
    report_to="none",
    fp16=torch.cuda.is_available(),
)

print_step("8. Trainer 초기화")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print_step("9. 훈련 및 평가")
train_results = trainer.train()
print(f"훈련 결과: {train_results}")

eval_results = trainer.evaluate()
print(f"평가 결과: {eval_results}")


print_step("10. 모델 저장")
model_save_path = str(parent_dir / "saved_model")
trainer.save_model(model_save_path)


print_step("11. 테스트")
def predict_sentiment(text):
    model.eval()
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    positive_prob = predictions[0][1].item()
    return {
        'sentiment': 'positive' if positive_prob > 0.5 else 'negative',
        'confidence': positive_prob if positive_prob > 0.5 else 1 - positive_prob
    }

print_step("12. 테스트 문장 예측")

# Test prediction
test_text = "This movie was absolutely fantastic! The acting was superb."
result = predict_sentiment(test_text)
print(f"\nTest prediction for '{test_text}':")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2%}")


print_step("13. 모니터링")

def plot_training_history():
    import matplotlib.pyplot as plt

    logs = trainer.state.log_history
    train_steps = [l["step"] for l in logs if "loss" in l]
    train_loss  = [l["loss"]  for l in logs if "loss" in l]
    eval_steps  = [l["step"] for l in logs if "eval_accuracy" in l]
    eval_acc    = [l["eval_accuracy"] for l in logs if "eval_accuracy" in l]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(train_steps, train_loss, label="train loss")
    ax1.set_xlabel("step"); ax1.set_ylabel("loss")

    ax2 = ax1.twinx()
    ax2.plot(eval_steps, eval_acc, "o-", color="tab:orange", label="eval accuracy")
    ax2.set_ylabel("accuracy")

    fig.legend(loc="lower right")
    plt.title("Training History")
    plt.savefig(str(parent_dir / "training_history.png"))
    plt.close()

plot_training_history()
     