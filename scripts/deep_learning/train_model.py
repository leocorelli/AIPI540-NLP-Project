from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
import numpy as np
from datasets import load_metric
from load_dataset import get_dataset
from build_features import get_tokenized_dataset
from transformers.utils import logging
import time
import torch

logging.set_verbosity_error()
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def train_model(tokenizer, model, tokenized_dataset, batch_size=16):
    if torch.cuda.is_available():
        fp16 = True
    else:
        fp16 = False

    training_args = TrainingArguments(output_dir=f"models/run{time.time()}", evaluation_strategy="epoch", save_strategy="epoch", num_train_epochs=25, fp16=fp16, per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size, gradient_accumulation_steps=4, load_best_model_at_end=True)
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset['train'], eval_dataset=tokenized_dataset['val'], tokenizer=tokenizer, compute_metrics=compute_metrics, callbacks=[EarlyStoppingCallback(3, 0.0)])
    trainer.train()
    return trainer

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small", model_max_length=256)
    deberta = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-small", num_labels=6)
    tokenized_dataset = get_tokenized_dataset()
    train_model(tokenizer, deberta, tokenized_dataset)