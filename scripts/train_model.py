from scripts.load_dataset import get_dataset
from scripts.build_features import get_tokenized_dataset

from transformers.utils import logging

logging.set_verbosity_error()


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from datasets import load_metric


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def train_model():
    


if __name__ == "__main__":
