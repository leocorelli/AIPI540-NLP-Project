from transformers import AutoTokenizer
from load_dataset import get_dataset
from transformers.utils import logging

logging.set_verbosity_error()

def tokenize_function(examples):
    '''
    This is the tokenization function that will be applied to dataset.
    '''
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small", model_max_length=256)
    return tokenizer(examples["Review"], padding="max_length", truncation=True)

def get_tokenized_dataset():
    '''
    Pass dataset of string reviews in, get dataset of tokenized reviews out. This is now ready to go to model for training.
    '''
    dataset = get_dataset()
    tokenized_dataset = dataset.map(tokenize_function, batched=True) # tokenize dataset reviews
    return tokenized_dataset