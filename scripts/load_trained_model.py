from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers.utils import logging

logging.set_verbosity_error()

def get_classifier():
    '''
    Loads tokenizer, trained model, and returns classifier object for easy inference.
    '''
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small", model_max_length=256)
    model = AutoModelForSequenceClassification.from_pretrained("./models/trained_deberta_params")
    classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)
    return classifier