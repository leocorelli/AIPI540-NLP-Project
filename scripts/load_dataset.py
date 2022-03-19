import sys
from datasets import load_dataset, DatasetDict

sys.path.append('.')

def get_dataset():
    '''
    Loads dataset from .csv file as a huggingface dataset, and returns dataset with 80%, 10%, 10% train/val/test splits.
    '''
    dataset = load_dataset('csv', data_files='./data/reviews.csv', split='train[:100%]')
    dataset = dataset.rename_column("Label", "labels")

    # create train, val, and test splits
    dataset = dataset.train_test_split(test_size=0.2) # 80% training
    val_and_test = dataset['test'].train_test_split(test_size=0.5) # 10% val, 10% test
    val_and_test['val'] = val_and_test['train']
    dataset = DatasetDict({'train': dataset['train'], 'val': val_and_test['val'], 'test': val_and_test['test']}) # rename to train/val/test

    return dataset