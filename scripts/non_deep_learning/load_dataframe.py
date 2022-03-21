import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataframe():
    '''
    Reads in csv file and returns train/test splits
    '''
    df = pd.read_csv("./data/reviews.csv")
    X = df['Review']
    y = df['Label'] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    return X_train, X_test, y_train, y_test