import pandas as pd
from scripts.non_deep_learning.build_features import tokenize

def logreg_predict(model, vec, review):
    df = pd.DataFrame([review])
    df[0] = df[0].apply(lambda x: tokenize(x,method='nltk'))
    features = vec.transform(df[0])
    return model.predict(features)