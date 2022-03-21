import string
import nltk
from nltk.stem import WordNetLemmatizer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

nlp = spacy.load('en_core_web_sm')
nltk.download('wordnet')
nltk.download('omw-1.4')

def tokenize(sentence,method='spacy'):
# Tokenize and lemmatize text, remove stopwords and punctuation

    punctuations = string.punctuation
    stopwords = list(STOP_WORDS)

    if method=='nltk':
        # Tokenize
        tokens = nltk.word_tokenize(sentence,preserve_line=True)
        # Remove stopwords and punctuation
        tokens = [word for word in tokens if word not in stopwords and word not in punctuations]
        # Lemmatize
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = [wordnet_lemmatizer.lemmatize(word) for word in tokens]
        tokens = " ".join([i for i in tokens])
    else:
        # Tokenize
        with nlp.select_pipes(enable=['tokenizer','lemmatizer']):
            tokens = nlp(sentence)
        # Lemmatize
        tokens = [word.lemma_.lower().strip() for word in tokens]
        # Remove stopwords and punctuation
        tokens = [word for word in tokens if word not in stopwords and word not in punctuations]
        tokens = " ".join([i for i in tokens])
    return tokens

def build_features(train_df, test_df, ngram_range, method='count'):
    if method == 'tfidf':
        # Create features using TFIDF
        vec = TfidfVectorizer(ngram_range=ngram_range)
        X_train = vec.fit_transform(train_df['processed_text'])
        X_test = vec.transform(test_df['processed_text'])

    else:
        # Create features using word counts
        vec = CountVectorizer(ngram_range=ngram_range)
        X_train = vec.fit_transform(train_df['processed_text'])
        X_test = vec.transform(test_df['processed_text'])

    return X_train, X_test, vec


# from load_dataframe import load_dataframe
# from train_model import train_model
# from joblib import dump
# import pickle

# if __name__ == "__main__":
#     X_train, X_test, y_train, y_test = load_dataframe()
#     tqdm.pandas()
#     train_df = X_train
#     test_df = X_test
#     train_df['processed_text'] = train_df.progress_apply(lambda x: tokenize(x,method='nltk'))
#     test_df['processed_text'] = test_df.progress_apply(lambda x: tokenize(x,method='nltk'))
#     # create features
#     method = 'tfidf'
#     ngram_range = (1, 2)
#     X_train,X_test,vec = build_features(train_df,test_df,ngram_range,method)
#     #dump(vec, './models/vec.bin', compress=True)

#     model = train_model(X_train, y_train)
#     #pickle.dump(model, open("./models/logreg.pkl", 'wb'))

#     test_preds = model.predict(X_test)
#     test_acc = sum(test_preds==y_test)/len(y_test)
#     print('Accuracy on the test set is {:.3f}'.format(test_acc))



