import pickle

def load_logreg_model():
    model = pickle.load(open('./models/logreg.pkl', 'rb'))
    return model