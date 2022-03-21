from scripts.deep_learning.load_trained_model import get_classifier
from scripts.non_deep_learning.load_trained_model import load_logreg_model
from scripts.non_deep_learning.logreg_inference import logreg_predict
from joblib import load

if __name__ == "__main__":
    classifier = get_classifier()
    logreg_model = load_logreg_model()
    vec = load('./models/vec.bin')
    print()
    while 1:
        sentence = input("Please enter review (type STOP to exit): ")
        if sentence == "STOP":
            break
        print("Deep learning:", classifier(sentence))
        print("Non-Deep learning", logreg_predict(logreg_model, vec, sentence))
        print()