import fire
from scripts.load_trained_model import get_classifier

def get_review(review: str):
    '''
    This function is used with fire package to turn model into command line tool.
    '''
    classifier = get_classifier()
    return classifier(review)

if __name__ == "__main__":
    fire.Fire(get_review)