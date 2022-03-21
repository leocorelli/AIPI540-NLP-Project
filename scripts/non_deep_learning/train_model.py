from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train):
    # Train a classification model using logistic regression classifier
    logreg_model = LogisticRegression(solver='saga')
    logreg_model.fit(X_train,y_train)
    return logreg_model