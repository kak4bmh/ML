# Logistic Regression Model Parameters
import numpy as np

classes_ = np.array([0, 1])
coef_ = np.array([[0.295789440924311, -0.5956454454511398, -0.2795659616374921, 0.12332870404702677, -0.17216135147664707, 0.24764158406279588, -0.5669861000127375, -0.15651321794656076, -0.03759442410427558, 0.4916537084493083, 0.8558569175953595]])
intercept_ = np.array([0.2339285377165923])

def predict_proba(X):
    """Predict class probabilities"""
    X = np.array(X)
    logits = np.dot(X, coef_.T) + intercept_
    proba = 1 / (1 + np.exp(-logits))
    return np.column_stack([1 - proba, proba]) if proba.ndim == 1 else np.hstack([1 - proba, proba])

def predict(X):
    """Predict class labels"""
    proba = predict_proba(X)
    return classes_[(proba[:, 1] > 0.5).astype(int)] if proba.ndim > 1 else classes_[int(proba[1] > 0.5)]
