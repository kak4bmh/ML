# Logistic Regression Model Parameters
import numpy as np

coef_ = np.array([[0.3611500684491406, 0.4822194032961759, 0.35315986626691875, 0.43995028198850633, 0.3506215582917179, -0.4395461480874417, 0.7822983533225173, 0.9528128318119848, -0.16399086837707139, -0.0808651204777729, 1.2333251663808191, -0.4076112582620678, 0.7482947115713618, 0.9090290582882133, 0.2479909817163083, -0.906924802419269, -0.09234069448239254, 0.4820890785923534, -0.33065773405654036, -0.5938763200364356, 0.8969678339058851, 1.4340931710338327, 0.7231114802873447, 0.9004766112545047, 0.42020675465699103, -0.17348750804500165, 0.9114058021854936, 0.7039988067031971, 1.0612636621709532, 0.05486987894733817]])
intercept_ = np.array([-0.24300532690560606])
classes_ = np.array([0, 1])

def predict_proba(X):
    """Predict class probabilities"""
    X = np.array(X)
    logits = np.dot(X, coef_.T) + intercept_
    proba = 1 / (1 + np.exp(-logits))
    return np.column_stack([1 - proba, proba]).flatten() if X.ndim == 1 else np.column_stack([1 - proba, proba])

def predict(X):
    """Predict class labels"""
    proba = predict_proba(X)
    if len(proba.shape) == 1:
        return classes_[1] if proba[1] > 0.5 else classes_[0]
    return classes_[(proba[:, 1] > 0.5).astype(int)]
