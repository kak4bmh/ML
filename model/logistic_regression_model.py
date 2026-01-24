# Logistic Regression Model Parameters
import numpy as np

coef_ = np.array([[0.36115006844913494, 0.48221940329617796, 0.35315986626691465, 0.43995028198850433, 0.350621558291717, -0.43954614808744324, 0.7822983533225191, 0.9528128318119833, -0.16399086837706922, -0.08086512047777301, 1.2333251663808185, -0.4076112582620659, 0.7482947115713602, 0.9090290582882155, 0.2479909817163055, -0.9069248024192642, -0.09234069448239378, 0.48208907859235295, -0.33065773405654114, -0.5938763200364329, 0.8969678339058819, 1.4340931710338372, 0.7231114802873427, 0.9004766112545058, 0.4202067546569991, -0.1734875080449964, 0.9114058021854962, 0.7039988067031911, 1.061263662170949, 0.05486987894732853]])
intercept_ = np.array([-0.24300532690560292])
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
