# XGBoost Model Parameters
import numpy as np
import json
import os

classes_ = np.array([0, 1])

# Get the directory where this script is located
_model_dir = os.path.dirname(os.path.abspath(__file__))
_booster_path = os.path.join(_model_dir, 'xgboost_booster.json')

def predict_proba(X):
    """Predict class probabilities - requires xgboost library"""
    try:
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(_booster_path)
        dmatrix = xgb.DMatrix(np.array(X))
        preds = booster.predict(dmatrix)
        return np.column_stack([1 - preds, preds]) if preds.ndim == 1 else preds
    except ImportError:
        raise ImportError("XGBoost library required for predictions")

def predict(X):
    """Predict class labels"""
    proba = predict_proba(X)
    return classes_[(proba[:, 1] > 0.5).astype(int)] if proba.ndim > 1 else classes_[int(proba[1] > 0.5)]
