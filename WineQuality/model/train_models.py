"""
Wine Quality Classification - Model Training Script
This script trains 6 different classification models and evaluates them using multiple metrics.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
import warnings
warnings.filterwarnings('ignore')

def save_scaler_as_py(scaler, filename='model/scaler_model.py'):
    """Save StandardScaler as a Python file"""
    mean = scaler.mean_
    scale = scaler.scale_
    var = scaler.var_
    
    code = f'''# StandardScaler Parameters
import numpy as np

mean_ = np.array({mean.tolist()})
scale_ = np.array({scale.tolist()})
var_ = np.array({var.tolist()})

def transform(X):
    """Transform features using stored parameters"""
    X = np.array(X)
    return (X - mean_) / scale_

def inverse_transform(X):
    """Inverse transform scaled features"""
    X = np.array(X)
    return X * scale_ + mean_
'''
    
    with open(filename, 'w') as f:
        f.write(code)
    print(f"Scaler saved to {filename}")

def save_model_as_py(model, model_name, model_type):
    """Save model parameters as a Python file"""
    
    if model_type == 'logistic_regression':
        coef = model.coef_
        intercept = model.intercept_
        classes = model.classes_
        
        code = f'''# Logistic Regression Model Parameters
import numpy as np

classes_ = np.array({classes.tolist()})
coef_ = np.array({coef.tolist()})
intercept_ = np.array({intercept.tolist()})

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
'''
    
    elif model_type == 'decision_tree':
        tree = model.tree_
        n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold
        value = tree.value
        classes = model.classes_
        
        code = f'''# Decision Tree Model Parameters
import numpy as np

classes_ = np.array({classes.tolist()})
children_left = np.array({children_left.tolist()})
children_right = np.array({children_right.tolist()})
feature = np.array({feature.tolist()})
threshold = np.array({threshold.tolist()})
value = np.array({value.tolist()})

def _predict_tree(X_sample):
    """Predict single sample using decision tree"""
    node = 0
    while children_left[node] != children_right[node]:
        if X_sample[feature[node]] <= threshold[node]:
            node = children_left[node]
        else:
            node = children_right[node]
    return value[node]

def predict_proba(X):
    """Predict class probabilities"""
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    proba = np.array([_predict_tree(x) for x in X])
    proba = proba.reshape(proba.shape[0], -1)
    return proba / proba.sum(axis=1, keepdims=True)

def predict(X):
    """Predict class labels"""
    proba = predict_proba(X)
    return classes_[np.argmax(proba, axis=1)]
'''
    
    elif model_type == 'knn':
        X_train = model._fit_X
        y_train = model._y
        classes = model.classes_
        n_neighbors = model.n_neighbors
        
        code = f'''# KNN Model Parameters
import numpy as np

classes_ = np.array({classes.tolist()})
n_neighbors = {n_neighbors}
X_train = np.array({X_train.tolist()})
y_train = np.array({y_train.tolist()})

def predict_proba(X):
    """Predict class probabilities"""
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    probas = []
    for x in X:
        distances = np.sqrt(np.sum((X_train - x) ** 2, axis=1))
        nearest_indices = np.argsort(distances)[:n_neighbors]
        nearest_labels = y_train[nearest_indices]
        proba = np.array([(nearest_labels == c).sum() / n_neighbors for c in classes_])
        probas.append(proba)
    
    return np.array(probas)

def predict(X):
    """Predict class labels"""
    proba = predict_proba(X)
    return classes_[np.argmax(proba, axis=1)]
'''
    
    elif model_type == 'naive_bayes':
        class_prior = model.class_prior_
        theta = model.theta_
        var = model.var_
        classes = model.classes_
        
        code = f'''# Naive Bayes Model Parameters
import numpy as np

classes_ = np.array({classes.tolist()})
class_prior_ = np.array({class_prior.tolist()})
theta_ = np.array({theta.tolist()})
var_ = np.array({var.tolist()})

def predict_proba(X):
    """Predict class probabilities"""
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    log_proba = []
    for i in range(len(classes_)):
        log_prior = np.log(class_prior_[i])
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var_[i]))
        log_likelihood -= 0.5 * np.sum(((X - theta_[i]) ** 2) / var_[i], axis=1)
        log_proba.append(log_prior + log_likelihood)
    
    log_proba = np.array(log_proba).T
    log_proba -= np.max(log_proba, axis=1, keepdims=True)
    proba = np.exp(log_proba)
    return proba / proba.sum(axis=1, keepdims=True)

def predict(X):
    """Predict class labels"""
    proba = predict_proba(X)
    return classes_[np.argmax(proba, axis=1)]
'''
    
    elif model_type == 'random_forest':
        n_estimators = len(model.estimators_)
        trees_data = []
        
        for tree_model in model.estimators_:
            tree = tree_model.tree_
            trees_data.append({
                'children_left': tree.children_left.tolist(),
                'children_right': tree.children_right.tolist(),
                'feature': tree.feature.tolist(),
                'threshold': tree.threshold.tolist(),
                'value': tree.value.tolist()
            })
        
        classes = model.classes_
        
        code = f'''# Random Forest Model Parameters
import numpy as np

classes_ = np.array({classes.tolist()})
n_estimators = {n_estimators}
trees_data = {trees_data}

def _predict_tree(X_sample, tree_data):
    """Predict single sample using one decision tree"""
    node = 0
    children_left = np.array(tree_data['children_left'])
    children_right = np.array(tree_data['children_right'])
    feature = np.array(tree_data['feature'])
    threshold = np.array(tree_data['threshold'])
    value = np.array(tree_data['value'])
    
    while children_left[node] != children_right[node]:
        if X_sample[feature[node]] <= threshold[node]:
            node = children_left[node]
        else:
            node = children_right[node]
    return value[node]

def predict_proba(X):
    """Predict class probabilities"""
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    all_proba = []
    for tree_data in trees_data:
        tree_proba = np.array([_predict_tree(x, tree_data) for x in X])
        tree_proba = tree_proba.reshape(tree_proba.shape[0], -1)
        tree_proba = tree_proba / tree_proba.sum(axis=1, keepdims=True)
        all_proba.append(tree_proba)
    
    mean_proba = np.mean(all_proba, axis=0)
    return mean_proba

def predict(X):
    """Predict class labels"""
    proba = predict_proba(X)
    return classes_[np.argmax(proba, axis=1)]
'''
    
    elif model_type == 'xgboost':
        # Save XGBoost booster as JSON
        model.get_booster().save_model(f'model/{model_type}_booster.json')
        classes = model.classes_
        
        code = f'''# XGBoost Model Parameters
import numpy as np
import json
import os

classes_ = np.array({classes.tolist()})

# Get the directory where this script is located
_model_dir = os.path.dirname(os.path.abspath(__file__))
_booster_path = os.path.join(_model_dir, '{model_type}_booster.json')

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
'''
    
    with open(f'model/{model_type}_model.py', 'w') as f:
        f.write(code)
    print(f"Model saved to model/{model_type}_model.py")

# Load Wine Quality Dataset (Red Wine)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
print("Loading Wine Quality dataset...")
df = pd.read_csv(url, sep=';')

print(f"Dataset shape: {df.shape}")
print(f"Features: {df.columns.tolist()}")
print(f"\nTarget distribution:\n{df['quality'].value_counts().sort_index()}")

# Convert to binary classification (good wine: quality >= 6, bad wine: quality < 6)
# This makes it simpler and allows for AUC calculation
df['quality_class'] = (df['quality'] >= 6).astype(int)

# Features and target
X = df.drop(['quality', 'quality_class'], axis=1)
y = df['quality_class']

print(f"\nBinary Classification - Good Wine (1) vs Bad Wine (0)")
print(f"Class distribution:\n{y.value_counts()}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler and test data
save_scaler_as_py(scaler)
test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
test_df['quality_class'] = y_test.values
test_df.to_csv('model/test_data.csv', index=False)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
}

# Store results
results = []

print("\n" + "="*80)
print("TRAINING AND EVALUATING MODELS")
print("="*80)

for model_name, model in models.items():
    print(f"\n{model_name}:")
    print("-" * 50)
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # Print metrics
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"MCC:       {mcc:.4f}")
    
    # Store results
    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'AUC': auc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'MCC': mcc
    })
    
    # Save model as Python file
    model_type = model_name.lower().replace(' ', '_')
    save_model_as_py(model, model_name, model_type)
    print(f"Model saved: model/{model_type}_model.py")

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('model/model_results.csv', index=False)

print("\n" + "="*80)
print("MODEL COMPARISON TABLE")
print("="*80)
print(results_df.to_string(index=False))

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print("\nFiles saved:")
print("  - model/test_data.csv")
print("  - model/model_results.csv")
print("  - model/scaler_model.py")
print("  - model/*_model.py (6 model files)")
print("  - model/xgboost_booster.json (XGBoost configuration)")
