"""
ML Assignment 2 - Classification Models Implementation
Dataset: Breast Cancer Wisconsin (Diagnostic)
Author: M.Tech AIML Student
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Evaluation Metrics
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

def load_and_prepare_data():
    """Load the Breast Cancer Wisconsin dataset from UCI and prepare it for training"""
    print("Loading Breast Cancer Wisconsin Dataset from UCI...")
    
    # Load from UCI Machine Learning Repository
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    
    # Column names for the dataset
    column_names = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
    
    # Read the dataset
    df = pd.read_csv(url, header=None, names=column_names)
    
    # Drop ID column as it's not a feature
    df = df.drop('ID', axis=1)
    
    # Convert diagnosis to binary (M=1 for Malignant, B=0 for Benign)
    df['target'] = (df['Diagnosis'] == 'M').astype(int)
    df = df.drop('Diagnosis', axis=1)
    
    # Features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    print(f"Dataset Shape: {X.shape}")
    print(f"Number of Features: {X.shape[1]}")
    print(f"Number of Instances: {X.shape[0]}")
    print(f"Class Distribution:\n{pd.Series(y).value_counts()}\n")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save test data for Streamlit app
    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df['target'] = y_test.values
    test_df.to_csv('model/test_data.csv', index=False)
    print("Test data saved to model/test_data.csv")
    
    # Save scaler parameters as Python file
    scaler_code = f"""# Scaler parameters for Breast Cancer Wisconsin dataset
# Generated automatically during model training

import numpy as np

# Feature means (used for scaling)
SCALER_MEAN = {scaler.mean_.tolist()}

# Feature scales (standard deviations)
SCALER_SCALE = {scaler.scale_.tolist()}

# Number of features
N_FEATURES = {len(scaler.mean_)}

def transform(X):
    \"\"\"Transform features using the saved scaler parameters\"\"\"
    X_array = np.array(X)
    return (X_array - SCALER_MEAN) / SCALER_SCALE

def inverse_transform(X_scaled):
    \"\"\"Inverse transform scaled features back to original scale\"\"\"
    X_array = np.array(X_scaled)
    return (X_array * SCALER_SCALE) + SCALER_MEAN
"""
    
    with open('model/scaler.py', 'w') as f:
        f.write(scaler_code)
    print("Scaler parameters saved to model/scaler.py")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

def calculate_metrics(y_true, y_pred, y_pred_proba, model_name):
    """Calculate all required evaluation metrics"""
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_pred_proba),
        'Precision': precision_score(y_true, y_pred, average='binary'),
        'Recall': recall_score(y_true, y_pred, average='binary'),
        'F1': f1_score(y_true, y_pred, average='binary'),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    return metrics

def save_model_as_py(model, model_name, model_type):
    """Save sklearn model as a Python file with parameters"""
    
    if model_type == 'logistic_regression':
        code = f'''# Logistic Regression Model Parameters
import numpy as np

coef_ = np.array({model.coef_.tolist()})
intercept_ = np.array({model.intercept_.tolist()})
classes_ = np.array({model.classes_.tolist()})

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
'''
    
    elif model_type == 'decision_tree':
        code = f'''# Decision Tree Model Parameters
import numpy as np

tree_ = {{
    'feature': {model.tree_.feature.tolist()},
    'threshold': {model.tree_.threshold.tolist()},
    'children_left': {model.tree_.children_left.tolist()},
    'children_right': {model.tree_.children_right.tolist()},
    'value': {model.tree_.value.tolist()}
}}
classes_ = np.array({model.classes_.tolist()})
n_classes_ = {model.n_classes_}

def _predict_single(X, node=0):
    """Predict for a single sample"""
    if tree_['children_left'][node] == tree_['children_right'][node]:  # leaf node
        class_counts = np.array(tree_['value'][node][0])
        return classes_[np.argmax(class_counts)]
    
    if X[tree_['feature'][node]] <= tree_['threshold'][node]:
        return _predict_single(X, tree_['children_left'][node])
    else:
        return _predict_single(X, tree_['children_right'][node])

def predict(X):
    """Predict class labels"""
    X = np.array(X)
    if X.ndim == 1:
        return _predict_single(X)
    return np.array([_predict_single(x) for x in X])

def predict_proba(X):
    """Predict class probabilities"""
    X = np.array(X)
    if X.ndim == 1:
        node = 0
        while tree_['children_left'][node] != tree_['children_right'][node]:
            if X[tree_['feature'][node]] <= tree_['threshold'][node]:
                node = tree_['children_left'][node]
            else:
                node = tree_['children_right'][node]
        class_counts = np.array(tree_['value'][node][0])
        return class_counts / class_counts.sum()
    return np.array([predict_proba(x) for x in X])
'''
    
    elif model_type == 'knn':
        code = f'''# KNN Model Parameters
import numpy as np

X_train_ = np.array({model._fit_X.tolist()})
y_train_ = np.array({model._y.tolist()})
n_neighbors = {model.n_neighbors}
classes_ = np.array({model.classes_.tolist()})

def predict(X):
    """Predict class labels using KNN"""
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    predictions = []
    for x in X:
        distances = np.sqrt(np.sum((X_train_ - x) ** 2, axis=1))
        nearest_indices = np.argsort(distances)[:n_neighbors]
        nearest_labels = y_train_[nearest_indices]
        prediction = np.bincount(nearest_labels).argmax()
        predictions.append(prediction)
    
    return predictions[0] if len(predictions) == 1 else np.array(predictions)

def predict_proba(X):
    """Predict class probabilities"""
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    probas = []
    for x in X:
        distances = np.sqrt(np.sum((X_train_ - x) ** 2, axis=1))
        nearest_indices = np.argsort(distances)[:n_neighbors]
        nearest_labels = y_train_[nearest_indices]
        proba = np.bincount(nearest_labels, minlength=len(classes_)) / n_neighbors
        probas.append(proba)
    
    return probas[0] if len(probas) == 1 else np.array(probas)
'''
    
    elif model_type == 'naive_bayes':
        code = f'''# Naive Bayes Model Parameters
import numpy as np

classes_ = np.array({model.classes_.tolist()})
class_prior_ = np.array({model.class_prior_.tolist()})
theta_ = np.array({model.theta_.tolist()})
var_ = np.array({model.var_.tolist()})

def predict_proba(X):
    """Predict class probabilities"""
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    joint_log_likelihood = []
    for i in range(len(classes_)):
        jointi = np.log(class_prior_[i])
        n_ij = -0.5 * np.sum(np.log(2.0 * np.pi * var_[i, :]))
        n_ij -= 0.5 * np.sum(((X - theta_[i, :]) ** 2) / var_[i, :], axis=1)
        joint_log_likelihood.append(jointi + n_ij)
    
    joint_log_likelihood = np.array(joint_log_likelihood).T
    log_prob_x = np.logaddexp.reduce(joint_log_likelihood, axis=1)
    log_prob = joint_log_likelihood - np.atleast_2d(log_prob_x).T
    return np.exp(log_prob)

def predict(X):
    """Predict class labels"""
    proba = predict_proba(X)
    return classes_[np.argmax(proba, axis=1)] if proba.ndim > 1 else classes_[np.argmax(proba)]
'''
    
    elif model_type == 'random_forest':
        # For Random Forest, save the ensemble structure
        trees_data = []
        for estimator in model.estimators_:
            tree_dict = {
                'feature': estimator.tree_.feature.tolist(),
                'threshold': estimator.tree_.threshold.tolist(),
                'children_left': estimator.tree_.children_left.tolist(),
                'children_right': estimator.tree_.children_right.tolist(),
                'value': estimator.tree_.value.tolist()
            }
            trees_data.append(tree_dict)
        
        code = f'''# Random Forest Model Parameters
import numpy as np

classes_ = np.array({model.classes_.tolist()})
n_estimators = {model.n_estimators}
trees = {trees_data}

def _predict_tree(X, tree, node=0):
    """Predict using a single tree"""
    if tree['children_left'][node] == tree['children_right'][node]:  # leaf
        class_counts = np.array(tree['value'][node][0])
        return class_counts / class_counts.sum()
    
    if X[tree['feature'][node]] <= tree['threshold'][node]:
        return _predict_tree(X, tree, tree['children_left'][node])
    else:
        return _predict_tree(X, tree, tree['children_right'][node])

def predict_proba(X):
    """Predict class probabilities"""
    X = np.array(X)
    if X.ndim == 1:
        probas = np.array([_predict_tree(X, tree) for tree in trees])
        return probas.mean(axis=0)
    return np.array([predict_proba(x) for x in X])

def predict(X):
    """Predict class labels"""
    proba = predict_proba(X)
    if proba.ndim == 1:
        return classes_[np.argmax(proba)]
    return classes_[np.argmax(proba, axis=1)]
'''
    
    elif model_type == 'xgboost':
        # For XGBoost, save the booster as JSON and create loading code
        import json
        
        # Save booster to JSON file
        model.get_booster().save_model(f'model/{model_type}_booster.json')
        
        code = f'''# XGBoost Model Parameters
import numpy as np
import json
import os

classes_ = np.array({model.classes_.tolist()})

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

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train all 6 models and evaluate them"""
    
    results = []
    
    # 1. Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]
    metrics_lr = calculate_metrics(y_test, y_pred_lr, y_pred_proba_lr, 'Logistic Regression')
    results.append(metrics_lr)
    
    # Save model as Python file
    save_model_as_py(lr_model, 'Logistic Regression', 'logistic_regression')
    print(f"Logistic Regression - Accuracy: {metrics_lr['Accuracy']:.4f}, AUC: {metrics_lr['AUC']:.4f}\n")
    
    # 2. Decision Tree Classifier
    print("Training Decision Tree...")
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    y_pred_proba_dt = dt_model.predict_proba(X_test)[:, 1]
    metrics_dt = calculate_metrics(y_test, y_pred_dt, y_pred_proba_dt, 'Decision Tree')
    results.append(metrics_dt)
    
    # Save model as Python file
    save_model_as_py(dt_model, 'Decision Tree', 'decision_tree')
    print(f"Decision Tree - Accuracy: {metrics_dt['Accuracy']:.4f}, AUC: {metrics_dt['AUC']:.4f}\n")
    
    # 3. K-Nearest Neighbor Classifier
    print("Training KNN...")
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    y_pred_proba_knn = knn_model.predict_proba(X_test)[:, 1]
    metrics_knn = calculate_metrics(y_test, y_pred_knn, y_pred_proba_knn, 'KNN')
    results.append(metrics_knn)
    
    # Save model as Python file
    save_model_as_py(knn_model, 'KNN', 'knn')
    print(f"KNN - Accuracy: {metrics_knn['Accuracy']:.4f}, AUC: {metrics_knn['AUC']:.4f}\n")
    
    # 4. Naive Bayes Classifier (Gaussian)
    print("Training Naive Bayes...")
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    y_pred_proba_nb = nb_model.predict_proba(X_test)[:, 1]
    metrics_nb = calculate_metrics(y_test, y_pred_nb, y_pred_proba_nb, 'Naive Bayes')
    results.append(metrics_nb)
    
    # Save model as Python file
    save_model_as_py(nb_model, 'Naive Bayes', 'naive_bayes')
    print(f"Naive Bayes - Accuracy: {metrics_nb['Accuracy']:.4f}, AUC: {metrics_nb['AUC']:.4f}\n")
    
    # 5. Random Forest (Ensemble)
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    metrics_rf = calculate_metrics(y_test, y_pred_rf, y_pred_proba_rf, 'Random Forest')
    results.append(metrics_rf)
    
    # Save model as Python file
    save_model_as_py(rf_model, 'Random Forest', 'random_forest')
    print(f"Random Forest - Accuracy: {metrics_rf['Accuracy']:.4f}, AUC: {metrics_rf['AUC']:.4f}\n")
    
    # 6. XGBoost (Ensemble)
    print("Training XGBoost...")
    xgb_model = XGBClassifier(random_state=42, eval_metric='logloss', max_depth=5)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
    metrics_xgb = calculate_metrics(y_test, y_pred_xgb, y_pred_proba_xgb, 'XGBoost')
    results.append(metrics_xgb)
    
    # Save model as Python file
    save_model_as_py(xgb_model, 'XGBoost', 'xgboost')
    print(f"XGBoost - Accuracy: {metrics_xgb['Accuracy']:.4f}, AUC: {metrics_xgb['AUC']:.4f}\n")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv('model/model_results.csv', index=False)
    print("Results saved to model/model_results.csv")
    
    return results_df

def display_results(results_df):
    """Display results in a formatted table"""
    print("\n" + "="*100)
    print("MODEL COMPARISON TABLE")
    print("="*100)
    print(results_df.to_string(index=False))
    print("="*100)

if __name__ == "__main__":
    print("="*100)
    print("ML ASSIGNMENT 2 - CLASSIFICATION MODELS")
    print("="*100 + "\n")
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data()
    
    # Train and evaluate all models
    results_df = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Display results
    display_results(results_df)
    
    print("\n✓ All models trained and saved successfully!")
    print("✓ Model files saved in 'model/' directory")
    print("✓ Test data saved for Streamlit app")
