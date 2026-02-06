"""
ML Assignment 2 - Classification Models Implementation
Dataset: Breast Cancer Wisconsin (Diagnostic)
Author: M.Tech AIML Student
"""

import numpy as np
import pandas as pd
import pickle
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
    """Load the Breast Cancer Wisconsin dataset and prepare it for training"""
    print("Loading Breast Cancer Wisconsin Dataset...")
    
    # Load from sklearn's built-in dataset (more reliable, no proxy issues)
    from sklearn.datasets import load_breast_cancer
    
    # Load the dataset
    data = load_breast_cancer()
    
    # Create DataFrame with feature names
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)  # 0 = malignant, 1 = benign (sklearn convention)
    
    # Note: sklearn uses 0=malignant, 1=benign (opposite of UCI)
    # We'll keep this convention for consistency with sklearn
    
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

def save_model_as_pkl(model, model_name, model_type):
    """Save sklearn model as a pickle file"""
    filename = f'model/{model_type}_model.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

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
    
    # Save model as pickle file
    save_model_as_pkl(lr_model, 'Logistic Regression', 'logistic_regression')
    print(f"Logistic Regression - Accuracy: {metrics_lr['Accuracy']:.4f}, AUC: {metrics_lr['AUC']:.4f}\n")
    
    # 2. Decision Tree Classifier
    print("Training Decision Tree...")
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    y_pred_proba_dt = dt_model.predict_proba(X_test)[:, 1]
    metrics_dt = calculate_metrics(y_test, y_pred_dt, y_pred_proba_dt, 'Decision Tree')
    results.append(metrics_dt)
    
    # Save model as pickle file
    save_model_as_pkl(dt_model, 'Decision Tree', 'decision_tree')
    print(f"Decision Tree - Accuracy: {metrics_dt['Accuracy']:.4f}, AUC: {metrics_dt['AUC']:.4f}\n")
    
    # 3. K-Nearest Neighbor Classifier
    print("Training KNN...")
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    y_pred_proba_knn = knn_model.predict_proba(X_test)[:, 1]
    metrics_knn = calculate_metrics(y_test, y_pred_knn, y_pred_proba_knn, 'KNN')
    results.append(metrics_knn)
    
    # Save model as pickle file
    save_model_as_pkl(knn_model, 'KNN', 'knn')
    print(f"KNN - Accuracy: {metrics_knn['Accuracy']:.4f}, AUC: {metrics_knn['AUC']:.4f}\n")
    
    # 4. Naive Bayes Classifier (Gaussian)
    print("Training Naive Bayes...")
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    y_pred_proba_nb = nb_model.predict_proba(X_test)[:, 1]
    metrics_nb = calculate_metrics(y_test, y_pred_nb, y_pred_proba_nb, 'Naive Bayes')
    results.append(metrics_nb)
    
    # Save model as pickle file
    save_model_as_pkl(nb_model, 'Naive Bayes', 'naive_bayes')
    print(f"Naive Bayes - Accuracy: {metrics_nb['Accuracy']:.4f}, AUC: {metrics_nb['AUC']:.4f}\n")
    
    # 5. Random Forest (Ensemble)
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    metrics_rf = calculate_metrics(y_test, y_pred_rf, y_pred_proba_rf, 'Random Forest')
    results.append(metrics_rf)
    
    # Save model as pickle file
    save_model_as_pkl(rf_model, 'Random Forest', 'random_forest')
    print(f"Random Forest - Accuracy: {metrics_rf['Accuracy']:.4f}, AUC: {metrics_rf['AUC']:.4f}\n")
    
    # 6. XGBoost (Ensemble)
    print("Training XGBoost...")
    xgb_model = XGBClassifier(random_state=42, eval_metric='logloss', max_depth=5)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
    metrics_xgb = calculate_metrics(y_test, y_pred_xgb, y_pred_proba_xgb, 'XGBoost')
    results.append(metrics_xgb)
    
    # Save model as pickle file
    save_model_as_pkl(xgb_model, 'XGBoost', 'xgboost')
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
