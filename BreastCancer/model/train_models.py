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
    
    # Save scaler
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
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
    
    # Save model
    with open('model/logistic_regression.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    print(f"Logistic Regression - Accuracy: {metrics_lr['Accuracy']:.4f}, AUC: {metrics_lr['AUC']:.4f}\n")
    
    # 2. Decision Tree Classifier
    print("Training Decision Tree...")
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    y_pred_proba_dt = dt_model.predict_proba(X_test)[:, 1]
    metrics_dt = calculate_metrics(y_test, y_pred_dt, y_pred_proba_dt, 'Decision Tree')
    results.append(metrics_dt)
    
    # Save model
    with open('model/decision_tree.pkl', 'wb') as f:
        pickle.dump(dt_model, f)
    print(f"Decision Tree - Accuracy: {metrics_dt['Accuracy']:.4f}, AUC: {metrics_dt['AUC']:.4f}\n")
    
    # 3. K-Nearest Neighbor Classifier
    print("Training KNN...")
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    y_pred_proba_knn = knn_model.predict_proba(X_test)[:, 1]
    metrics_knn = calculate_metrics(y_test, y_pred_knn, y_pred_proba_knn, 'KNN')
    results.append(metrics_knn)
    
    # Save model
    with open('model/knn.pkl', 'wb') as f:
        pickle.dump(knn_model, f)
    print(f"KNN - Accuracy: {metrics_knn['Accuracy']:.4f}, AUC: {metrics_knn['AUC']:.4f}\n")
    
    # 4. Naive Bayes Classifier (Gaussian)
    print("Training Naive Bayes...")
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    y_pred_proba_nb = nb_model.predict_proba(X_test)[:, 1]
    metrics_nb = calculate_metrics(y_test, y_pred_nb, y_pred_proba_nb, 'Naive Bayes')
    results.append(metrics_nb)
    
    # Save model
    with open('model/naive_bayes.pkl', 'wb') as f:
        pickle.dump(nb_model, f)
    print(f"Naive Bayes - Accuracy: {metrics_nb['Accuracy']:.4f}, AUC: {metrics_nb['AUC']:.4f}\n")
    
    # 5. Random Forest (Ensemble)
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    metrics_rf = calculate_metrics(y_test, y_pred_rf, y_pred_proba_rf, 'Random Forest')
    results.append(metrics_rf)
    
    # Save model
    with open('model/random_forest.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"Random Forest - Accuracy: {metrics_rf['Accuracy']:.4f}, AUC: {metrics_rf['AUC']:.4f}\n")
    
    # 6. XGBoost (Ensemble)
    print("Training XGBoost...")
    xgb_model = XGBClassifier(random_state=42, eval_metric='logloss', max_depth=5)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
    metrics_xgb = calculate_metrics(y_test, y_pred_xgb, y_pred_proba_xgb, 'XGBoost')
    results.append(metrics_xgb)
    
    # Save model
    with open('model/xgboost.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
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


