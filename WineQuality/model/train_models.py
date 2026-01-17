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
import pickle
import warnings
warnings.filterwarnings('ignore')

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
pickle.dump(scaler, open('model/scaler.pkl', 'wb'))
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
    
    # Save model
    model_filename = f"model/{model_name.lower().replace(' ', '_')}_model.pkl"
    pickle.dump(model, open(model_filename, 'wb'))
    print(f"Model saved: {model_filename}")

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
print("  - model/scaler.pkl")
print("  - model/*_model.pkl (6 model files)")
