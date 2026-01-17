"""
ML Assignment 2 - Streamlit Web Application
Interactive Classification Model Comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import importlib.util
import sys
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

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

# Page configuration
st.set_page_config(
    page_title="ML Classification Models",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 Machine Learning Classification Model Comparison")
st.markdown("---")
st.markdown("""
This application demonstrates the performance of **6 different classification models** 
on the **Breast Cancer Wisconsin (Diagnostic)** dataset.

**Models Implemented:**
- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors (KNN)
- Naive Bayes (Gaussian)
- Random Forest (Ensemble)
- XGBoost (Ensemble)
""")

# Sidebar for model selection
st.sidebar.header("🎯 Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose a Model:",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

# Model file mapping
model_files = {
    "Logistic Regression": "logistic_regression_model",
    "Decision Tree": "decision_tree_model",
    "KNN": "knn_model",
    "Naive Bayes": "naive_bayes_model",
    "Random Forest": "random_forest_model",
    "XGBoost": "xgboost_model"
}

@st.cache_resource
def load_model(model_name):
    """Load the trained model from .py file"""
    try:
        model_path = MODEL_DIR / f"{model_name}.py"
        spec = importlib.util.spec_from_file_location(model_name, model_path)
        model_module = importlib.util.module_from_spec(spec)
        sys.modules[model_name] = model_module
        spec.loader.exec_module(model_module)
        return model_module
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_precomputed_results():
    """Load precomputed model results"""
    try:
        results_df = pd.read_csv(MODEL_DIR / "model_results.csv")
        return results_df
    except Exception as e:
        st.warning(f"Precomputed results not found: {e}")
        return None

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate all evaluation metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC Score': roc_auc_score(y_true, y_pred_proba),
        'Precision': precision_score(y_true, y_pred, average='binary'),
        'Recall': recall_score(y_true, y_pred, average='binary'),
        'F1 Score': f1_score(y_true, y_pred, average='binary'),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    return metrics

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Malignant', 'Benign'],
                yticklabels=['Malignant', 'Benign'],
                ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    
    return fig

def display_classification_report(y_true, y_pred):
    """Display classification report"""
    report = classification_report(y_true, y_pred, 
                                   target_names=['Malignant', 'Benign'],
                                   output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    return report_df

# Main content
st.markdown("---")

# Dataset Upload Section
st.header("📊 Dataset Upload")
st.markdown("Upload your test dataset (CSV format) or use the provided test data.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # User uploaded file
    try:
        data = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
        
        st.subheader("Dataset Preview")
        st.dataframe(data.head(10))
        
        st.write(f"**Shape:** {data.shape[0]} rows × {data.shape[1]} columns")
        
        # Check if target column exists
        if 'target' in data.columns:
            X = data.drop('target', axis=1)
            y = data['target']
            
            # Load selected model
            model_path = model_files[model_choice]
            model = load_model(model_path)
            
            if model is not None:
                st.markdown("---")
                st.header(f"📈 Model Performance: {model_choice}")
                
                # Make predictions
                y_pred = model.predict(X.values)
                y_pred_proba_full = model.predict_proba(X.values)
                y_pred_proba = y_pred_proba_full[:, 1] if y_pred_proba_full.ndim > 1 else y_pred_proba_full[1]
                
                # Calculate metrics
                metrics = calculate_metrics(y, y_pred, y_pred_proba)
                
                # Display metrics
                st.subheader("🎯 Evaluation Metrics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                    st.metric("Precision", f"{metrics['Precision']:.4f}")
                
                with col2:
                    st.metric("AUC Score", f"{metrics['AUC Score']:.4f}")
                    st.metric("Recall", f"{metrics['Recall']:.4f}")
                
                with col3:
                    st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
                    st.metric("MCC", f"{metrics['MCC']:.4f}")
                
                # Confusion Matrix
                st.markdown("---")
                st.subheader("🔍 Confusion Matrix")
                fig = plot_confusion_matrix(y, y_pred, model_choice)
                st.pyplot(fig)
                
                # Classification Report
                st.markdown("---")
                st.subheader("📋 Classification Report")
                report_df = display_classification_report(y, y_pred)
                st.dataframe(report_df.style.highlight_max(axis=0, color='lightgreen'))
        else:
            st.error("❌ The uploaded file must contain a 'target' column!")
            
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    # Use default test data if available
    st.info("ℹ️ No file uploaded. Loading default test data...")
    
    try:
        data = pd.read_csv(MODEL_DIR / "test_data.csv")
        
        st.subheader("Dataset Preview (Default Test Data)")
        st.dataframe(data.head(10))
        
        st.write(f"**Shape:** {data.shape[0]} rows × {data.shape[1]} columns")
        
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Load selected model
        model_name = model_files[model_choice]
        model = load_model(model_name)
        
        if model is not None:
            st.markdown("---")
            st.header(f"📈 Model Performance: {model_choice}")
            
            # Make predictions
            y_pred = model.predict(X.values)
            y_pred_proba_full = model.predict_proba(X.values)
            y_pred_proba = y_pred_proba_full[:, 1] if y_pred_proba_full.ndim > 1 else y_pred_proba_full[1]
            
            # Calculate metrics
            metrics = calculate_metrics(y, y_pred, y_pred_proba)
            
            # Display metrics
            st.subheader("🎯 Evaluation Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                st.metric("Precision", f"{metrics['Precision']:.4f}")
            
            with col2:
                st.metric("AUC Score", f"{metrics['AUC Score']:.4f}")
                st.metric("Recall", f"{metrics['Recall']:.4f}")
            
            with col3:
                st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
                st.metric("MCC", f"{metrics['MCC']:.4f}")
            
            # Confusion Matrix
            st.markdown("---")
            st.subheader("🔍 Confusion Matrix")
            fig = plot_confusion_matrix(y, y_pred, model_choice)
            st.pyplot(fig)
            
            # Classification Report
            st.markdown("---")
            st.subheader("📋 Classification Report")
            report_df = display_classification_report(y, y_pred)
            st.dataframe(report_df.style.highlight_max(axis=0, color='lightgreen'))
            
    except FileNotFoundError:
        st.warning("⚠️ Default test data not found. Please upload a CSV file.")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# Model Comparison Section
st.markdown("---")
st.header("📊 All Models Comparison")

results_df = load_precomputed_results()

if results_df is not None:
    st.subheader("Performance Comparison Table")
    st.dataframe(
        results_df.style.highlight_max(axis=0, subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'], color='lightgreen')
        .format({
            'Accuracy': '{:.4f}',
            'AUC': '{:.4f}',
            'Precision': '{:.4f}',
            'Recall': '{:.4f}',
            'F1': '{:.4f}',
            'MCC': '{:.4f}'
        })
    )
    
    # Bar charts for comparison
    st.subheader("📊 Visual Comparison")
    
    metrics_to_plot = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        results_df.plot(x='Model', y=metric, kind='bar', ax=ax, color='skyblue', legend=False)
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.info("Run the training script first to generate comparison results.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><b>ML Assignment 2 - M.Tech AIML</b></p>
    <p>Classification Model Comparison on Breast Cancer Wisconsin Dataset</p>
</div>
""", unsafe_allow_html=True)


