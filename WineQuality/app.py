"""
ML Assignment 2 - Streamlit Web Application
Wine Quality Classifiction Model Comparison
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

# Page configuration
st.set_page_config(
    page_title="Wine Quality Classification",
    page_icon="🍷",
    layout="wide"
)

# Title and description
st.title("🍷 Wine Quality Classification App")
st.markdown("""
This application demonstrates **6 Machine Learning Classification Models** trained on the Wine Quality dataset.
Upload test data and select a model to see predictions and evaluation metrics.
""")

st.divider()

# Sidebar for model selection
st.sidebar.header("Model Configuration")

# Model selection dropdown
model_options = {
    'Logistic Regression': 'logistic_regression_model',
    'Decision Tree': 'decision_tree_model',
    'KNN': 'knn_model',
    'Naive Bayes': 'naive_bayes_model',
    'Random Forest': 'random_forest_model',
    'XGBoost': 'xgboost_model'
}

selected_model_name = st.sidebar.selectbox(
    "Select Classification Model",
    options=list(model_options.keys()),
    help="Choose a model to make predictions and view metrics"
)

st.sidebar.divider()
st.sidebar.info("""
**Dataset Features:**
- Fixed Acidity
- Volatile Acidity
- Citric Acid
- Residual Sugar
- Chlorides
- Free Sulfur Dioxide
- Total Sulfur Dioxide
- Density
- pH
- Sulphates
- Alcohol

**Target:** Wine Quality (Binary)
- 0: Bad Wine (quality < 6)
- 1: Good Wine (quality >= 6)
""")

# File upload section
st.header("📂 Upload Test Data")
uploaded_file = st.file_uploader(
    "Upload CSV file with test data",
    type=['csv'],
    help="Upload a CSV file containing wine features and quality_class column"
)

if uploaded_file is not None:
    try:
        # Load the uploaded data
        test_data = pd.read_csv(uploaded_file)
        
        st.success(f"✅ File uploaded successfully! Shape: {test_data.shape}")
        
        # Show data preview
        with st.expander("📊 View Data Preview"):
            st.dataframe(test_data.head(10), use_container_width=True)
            st.write(f"**Total samples:** {len(test_data)}")
            st.write(f"**Features:** {test_data.shape[1] - 1}")
        
        # Check if quality_class column exists
        if 'quality_class' not in test_data.columns:
            st.error("❌ The uploaded file must contain a 'quality_class' column!")
            st.stop()
        
        # Separate features and target
        X_test = test_data.drop('quality_class', axis=1)
        y_test = test_data['quality_class']
        
        # Load scaler
        try:
            scaler = load_model('scaler_model')
            X_test_scaled = scaler.transform(X_test)
        except FileNotFoundError:
            st.warning("⚠️ Scaler not found. Using unscaled features.")
            X_test_scaled = X_test.values
        
        # Load selected model
        try:
            model = load_model(model_options[selected_model_name])
            st.success(f"✅ Loaded model: **{selected_model_name}**")
        except FileNotFoundError:
            st.error(f"❌ Model file not found: model/{model_options[selected_model_name]}.py")
            st.stop()
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        st.divider()
        
        # Display evaluation metrics
        st.header("📈 Evaluation Metrics")
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}")
            st.metric("AUC Score", f"{auc:.4f}")
        
        with col2:
            st.metric("Precision", f"{precision:.4f}")
            st.metric("Recall", f"{recall:.4f}")
        
        with col3:
            st.metric("F1 Score", f"{f1:.4f}")
            st.metric("MCC Score", f"{mcc:.4f}")
        
        st.divider()
        
        # Confusion Matrix and Classification Report
        st.header("🎯 Model Performance Analysis")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Bad Wine (0)', 'Good Wine (1)'],
                       yticklabels=['Bad Wine (0)', 'Good Wine (1)'],
                       ax=ax)
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('True Label', fontsize=12)
            ax.set_title(f'Confusion Matrix - {selected_model_name}', fontsize=14)
            st.pyplot(fig)
            plt.close()
        
        with col_right:
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, 
                                          target_names=['Bad Wine (0)', 'Good Wine (1)'],
                                          output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)
        
        st.divider()
        
        # Model comparison (if results file exists)
        st.header("🏆 Model Comparison")
        try:
            results_df = pd.read_csv(MODEL_DIR / 'model_results.csv')
            st.dataframe(results_df.style.format({
                'Accuracy': '{:.4f}',
                'AUC': '{:.4f}',
                'Precision': '{:.4f}',
                'Recall': '{:.4f}',
                'F1': '{:.4f}',
                'MCC': '{:.4f}'
            }), use_container_width=True)
            
            # Highlight best model for each metric
            st.markdown("**🌟 Best Models by Metric:**")
            metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
            best_models = {}
            for metric in metrics:
                best_idx = results_df[metric].idxmax()
                best_models[metric] = results_df.loc[best_idx, 'Model']
            
            cols = st.columns(3)
            for idx, (metric, model) in enumerate(best_models.items()):
                with cols[idx % 3]:
                    st.info(f"**{metric}:** {model}")
            
        except FileNotFoundError:
            st.warning("⚠️ Model comparison results not found. Train models first using train_models.py")
        
        st.divider()
        
        # Predictions preview
        with st.expander("🔍 View Predictions (First 20 samples)"):
            predictions_df = pd.DataFrame({
                'True Label': y_test.values[:20],
                'Predicted Label': y_pred[:20],
                'Prediction Probability': y_pred_proba[:20],
                'Correct': (y_test.values[:20] == y_pred[:20])
            })
            st.dataframe(predictions_df, use_container_width=True)
            
            correct_count = (y_test == y_pred).sum()
            st.write(f"**Total Correct Predictions:** {correct_count} / {len(y_test)} ({100*correct_count/len(y_test):.2f}%)")
        
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.exception(e)

else:
    st.info("👆 Please upload a CSV file to begin evaluation")
    st.markdown("""
    **Expected CSV Format:**
    - Features: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, 
      free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol
    - Target: quality_class (0 or 1)
    
    **Note:** The test data file is available in the `model/` directory after training.
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center'>
    <p><b>ML Assignment 2 - M.Tech AIML</b></p>
    <p>Classification Model on Wine Quality Classification</p>
</div>
""", unsafe_allow_html=True)

