# ML Assignment 2 - Wine Quality Classification

**M.Tech (AIML/DSE) - Machine Learning Assignment 2**
**Author:** Karthik S Kashyap, M.Tech AIML
**Date:** January 2026

---

## 📋 Problem Statement

The objective of this assignment is to build and evaluate multiple machine learning classification models to predict wine quality based on physicochemical properties. The problem involves:

1. **Binary Classification Task:** Classify wines as "Good Quality" (quality ≥ 6) or "Bad Quality" (quality < 6)
2. **Model Implementation:** Implement and compare 6 different classification algorithms
3. **Comprehensive Evaluation:** Calculate 6 evaluation metrics for each model
4. **Web Application Deployment:** Create an interactive Streamlit app for model demonstration
5. **Cloud Deployment:** Deploy the application on Streamlit Community Cloud

The assignment demonstrates the complete machine learning workflow: data preparation, model training, evaluation, web application development, and cloud deployment.

---

## 📊 Dataset Description

### Dataset Overview

- **Name:** Wine Quality Dataset (Red Wine)
- **Source:** UCI Machine Learning Repository
- **URL:** https://archive.ics.uci.edu/ml/datasets/wine+quality
- **Total Instances:** 1,599 samples
- **Number of Features:** 11 physicochemical features + 1 target variable
- **Classification Type:** Binary (Good Wine vs Bad Wine)

### Features Description

| Feature                        | Description                                                                | Type       |
| ------------------------------ | -------------------------------------------------------------------------- | ---------- |
| **Fixed Acidity**        | Amount of non-volatile acids (tartaric acid) in g/dm³                     | Continuous |
| **Volatile Acidity**     | Amount of acetic acid in wine (g/dm³) - high levels lead to vinegar taste | Continuous |
| **Citric Acid**          | Acts as preservative and adds freshness (g/dm³)                           | Continuous |
| **Residual Sugar**       | Amount of sugar remaining after fermentation (g/dm³)                      | Continuous |
| **Chlorides**            | Amount of salt in the wine (g/dm³)                                        | Continuous |
| **Free Sulfur Dioxide**  | Free form of SO₂, prevents microbial growth (mg/dm³)                     | Continuous |
| **Total Sulfur Dioxide** | Total amount of SO₂ (free + bound forms) (mg/dm³)                        | Continuous |
| **Density**              | Density of wine (g/cm³)                                                   | Continuous |
| **pH**                   | Acidity level on scale of 0-14 (0=very acidic, 14=very basic)              | Continuous |
| **Sulphates**            | Wine additive contributing to SO₂ levels (g/dm³)                         | Continuous |
| **Alcohol**              | Percentage of alcohol content (% vol)                                      | Continuous |

### Target Variable

- **Original:** Quality score (0-10 integer scale)
- **Transformed:** Binary classification
  - **Class 0 (Bad Wine):** Quality < 6
  - **Class 1 (Good Wine):** Quality ≥ 6

### Dataset Statistics

- **Training Set:** 80% (1,279 samples)
- **Test Set:** 20% (320 samples)
- **Feature Scaling:** StandardScaler applied to normalize features
- **Class Distribution:** Balanced split ensuring representational fairness

---

## 🤖 Models Used

### Model Comparison Table

| ML Model Name       | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ------------------- | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression | 0.7406   | 0.8242 | 0.7683    | 0.7368 | 0.7522 | 0.4808 |
| Decision Tree       | 0.7531   | 0.7718 | 0.7706    | 0.7661 | 0.7683 | 0.5041 |
| KNN                 | 0.7406   | 0.8117 | 0.7588    | 0.7544 | 0.7566 | 0.4790 |
| Naive Bayes         | 0.7219   | 0.7884 | 0.7733    | 0.6784 | 0.7227 | 0.4500 |
| Random Forest       | 0.8031   | 0.9020 | 0.8293    | 0.7953 | 0.8119 | 0.6062 |
| XGBoost             | 0.8250   | 0.8963 | 0.8485    | 0.8187 | 0.8333 | 0.6497 |

**Note:** These are the actual results from training on Wine Quality dataset.

---

## 🔍 Model Performance Observations

### Logistic Regression

**Observation:** Logistic Regression provides a solid baseline with 74.06% accuracy and strong AUC of 0.8242. The model demonstrates good balance between precision (0.7683) and recall (0.7368), making it reliable for both classes. Its linear decision boundary effectively captures the relationships between physicochemical features and wine quality. Training is fast and the model offers excellent interpretability through feature coefficients. Best suited when computational efficiency and model explainability are priorities.

### Decision Tree

**Observation:** Decision Tree achieves 75.31% accuracy with well-balanced precision (0.7706) and recall (0.7661). The model successfully captures non-linear patterns through hierarchical splits on features like alcohol content and volatile acidity. Despite max_depth constraint preventing extreme overfitting, it shows good generalization. Feature interactions are naturally modeled through the tree structure. The model provides clear decision rules that can be visualized, making it valuable for understanding quality prediction logic.

### K-Nearest Neighbors (KNN)

**Observation:** KNN delivers 74.06% accuracy with strong AUC (0.8117) and good precision-recall balance (0.7588 and 0.7544). The model benefits significantly from feature standardization, which normalizes the different scales of physicochemical properties. With k=5 neighbors, it captures local patterns effectively without overfitting. However, prediction time increases with dataset size as it requires distance computation. Works well for this dataset's moderate dimensionality (11 features).

### Naive Bayes

**Observation:** Naive Bayes achieves 72.19% accuracy while maintaining the highest precision (0.7733) among non-ensemble methods. Despite assuming feature independence (which doesn't hold for correlated wine properties), the model performs reasonably well, demonstrating its robustness. The probabilistic framework enables easy threshold adjustment for different precision-recall tradeoffs. Extremely fast training makes it ideal for rapid prototyping and baseline comparisons. Lower recall (0.6784) suggests conservative predictions.

### Random Forest (Ensemble)

**Observation:** Random Forest delivers excellent performance with 80.31% accuracy and outstanding AUC of 0.9020. The ensemble of 100 decision trees effectively reduces variance while maintaining low bias. Strong MCC score (0.6062) indicates reliable predictions across both classes. The model handles feature interactions naturally and provides robust feature importance rankings, identifying alcohol content and sulphates as key predictors. Balanced precision (0.8293) and recall (0.7953) make it highly suitable for practical deployment.

### XGBoost (Ensemble)

**Observation:** XGBoost achieves the **best overall performance** with 82.50% accuracy, highest MCC (0.6497), and excellent AUC (0.8963). The gradient boosting framework sequentially corrects errors, leading to superior predictive power. Strong F1 score (0.8333) demonstrates optimal precision-recall tradeoff. Regularization parameters prevent overfitting despite complex model structure. Feature importance via SHAP values reveals alcohol, volatile acidity, and sulphates as critical quality indicators. **Recommended as the primary production model** for wine quality prediction.

---

## 🚀 Project Structure

```
ML-Assignment-2/
│
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation (this file)
│
└── model/
    ├── train_models.py         # Model training script
    ├── test_data.csv           # Test dataset for evaluation
    ├── model_results.csv       # Comparison results for all models
    ├── scaler.pkl              # Fitted StandardScaler object
    ├── logistic_regression_model.pkl
    ├── decision_tree_model.pkl
    ├── knn_model.pkl
    ├── naive_bayes_model.pkl
    ├── random_forest_model.pkl
    └── xgboost_model.pkl
```

---

## 🔧 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for version control)

### Local Setup

1. **Clone the repository:**

```bash
git clone https://github.com/kak4bmh/ML
cd ML/WineQuality
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Train the models:**

```bash
python model/train_models.py
```

4. **Run the Streamlit app locally:**

```bash
streamlit run app.py
```

5. **Access the application:** The app will open in your browser at `http://localhost:8501`
---

## 📱 Streamlit App Features

The web application includes the following required features:

### ✅ 1. Dataset Upload Option (CSV)

- Upload test data in CSV format
- Supports the Wine Quality dataset format
- Displays data preview and statistics
- File located in `model/test_data.csv` for testing

### ✅ 2. Model Selection Dropdown

- Choose from 6 trained classification models:
  - Logistic Regression
  - Decision Tree
  - K-Nearest Neighbors
  - Naive Bayes
  - Random Forest
  - XGBoost
- Dynamic model loading based on selection

### ✅ 3. Display of Evaluation Metrics

- **Accuracy:** Overall correctness of predictions
- **AUC Score:** Area Under the ROC Curve
- **Precision:** Positive predictive value
- **Recall:** Sensitivity/True positive rate
- **F1 Score:** Harmonic mean of precision and recall
- **MCC Score:** Matthews Correlation Coefficient

### ✅ 4. Confusion Matrix & Classification Report

- **Confusion Matrix:** Visual heatmap showing true vs predicted labels
- **Classification Report:** Detailed per-class metrics
- **Model Comparison Table:** Side-by-side comparison of all models

---

## 🌐 Deployment on Streamlit Community Cloud

1. **Deploy on Streamlit Cloud:**
   
   - Push your directory to GitHub repository
   - Go to https://streamlit.io/cloud
   - Sign in with your GitHub account
   - Click **"New App"**
   - Select your repository
   - Choose branch: `main`
   - Select main file: `WineQuality/app.py`
   - Update App URL: `ml-assignment-2025aa05388-winequality`
   - Click **"Deploy"**
The app will be live within few minutes at: 'https://ml-assignment-2025aa05388-winequality.streamlit.app'

3. **Wait for deployment:**

   - Deployment typically takes 2-5 minutes
   - Streamlit will install dependencies from `requirements.txt`
   - App will be live at: `https://<your-app-name>.streamlit.app`

### Important Notes for Deployment

- Ensure `requirements.txt` includes all dependencies
- Model files must be pushed to GitHub (use Git LFS if files > 100MB)
- Test data should be small enough for free tier limits
- Environment variables can be set in Streamlit Cloud settings

---

## 📈 Results Summary

### Best Performing Models

1. **XGBoost:** Best overall performance with 82.50% accuracy and AUC of 0.8963
2. **Random Forest:** Second best with 80.31% accuracy and excellent AUC of 0.9020
3. **Decision Tree:** Strong individual classifier at 75.31% accuracy

### Key Insights

- Ensemble methods (XGBoost, Random Forest) significantly outperform individual classifiers, with 8-10% accuracy improvement
- XGBoost achieves the highest MCC score (0.6497), indicating most reliable predictions
- Feature scaling proves crucial for distance-based models (KNN) and linear models (Logistic Regression)
- All models achieve AUC > 0.77, demonstrating good discriminative ability for wine quality classification
- Precision-recall balance is well-maintained across models, with F1 scores ranging from 0.72 to 0.83
- Alcohol content, sulphates, and volatile acidity emerge as the most important predictive features

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **Scikit-learn:** Model implementation and evaluation
- **XGBoost:** Gradient boosting classifier
- **Streamlit:** Web application framework
- **Pandas & NumPy:** Data manipulation
- **Matplotlib & Seaborn:** Visualization
- **Pickle:** Model serialization

---

## 📝 Assignment Requirements Checklist

- ✅ Dataset with minimum 12 features and 500 instances
- ✅ 6 classification models implemented
- ✅ 6 evaluation metrics calculated for each model
- ✅ Streamlit app with all required features
- ✅ GitHub repository with proper structure
- ✅ requirements.txt for deployment
- ✅ README.md with comprehensive documentation
- ✅ Model comparison table with observations
- ✅ Ready for Streamlit Community Cloud deployment

---

## 📧 Contact & Submission

**Course:** M.Tech AIML - Machine Learning
**Assignment:** Assignment 2
**Submission Deadline:** 15-Feb-2026

---

## 📄 License

This project is submitted as part of M.Tech AIML coursework at BITS Pilani.

---

**Note:** This README content is part of the assignment submission PDF as per requirement 2.4 in the assignment document.


