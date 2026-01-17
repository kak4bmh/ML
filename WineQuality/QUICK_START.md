# Quick Start Guide - Wine Quality Classification

## 🚀 Get Started in 3 Steps

### 1️⃣ Train the Models (One-time setup)

```powershell
# Navigate to project directory
cd ML-Assignment-2

# Install dependencies
pip install -r requirements.txt

# Train all models (takes 1-2 minutes)
python model\train_models.py
```

**Output:** Creates 6 model files, test data, and results CSV in `model/` directory

---

### 2️⃣ Run the Streamlit App Locally

```powershell
# From ML-Assignment-2 directory
streamlit run app.py
```

**App opens at:** http://localhost:8501

**Test the app:**
1. Upload `model/test_data.csv`
2. Select a model from dropdown
3. View metrics and confusion matrix

---

### 3️⃣ Deploy to Streamlit Cloud

```powershell
# Initialize Git
git init
git add .
git commit -m "Initial commit: Wine Quality Classification"

# Create repository on GitHub (do this via web interface)
# Then push code
git remote add origin https://github.com/YOUR_USERNAME/ml-assignment-2-wine-quality.git
git push -u origin main
```

**Then:**
1. Go to https://streamlit.io/cloud
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set main file: `app.py`
6. Click "Deploy"

**Your app will be live in 2-5 minutes! 🎉**

---

## 📊 Model Performance Summary

| Model | Accuracy | AUC | F1 Score |
|-------|----------|-----|----------|
| **XGBoost** 🥇 | 82.50% | 0.8963 | 0.8333 |
| **Random Forest** 🥈 | 80.31% | 0.9020 | 0.8119 |
| **Decision Tree** 🥉 | 75.31% | 0.7718 | 0.7683 |
| Logistic Regression | 74.06% | 0.8242 | 0.7522 |
| KNN | 74.06% | 0.8117 | 0.7566 |
| Naive Bayes | 72.19% | 0.7884 | 0.7227 |

---

## 📂 Project Structure

```
ML-Assignment-2/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Full documentation
├── DEPLOYMENT_GUIDE.md             # Detailed deployment steps
├── QUICK_START.md                  # This file
├── .gitignore                      # Git ignore rules
└── model/
    ├── train_models.py             # Model training script
    ├── test_data.csv               # Test dataset (320 samples)
    ├── model_results.csv           # Metrics comparison
    ├── scaler.pkl                  # Feature scaler
    ├── logistic_regression_model.pkl
    ├── decision_tree_model.pkl
    ├── knn_model.pkl
    ├── naive_bayes_model.pkl
    ├── random_forest_model.pkl
    └── xgboost_model.pkl
```

---

## 🎯 Assignment Submission

**Deadline:** 15-Feb-2026, 23:59 PM

**Submit a PDF with:**
1. ✅ GitHub repository link
2. ✅ Live Streamlit app link
3. ✅ BITS Virtual Lab screenshot
4. ✅ Complete README content

**Important:** Only ONE submission allowed!

---

## 💡 Tips for Success

✅ **Test locally first** - Run the app on your machine before deploying  
✅ **Check file sizes** - Model files should be < 100MB each  
✅ **Verify requirements.txt** - Ensure all packages are listed  
✅ **Test deployed app** - Upload test data and verify all features work  
✅ **Take clear screenshots** - Ensure BITS Lab watermark is visible  

---

## 🆘 Quick Troubleshooting

**Problem:** Models not found  
**Solution:** Run `python model\train_models.py` first

**Problem:** Streamlit command not found  
**Solution:** Use `python -m streamlit run app.py`

**Problem:** Module not found  
**Solution:** `pip install -r requirements.txt`

**Problem:** GitHub push fails  
**Solution:** Create Personal Access Token and use as password

---

## 📚 Useful Commands

```powershell
# Check Python version
python --version

# List installed packages
pip list

# View Streamlit version
streamlit --version

# Run specific model training
python -c "from model.train_models import *; print('Models trained!')"

# Check Git status
git status

# View Git commit history
git log --oneline
```

---

## 🎓 Learning Outcomes

By completing this assignment, you've learned:

✅ **Machine Learning:** Implementing 6 classification algorithms  
✅ **Model Evaluation:** Calculating 6 different metrics  
✅ **Web Development:** Building interactive apps with Streamlit  
✅ **Deployment:** Deploying ML apps to the cloud  
✅ **Version Control:** Using Git and GitHub  
✅ **Documentation:** Writing comprehensive READMEs  

---

**Ready to submit? Double-check the checklist in DEPLOYMENT_GUIDE.md! 📋**

---

*M.Tech AIML - Machine Learning Assignment 2 - January 2026*
