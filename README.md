# 📊 Bankruptcy Prediction using Machine Learning

This project builds an end-to-end machine learning system to predict corporate bankruptcy using financial ratio data from Taiwanese firms. The primary challenge addressed is severe class imbalance (~3% bankrupt firms), which requires specialized modeling, evaluation, and thresholding strategies.

---

## 🚀 Project Overview

Corporate bankruptcy prediction is a high-impact financial risk problem where missing a bankrupt firm (false negative) is far more costly than incorrectly flagging a healthy firm (false positive). This project:

- Uses multiple machine learning models to predict bankruptcy
- Handles extreme class imbalance using resampling and threshold tuning
- Prioritizes recall while maintaining reasonable precision
- Produces a reproducible, well-evaluated, and deployable pipeline

---

## 📁 Dataset

- **Source:** UCI Machine Learning Repository — Taiwanese Bankruptcy Prediction Dataset  
- **Instances:** 6,819 companies  
- **Features:** 95 financial ratios + 1 target  
- **Target:** `Bankrupt` (1 = bankrupt, 0 = non-bankrupt)  
- **Imbalance:** ~3.23% bankrupt firms  

---

## 🧠 Modeling Approach

### Models Evaluated
- Logistic Regression (baseline)
- Random Forest
- XGBoost (final selected model)

### Key Techniques
- Multicollinearity removal using VIF
- Class imbalance handling using SMOTE
- Stratified cross-validation
- Multi-metric evaluation (Accuracy, Precision, Recall, F1)
- Probability threshold optimization using Precision–Recall analysis

### Final Model
- **Model:** XGBoost
- **Threshold:** Tuned via PR curve (~0.21)
- **Goal:** Maximize recall with acceptable precision

---

## 📈 Results (Test Set)

| Metric | Value |
|--------|--------|
| Accuracy | ~0.96 |
| Precision | ~0.38 |
| Recall | ~0.77 |
| F1 Score | ~0.50 |
