# Loan Approval Prediction

## Overview

This project explores **credit risk modeling** using Lending Club’s publicly available loan dataset.  
The goal is to **predict whether a borrower will repay or default on a loan** based on application details, credit history, and income data.  
This work demonstrates a full **end-to-end machine learning workflow**: data cleaning, feature engineering, class imbalance handling, model tuning, and interpretability.

---
## Key Objectives
- Predict loan repayment outcomes (good vs. bad loans) to support **lending decisions**.
- Address **extreme class imbalance** (~98% of loans are good) using advanced sampling and modeling strategies.
- Focus on **recall for risky loans** to minimize financial losses from defaults.

---
## Project Structure
loan_default_prediction/
├── loan_prediction.ipynb # Full analysis notebook
├── loans_full_schema.csv # Dataset (if license permits, or link to source)
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---
## Technical Approach

1. **Data Preprocessing**
   - Missing value imputation (median for numeric, 'Unknown' for categorical)
   - Outlier inspection and distribution checks
   - Feature encoding (One-Hot Encoding)

2. **Feature Engineering**
   - Credit history length (time between earliest credit line and loan issue)
   - Income-to-loan ratios
   - Payment ratios and utilization features
   - Removal of uninformative features (e.g., Lending Club grade letter)

3. **Imbalance Handling**
   - **SMOTEENN** (Synthetic Minority Oversampling + Edited Nearest Neighbors)  
   - **BalancedRandomForestClassifier** with `class_weight='balanced'`

4. **Model Selection**
   - Hyperparameter tuning with `GridSearchCV`
   - Evaluation metrics: Precision, Recall, F1, ROC-AUC, PR curves

---

## Results

| Metric (Test Set)             | Class 0 (Bad Loans) | Class 1 (Good Loans) | Overall |
|------------------------------|--------------------|---------------------|---------|
| Precision                    | 0.19               | 0.99                | -       |
| Recall                       | 0.31               | 0.98                | -       |
| F1-score                     | 0.23               | 0.98                | -       |
| Accuracy                     | -                  | -                   | **96.4%** |
| ROC-AUC                      | -                  | -                   | **0.64** |

Confusion Matrix:
[[ 11 25] <- Bad loans: 11 correctly flagged, 25 missed
[ 47 1917]] <- Good loans: Almost all correctly classified


**Key Insights:**
- Accuracy alone is misleading with **severely imbalanced data**.
- The pipeline focuses on **detecting risky loans** (recall for Class 0 = 31%), which is crucial for lenders.
- Removing low-importance features and engineering ratio-based metrics improved model interpretability.

---

## Next Steps
- Tune probability thresholds to balance recall and precision for bad loans.
- Experiment with gradient boosting models (XGBoost, LightGBM, CatBoost).
- Use **PR AUC** instead of ROC AUC for rare-event evaluation.
- Perform **time-based validation** to simulate real-world loan approvals.
- Consider financial cost-based optimization to prioritize minimizing default risk.

---

## Skills Demonstrated
- **Data Analysis & Feature Engineering** (credit risk features, ratio-based metrics)
- **Imbalanced Classification** (SMOTEENN, balanced ensembles, class weights)
- **Model Tuning & Evaluation** (cross-validation, ROC/PR curves, confusion matrices)
- **Explainability** (feature importance, business trade-offs)
- **Professional Reporting** (portfolio-ready notebook, structured documentation)

---

## Business Value
Even modest gains in recall for default prediction can significantly reduce losses for lenders.  
This project showcases **machine learning as a decision-support tool** for financial risk teams, with transparency and actionable metrics over raw accuracy.

---

## Author
**Steve Hernandez**  
📧 [steve.hernamont@gmail.com] | 🔗 [[LinkedIn](https://www.linkedin.com/in/sthermon/)] | 🌐 []  

---


## How to Run
1. Clone this repo
2. Install requirements:
   ```bash
   pip install -r requirements.txt


   # 🏦 Loan Default Prediction — Lending Club Dataset

---



---




---

