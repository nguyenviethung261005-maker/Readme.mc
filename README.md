# Loan Default Prediction using Machine Learning

## Project Overview

This project aims to predict whether a borrower will default on a loan using machine learning classification models. Early identification of high-risk borrowers helps financial institutions improve credit risk management and reduce potential losses.

The project follows a complete machine learning workflow, including data preprocessing, exploratory data analysis (EDA), feature engineering, model development, evaluation, feature importance analysis, and error analysis.

---

# Dataset

The dataset contains borrower demographic information, financial status, loan characteristics, and historical credit information.

## Features

| Variable | Description |
|----------|-------------|
| Age | Borrower's age |
| Annual Income | Annual income |
| Home Ownership | Home ownership status |
| Employment Length | Years of employment |
| Loan Intent | Purpose of the loan |
| Loan Grade | Credit grade assigned to the loan |
| Loan Amount | Amount borrowed |
| Interest Rate | Loan interest rate |
| Percent Income | Loan amount as a percentage of annual income |
| Historical Default | Previous default history |
| Credit History Length | Length of credit history |
| Loan Status | Target variable (0 = Non-default, 1 = Default) |

---

# Exploratory Data Analysis (EDA)

EDA was conducted to understand the data distribution, detect missing values, identify outliers, and investigate relationships among variables.

## Missing Values

- Missing values were identified in:
  - Employment Length
  - Interest Rate
- Missing numerical values were imputed using the median.

---

## Class Distribution

Loan Status:

- Non-default: 78.2%
- Default: 21.8%

The dataset is moderately imbalanced but does not require aggressive resampling techniques.

---

## Distribution Analysis

Histograms and boxplots were created for continuous variables.

Observations:

- Annual Income is heavily right-skewed.
- Loan Amount is moderately right-skewed.
- Percent Income is right-skewed.
- Interest Rate is approximately symmetric.
- Credit History Length is concentrated at lower values.
- Age is concentrated among younger borrowers.

---

## Outlier Detection

Outliers were identified using boxplots.

Since the extreme observations may represent genuine high-risk borrowers, they were retained instead of being removed.

---

## Correlation Analysis

A correlation matrix was generated to investigate relationships among numerical variables.

Findings:

- No single variable showed a very strong linear correlation with Loan Status.
- Percent Income is naturally related to Loan Amount and Annual Income.
- Multicollinearity was assessed using Variance Inflation Factor (VIF).

---

## Categorical Variables

Categorical variables were analyzed using count plots.

Variables include:

- Home Ownership
- Loan Intent
- Loan Grade
- Historical Default

---

# Data Preprocessing

The following preprocessing steps were performed:

- Missing value imputation
- Label Encoding for ordinal variables
- One-Hot Encoding for nominal variables
- Feature scaling using StandardScaler
- Train-test split with stratification

---

# Machine Learning Models

Several classification algorithms were investigated.

Models include:

- Logistic Regression
- K Nearest Neighboor
- Decision Tree
- Random Forest
- Bagging Classifier

The Random Forest model achieved the best overall performance.

---

# Hyperparameter Selection

Random Forest hyperparameters:

- n_estimators = 300
- criterion = gini
- random_state = 42
- oob_score = True
- n_jobs = -1

The number of trees was selected based on the stabilization of the Out-of-Bag (OOB) error.

---

# Model Evaluation

The following evaluation metrics were used:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

Since the dataset is moderately imbalanced, F1-score and ROC-AUC were considered more informative than accuracy alone.

---

# Feature Importance

Permutation Feature Importance was applied to evaluate the contribution of each predictor.

The most influential variables include:

- Interest Rate
- Annual Income
- Percent Income
- Loan Amount
- Credit History Length

Variables with negligible importance were considered for removal and model retraining.

---

# Error Analysis

Error analysis was conducted by comparing:

- True Negative (TN)
- False Negative (FN)
- True Positive (TP)
- False Positive (FP)

Key findings:

### False Negative

Compared with True Negative borrowers:

- Lower annual income
- Higher interest rate
- Slightly higher loan-to-income ratio

These borrowers share many characteristics with correctly classified non-default borrowers, making them difficult for the model to distinguish.

### False Positive

Compared with True Positive borrowers:

- Lower annual income
- Higher interest rate
- Higher proportion of medical-purpose loans

These borrowers exhibit characteristics commonly associated with higher credit risk, causing the model to incorrectly classify them as likely to default.

Overall, the error analysis indicates that borrower characteristics overlap between the two classes, which explains the remaining classification errors.

---

# Project Workflow

1. Load dataset
2. Data cleaning
3. Exploratory Data Analysis
4. Missing value imputation
5. Feature engineering
6. Encoding categorical variables
7. Feature scaling
8. Train-test split
9. Model training
10. Hyperparameter tuning
11. Model evaluation
12. Feature importance analysis
13. Error analysis

---

# Libraries

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

# Results

The Random Forest classifier achieved the best overall performance among all evaluated models.

The model demonstrates:

- High overall accuracy
- Strong ROC-AUC
- Balanced Precision and Recall
- Good F1-score

Feature importance and error analysis further improved the interpretability of the model and provided insights into misclassified borrowers.

---

# Future Work

Potential improvements include:

- Cost-sensitive learning
- Gradient Boosting methods (XGBoost, LightGBM, CatBoost)
- SHAP value interpretation
- Probability calibration
- Additional borrower behavioral features
