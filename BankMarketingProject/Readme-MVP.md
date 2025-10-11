# Bank Marketing Campaign Optimization

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A machine learning project to predict term deposit subscriptions and optimize direct marketing campaign ROI through targeted customer outreach.

## Project Overview

This project develops a predictive model to identify bank customers most likely to subscribe to term deposits, enabling more efficient marketing campaigns by focusing resources on high-probability prospects.

### Key Results

| Metric | Value |
|--------|-------|
| F1-Score | 0.37 |
| Recall | 51% |
| Precision | 27% |
| ROI Improvement | 14.3% vs random |

## Business Context

Direct marketing campaigns are costly and inefficient when targeting customers randomly. This model helps financial institutions:

- Reduce marketing waste by avoiding unlikely prospects
- Improve conversion rates through better targeting
- Optimize resource allocation based on predicted probabilities
- Increase campaign ROI through data-driven decision making

## Dataset

**Source:** Bank marketing campaign data (41,188 records, 20 features)

**Target Variable:** Binary classification - will customer subscribe to term deposit?

**Class Distribution:**
- No: 75%
- Yes: 25%

**Feature Categories:**
- Customer demographics (age, job, education, marital status)
- Financial indicators (housing loan, personal loan, credit default)
- Campaign data (contact type, month, day of week, number of contacts)
- Economic indicators (employment rate, consumer confidence, interest rates)
- Previous campaign outcomes

**Critical Note:** The `duration` feature (call length) was removed due to data leakage - it's only known after the call completes, making it unusable for prediction before contact.

## Technical Approach

### 1. Data Preprocessing

- Handled 'unknown' categorical values via mode imputation
- Applied Winsorization (5th-95th percentile) for outlier treatment
- Removed data leakage by excluding duration feature
- Created train/validation/test split (64%/16%/20%)

### 2. Feature Engineering

Created domain-informed composite features:

| Feature Type | Examples |
|-------------|----------|
| Customer Behavior | Previous contact indicators, contact frequency scores |
| Economic Context | Stability index from macroeconomic indicators |
| Demographics | Life stage categories, financial stability indicators |
| Interactions | Age × campaign, consumer confidence × campaign |

### 3. Model Development

**Models Evaluated:**
- Logistic Regression (class_weight='balanced') - **Selected**
- Random Forest (class_weight='balanced')
- XGBoost (scale_pos_weight=3)
- SVM (class_weight='balanced')

**Selection Process:**
1. Trained all models on training set
2. Evaluated on validation set using F1-score
3. Selected best performer (Logistic Regression)
4. Hyperparameter tuning via GridSearchCV with 5-fold stratified CV
5. Final evaluation on held-out test set (one time only)

### 4. Best Model: Tuned Logistic Regression

**Hyperparameters:**
penalty: 'l1'          # Lasso regularization
C: 0.01                # Regularization strength
class_weight: 'balanced'
solver: 'liblinear'


Test Set Performance:
Accuracy:    0.51
Precision:   0.27 (for 'yes' class)
Recall:      0.51 (for 'yes' class)
F1-Score:    0.37 (for 'yes' class)
ROC-AUC:     0.55
Confusion Matrix (Test Set):
                | Predicted No | Predicted Yes |
----------------|--------------|---------------|
Actual No       |    2,052     |     1,697     |
Actual Yes      |      618     |       633     |


Key Insights
Feature Importance
Top predictive features (by coefficient magnitude):

Education (professional course) - Strongest positive predictor
Consumer Confidence Index - Higher confidence → higher subscription likelihood
Age - Modest negative correlation (younger slightly more likely)
Employment Variation Rate - Weak positive correlation

Business Recommendations

Prioritize educated professionals: Target customers with professional course backgrounds
Time campaigns strategically: Launch during periods of higher consumer confidence
Consider demographic factors: Age plays a secondary role in targeting decisions
Monitor economic indicators: Track macroeconomic trends for optimal timing

ROI Analysis
Assumptions (Example Parameters):

Cost per contact: $1
Revenue per subscription: $10

Scenario Comparison:
MetricModel TargetingRandom TargetingImprovementContacts2,3302,330-Subscriptions633583+8.6%Total Cost$2,330$2,330-Total Revenue$6,330$5,830+8.6%Net Profit$4,000$3,500+14.3%ROI172%150%+22 pts
Scaling to Realistic Parameters:
At $10/contact and $200/subscription:

Model profit: ~$360K
Random profit: ~$273K
Additional value: ~$87K per campaign

Project Structure
bank-marketing-optimization/
├── BankMarketing_MVP.ipynb              # Main analysis notebook
├── bank_marketing_2024.csv              # Dataset (not included)
├── tuned_logistic_regression_model.joblib  # Saved model
├── README.md                            # This file
└── requirements.txt                     # Python dependencies
Installation & Usage
Requirements
bashpip install -r requirements.txt
Key Dependencies:
pandas >= 1.5.0
numpy >= 1.23.0
scikit-learn >= 1.2.0
xgboost >= 1.7.0
matplotlib >= 3.6.0
seaborn >= 0.12.0
scipy >= 1.10.0
Running the Analysis
python# Open Jupyter notebook
jupyter notebook BankMarketing_MVP.ipynb

# Or load the saved model for predictions
import joblib
model = joblib.load('tuned_logistic_regression_model.joblib')
predictions = model.predict(new_customer_data)
Making Predictions on New Data
pythonimport pandas as pd
import joblib

# Load model
model = joblib.load('tuned_logistic_regression_model.joblib')

# Prepare new data (must include all engineered features)
new_customers = pd.read_csv('new_customers.csv')
# ... feature engineering steps ...

# Predict
predictions = model.predict(new_customers)
probabilities = model.predict_proba(new_customers)[:, 1]

# Target customers with probability > threshold
threshold = 0.5  # Adjust based on business costs
target_list = new_customers[probabilities > threshold]
Limitations & Future Work
Current Limitations

Modest precision (27%): 73% of predicted subscribers won't convert
Baseline comparison: Compared to random, not current targeting strategy
No threshold optimization: Uses default 0.5 probability cutoff
Limited hyperparameter search: Only tuned Logistic Regression extensively
No deployment pipeline: Model monitoring and retraining strategy not implemented

Recommended Improvements

 Threshold optimization: Use precision-recall curves to find optimal cutoff
 Cost-sensitive learning: Incorporate actual costs into loss function
 Ensemble methods: Combine multiple models for better performance
 Feature selection: Address multicollinearity by removing redundant features
 Temporal validation: Test model on future time periods
 A/B testing framework: Deploy with controlled experiment
 Model monitoring: Implement drift detection and automated retraining

Deployment Considerations
Before Production

 Validate with actual business cost/revenue parameters
 Benchmark against current targeting strategy (not random)
 Conduct threshold analysis for optimal precision/recall tradeoff
 Build feature engineering pipeline for new data
 Implement model monitoring dashboard
