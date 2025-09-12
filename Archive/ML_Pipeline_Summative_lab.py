import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

# Load the financial loan risk data
df_loan = pd.read_csv('financial_loan_data.csv')
df_loan.info()
df_loan.describe()
df_loan.head()
df_loan.isnull().sum()
df_loan['LoanApproved'].value_counts()

approval_counts= df_loan['LoanApproved'].value_counts()
print(approval_counts)
# Visualize (use the counts object)
X = approval_counts.index
y = approval_counts.values

# Code for plot provided
sns.barplot(x=X, y=y)
plt.xlabel('Approvals')
plt.ylabel('Count')
plt.title("Distribution of Loan Approvals")
#plt.show()

# Convert AnnualIncome to float and handle errors
df_loan['AnnualIncome'] = df_loan['AnnualIncome'].str.replace('$', '').str.replace(',', '')
df_loan['AnnualIncome'] = df_loan['AnnualIncome'].astype(float)
df_loan['AnnualIncome'] = pd.to_numeric(df_loan['AnnualIncome'], errors='coerce')
# Check for NaN values after conversion
df_loan['AnnualIncome'].isnull().sum()

# Find categorical, ordinal, and numerical columns
categorical_cols = df_loan.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df_loan.select_dtypes(include=[np.number]).columns.tolist()
ordinal_cols = ['EmploymentStatus', 'EducationLevel', 'MaritalStatus','BankruptcyHistory','HomeOwnershipStatus','NumberOfDependents','LoanPurpose']
#remove ordinal columns from numerical columns
numerical_cols = [col for col in numerical_cols if col not in ordinal_cols]
#remove ordinal columns from categorical columns
categorical_cols = [col for col in categorical_cols if col not in ordinal_cols]
print("Categorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)
print("Ordinal columns:", ordinal_cols)
# Plot the ordinal columns to LoanApproved
for col in ordinal_cols:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_loan, x=col, hue='LoanApproved')
    plt.title(f'Count of Loan Approvals by {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.legend(title='Loan Approved', loc='upper right')
    plt.show() 

for i, feature in enumerate(ordinal_cols):
    #plt.subplot(2, 2, i+1)
    plt.figure(figsize=(8, 5))
    
    # Calculate the percentage of approvals for each category
    category_pass_rate = df_loan.groupby(feature)['LoanApproved'].mean() * 100
    
    # Plot fail rate by category
    sns.barplot(x=category_pass_rate.index, y=category_pass_rate.values)
    plt.title(f'Loan Approval Rate by {feature}')
    plt.ylabel('Loan Approval (%)')
    plt.ylim(0, 100)
    
    # Add count annotations on each bar
    counts = df_loan[feature].value_counts()
    for j, p in enumerate(plt.gca().patches):
        if j < len(counts):
            category = category_pass_rate.index[j]
            count = counts.get(category, 0)
            plt.gca().annotate(f'n={count}', 
                              (p.get_x() + p.get_width()/2., p.get_height()), 
                              ha='center', va='center', xytext=(0, 10), 
                              textcoords='offset points')

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Fill missing with median
    ('scaler', StandardScaler())  # Scale features to have mean=0, variance=1
])

# Create transformer for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill with most common value
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))  # One-hot encode
])

# Combine preprocessing steps using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, ordinal_cols)
    ]
)

x=df_loan.drop('LoanApproved',axis=1)
y=df_loan.loc[:,'LoanApproved']

# Split into training and test sets (75% train, 25% test) use random_state = 42 and set stratify = y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create Random Forest pipeline (set random_state=42)
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Create Logistic Regression pipeline (set random_state=42 and max_inter=1000)
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])
# Create Gradient Boosting pipeline (set random_state=42)
gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])
# Create AdaBoost pipeline (set random_state=42)
ab_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', AdaBoostClassifier(random_state=42))
])
# Create XGBoost pipeline (set random_state=42)
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
])
# Create Desicion Tree pipeline (set random_state=42)
dt_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# Define parameter grids for each model
rf_param_grid = {
    # Preprocessing parameters
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    
    # Model parameters
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__class_weight': ['balanced', None]
}

lr_param_grid = {
    # Preprocessing parameters
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    
    # Model parameters
    'classifier__C': [0.1, 1.0, 10.0],
    'classifier__class_weight': ['balanced', None],
    'classifier__solver': ['liblinear', 'saga']
}

gb_param_grid = {
    # Preprocessing parameters
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    
    # Model parameters
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 5, 7],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__subsample': [0.8, 1.0]
}

ab_param_grid = {
    # Preprocessing parameters
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    
    # Model parameters
    'classifier__n_estimators': [50, 100, 200],
    'classifier__learning_rate': [0.01, 0.1, 1.0],
    'classifier__base_estimator': [None, DecisionTreeClassifier()]
}
xgb_param_grid = {
    # Preprocessing parameters
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    
    # Model parameters
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 5, 7],
    'classifier__subsample': [0.8, 1.0],
    'classifier__colsample_bytree': [0.8, 1.0]
}

dt_param_grid = {
    # Preprocessing parameters
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    
    # Model parameters
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__class_weight': ['balanced', None]
}
# Define scoring metrics - we'll track multiple metrics
# but optimize for recall since missing at-risk students is costly
scoring = {
    'accuracy': 'accuracy',
    'recall': 'recall',
    'f1': 'f1'
}
# Create grid search for Random Forest
print("Starting Random Forest grid search...")
rf_grid_search = GridSearchCV(
    rf_pipeline,
    rf_param_grid,
    cv=5,  # 5-fold cross-validation
    scoring=scoring,
    refit='recall',  # Optimize for recall
    return_train_score=True
)
rf_grid_search.fit(X_train, y_train)

# Create grid search for linear regression
print("Starting Logistic Regression grid search...")
lr_grid_search = GridSearchCV(
    lr_pipeline,
    lr_param_grid,
    cv=5,  # 5-fold cross-validation
    scoring=scoring,
    refit='recall',  # Optimize for recall
    return_train_score=True
)
"""
lr_grid_search.fit(X_train, y_train)
# Create grid search for Gradient Boosting
print("Starting Gradient Boosting grid search...")
gb_grid_search = GridSearchCV(
    gb_pipeline,
    gb_param_grid,
    cv=5,  # 5-fold cross-validation
    scoring=scoring,
    refit='recall',  # Optimize for recall
    return_train_score=True
)
gb_grid_search.fit(X_train, y_train)
# Create grid search for AdaBoost
print("Starting AdaBoost grid search...")
ab_grid_search = GridSearchCV(
    ab_pipeline,
    ab_param_grid,
    cv=5,  # 5-fold cross-validation
    scoring=scoring,
    refit='recall',  # Optimize for recall
    return_train_score=True
)
ab_grid_search.fit(X_train, y_train)
# Create grid search for XGBoost
print("Starting XGBoost grid search...")
xgb_grid_search = GridSearchCV(
    xgb_pipeline,
    xgb_param_grid,
    cv=5,  # 5-fold cross-validation
    scoring=scoring,
    refit='recall',  # Optimize for recall
    return_train_score=True
)
xgb_grid_search.fit(X_train, y_train)
# Create grid search for Decision Tree
print("Starting Decision Tree grid search...")
dt_grid_search = GridSearchCV(
    dt_pipeline,
    dt_param_grid,
    cv=5,  # 5-fold cross-validation
    scoring=scoring,
    refit='recall',  # Optimize for recall
    return_train_score=True
)
dt_grid_search.fit(X_train, y_train)"""

# Compare models
rf_best_cvscore = rf_grid_search.best_score_
lr_best_cvscore = lr_grid_search.best_score_

print(f"Best Random Forest f1_score: {rf_best_cvscore:.4f}")
print(f"Best Logistic Regression f1_score: {lr_best_cvscore:.4f}")

# Select the best model based on recall
if rf_best_cvscore >= lr_best_cvscore:
    best_model = rf_grid_search
    model_name = "Random Forest"
else:
    best_model = lr_grid_search
    model_name = "Logistic Regression"

print(f"\nBest model: {model_name}")

final_model = lr_grid_search.best_estimator_

# Make predictions on test set using final_model
y_pred = final_model.predict(X_test)

recall_score = final_model.score(X_test, y_test)
print(f"Recall score on test set: {recall_score:.4f}")

# Create classification report
cr = classification_report(y_test, y_pred)

# Create confusion matrix object
cm = confusion_matrix(y_test, y_pred)

print(f"\n===== Final Model Evaluation =====")
print(f"Recall score: {recall_score:.4f}")
print("\nClassification Report:")
print(cr)
print(f"Confusion Matrix:")
ConfusionMatrixDisplay(cm, display_labels=['Low Approval', 'High Approval']).plot();

feature_names = []

# Numerical feature names directly to list
feature_names.extend(numerical_cols)

# For categorical features, get the encoded feature names
# This is a bit complex because we need to extract the names from the preprocessing pipeline
# Access preprocessor
preprocessor = final_model.named_steps['preprocessor']

# Access the categorical Onehotencoder
cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']

# Extract feature names
encoded_cat_features = cat_encoder.get_feature_names_out(ordinal_cols)

# Extend list with categorical feature names
feature_names.extend(encoded_cat_features)

# Extract model from final_model pipelne (use named_steps)
lr_model = final_model.named_steps['classifier']

# Get feature coefficients (importance)
importances = lr_model.coef_[0]

# DataFrame for easier viewing
feature_imp = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sorting high to low
feature_imp = feature_imp.sort_values('Importance', ascending=False)
