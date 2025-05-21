# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
arrest_data=pd.read_csv("arrests_data.csv")
# chick_data basic information about the dataset
arrest_data.info()
# Display the first few rows of the dataset
arrest_data.head()
# Rename the unknown column to State
arrest_data.rename(columns={'Unnamed: 0': 'State'}, inplace=True)
# Check for missing values
print(arrest_data.isnull().sum())
# Check for duplicates
print(f"Number of duplicate rows: {arrest_data.duplicated().sum()}")
# Drop labels for unsupervised learning
arrest_data.drop(columns=['State'], inplace=True)
# Check the shape of the dataset
print(f"Shape of the dataset: {arrest_data.shape}")