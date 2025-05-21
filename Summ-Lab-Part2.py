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
chick_data=pd.read_csv("chickwts_data.csv")
# Display basic information about the dataset
chick_data.info()
# Display the first few rows of the dataset
chick_data.head()
# Check for missing values
print(chick_data.isnull().sum())
# Check for duplicates
print(f"Number of duplicate rows: {chick_data.duplicated().sum()}")
# Check the shape of the dataset
print(f"Shape of the dataset: {chick_data.shape}")
# Check the data types of the columns
print(chick_data.dtypes)

# Check the summary statistics of the dataset
print(chick_data.describe())

# Numerical features
numerical_features = chick_data.select_dtypes(include=[np.number]).columns.tolist()
numerical_features.remove('weight')
feature_matrix = chick_data[numerical_features].values

# Standardize the numerical features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(feature_matrix)
# Create a DataFrame with normalized features for better visualization
normalized_df = pd.DataFrame(
    normalized_features, 
    columns=numerical_features, 
    index=chick_data['feed_name']
)
print("\nNormalized Features:")
print(normalized_df)

# STandardize the weight feature
weight = chick_data['weight'].values.reshape(-1, 1)
weight_scaled = scaler.fit_transform(weight)
# Add the scaled weight feature to the feature matrix
scaled_features = np.hstack((normalized_features, weight_scaled))

scaled_df = pd.DataFrame(
    scaled_features, 
    columns=numerical_features+['weight'], 
    index=chick_data['feed_name']
)
print("\nScaled Features:")
print(scaled_df)
# PCA
# Apply PCA to reduce data to **two** principal components
pca = PCA(n_components=2)  # Changed to 2 components
pca.fit(scaled_df)
# Transform the data
feature_matrix_pca = pca.transform(scaled_df)
# Create a DataFrame with the principal components
pca_df = pd.DataFrame({
    'feed_name': chick_data['feed_name'],
    'pca_value_1': feature_matrix_pca[:, 0],  # Added column names
    'pca_value_2': feature_matrix_pca[:, 1]
}).groupby('feed_name').mean()
# Reset index to make 'feed_name' a column
pca_df.reset_index(inplace=True)

# Look at the transformed data
print("First 5 rows of PCA-transformed data:")
print(pca_df.head())

# Cosine similarity
#calculate cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim_matrix = cosine_similarity(pca_df[['pca_value_1', 'pca_value_2']])  # Using both components
cosine_dist_matrix = 1 - cosine_sim_matrix

# Create a df for visualization
cosine_sim_df = pd.DataFrame(cosine_dist_matrix, index=pca_df['feed_name'], columns=pca_df['feed_name'])

# Display the cosine similarity DataFrame
print(cosine_sim_df)

print(cosine_sim_df.describe())
def weighted_hybrid_distances(X, weights=1):
    """
    Calculate a weighted hybrid distance matrix from multiple distance metrics
    
    Parameters:
    X: feature matrix (normalized)
    weights: list of weights for [euclidean, manhattan, cosine]
    
    Returns:
    Weighted hybrid distance matrix
    """
    # Calculate individual distance matrices
    cos_dist = 1 - cosine_similarity(X)
    
    # Normalize each distance matrix to [0,1] range for fair weighting
    # For this simple example with normalized features, we'll just use the raw distances
    
    # Return weighted combination
    return (weights * cos_dist)

# Calculate hybrid distance matrix with scikit-learn
hybrid_matrix = weighted_hybrid_distances(feature_matrix_pca)
hybrid_df = pd.DataFrame(hybrid_matrix, index=chick_data['feed_name'], columns=chick_data['feed_name'])

print("\nHybrid Distance Matrix:")
print(hybrid_df.round(4))

def recommend_similar_feeds(feed_name, cosine_sim_df, top_n=5):
    """
    Recommend similar feeds based on cosine similarity scores.
    
    Parameters:
    - feed_name: The name of the feed to find similar feeds for.
    - cosine_sim_df: DataFrame containing cosine similarity scores.
    - top_n: Number of top similar feeds to recommend.
    
    Returns:
    - List of recommended feed names.
    """
    """if feed_name not in cosine_sim_df.index:
        raise ValueError(f"Feed '{feed_name}' not found in the dataset.")
    
    # Get the similarity scores for the specified feed
    sim_scores = cosine_sim_df[feed_name].sort_values(ascending=False)
    
    # Get the top N similar feeds (excluding the feed itself)
    similar_feeds = sim_scores.index[1:top_n + 1].tolist()"""

    if feed_name not in cosine_sim_df.index:
        return "Feed not found in the database."
    
    # Get the distances for the specified feed
    distances = cosine_sim_df.loc[feed_name].copy()
    # Set the distance to itself as infinity so it's not selected
    distances[feed_name] = float('inf')
    # Get indices of n most similar feeds (smallest distances)
    most_similar_idx = distances.nsmallest(top_n).index
    # Return feed names and their distances
    similar_feeds = [f"{name} (distance: {distances[name]:.4f})" for name in most_similar_idx]
    return similar_feeds
# Example usage
feed_name = 'PeepNourish'
similar_feeds = recommend_similar_feeds(feed_name, cosine_sim_df, top_n=5)
print(similar_feeds)