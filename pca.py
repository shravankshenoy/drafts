import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Sample data
data = {
    'feature1': [2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2.0, 1.0, 1.5, 1.1],
    'feature2': [2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9],
    'feature3': [1.2, 0.5, 1.8, 1.0, 1.6, 1.5, 0.5, 0.7, 1.0, 0.6]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Initialize PCA and fit it to the data
pca = PCA(n_components=2)  # Reduce it to 2 dimensions
principal_components = pca.fit_transform(scaled_data)

# Create a DataFrame with the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Compute the 0th, 25th, 50th, 75th, and 100th percentiles for each column
percentiles = [0, 25, 50, 75, 100]
vectors = {}

for column in pca_df.columns:
    # Calculate percentiles
    vector = np.percentile(pca_df[column], percentiles)
    # Normalize to get the unit vector
    unit_vector = vector / np.linalg.norm(vector)
    vectors[column] = unit_vector

# Display the results
for column, unit_vector in vectors.items():
    print(f"Unit vector for {column}: {unit_vector}")
