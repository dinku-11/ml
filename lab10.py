import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
from sklearn.metrics import silhouette_score 
import warnings 
warnings.filterwarnings('ignore') 
data = pd.read_csv("Breast_Cancer_Dataset.csv") 
data.head() 
data.info() 
data.diagnosis.unique() 
data.isnull().sum() 
data.duplicated().sum() 
df = data.drop(['id'], axis=1) 
df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0}) # Malignant:1, Benign:0 
#dropped the Diagnosis (target) since clustering is unsupervised. 
df.drop(columns=["diagnosis"], inplace=True) # Removing Target 
# Standardize the data 
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(df) 
# Apply PCA for Dimensionality Reduction 
pca = PCA(n_components=2) # Reduce to 2 dimensions for visualization 
X_pca = pca.fit_transform(X_scaled) 
# Check explained variance ratio 
explained_variance = pca.explained_variance_ratio_ 
total_explained_variance = np.sum(explained_variance) 
print(f"Variance explained by PC1: {explained_variance[0]:.4f}") 
print(f"Variance explained by PC2: {explained_variance[1]:.4f}") 
#Apply K-Means Clustering with the optimal k (usually where elbow occurs, k=2) 
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10) 
clusters = kmeans.fit_predict(X_pca) 
# Step 7: Visualize the Clusters 
plt.figure(figsize=(8, 6)) 
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", alpha=0.6) 
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200) 
plt.xlabel("Principal Component 1") 
plt.ylabel("Principal Component 2") 
plt.title("K-Means Clustering after PCA") 
plt.show() 
