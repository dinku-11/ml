from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the Iris dataset
iris = load_iris()
features = iris.data  # The 4 features: Sepal Length, Sepal Width, Petal Length, Petal Width
target = iris.target  # The target class (species)

# Step 2: Standardize the features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Step 3: Apply PCA to reduce to 2 components
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_standardized)

# Step 4: Create a DataFrame for the reduced data
pca_df = pd.DataFrame(data=features_pca, columns=["Principal Component 1", "Principal Component 2"])
pca_df["Target"] = target

# Step 5: Visualize the PCA results
plt.figure(figsize=(8, 6))
for label, color in zip(iris.target_names, ["red", "green", "blue"]):
    plt.scatter(
        pca_df.loc[pca_df["Target"] == list(iris.target_names).index(label), "Principal Component 1"],
        pca_df.loc[pca_df["Target"] == list(iris.target_names).index(label), "Principal Component 2"],
        label=label,
        alpha=0.7,
        color=color
    )

plt.title("PCA on Iris Dataset (4 features to 2 features)", fontsize=14)
plt.xlabel("Principal Component 1", fontsize=12)
plt.ylabel("Principal Component 2", fontsize=12)
plt.legend(title="Species")
plt.grid()
plt.show()

# Explained variance
explained_variance = pca.explained_variance_ratio_
print("Explained Variance by each Principal Component:")
print("Principal Component 1: ", explained_variance[0])
print("Principal Component 2: ", explained_variance[1])
print("Total Variance Retained: ", sum(explained_variance))

# Pairplot visualization of the original Iris dataset
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target
sns.pairplot(iris_df, hue='species', palette='bright')
plt.suptitle("Pairplot of Original Iris Dataset", y=1.02)
plt.show()
