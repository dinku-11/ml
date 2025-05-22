import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load California Housing dataset
# Alternatively, if not using sklearn's fetch_california_housing:
df = pd.read_csv("C:\\Users\\ACER\\OneDrive\\Documents\Datasets[1]\\housing.csv")  # Make sure this file exists in the correct path

# Add target column for consistency
df['Target'] = df['median_house_value']  # Rename for clarity

# Display basic dataset information
print("\nBasic Information about Dataset:")
print(df.info())  # Overview of the dataset structure

print("\nFirst Five Rows of Dataset:")
print(df.head())  # Display the first few records

print("\nSummary Statistics:")
print(df.describe())  # Numerical summaries of the data

# Check for missing values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Histogram for feature distribution
plt.figure(figsize=(12, 8))
df.hist(figsize=(12, 8), bins=30, edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

df_num = df.drop('ocean_proximity', axis=1)

# Boxplot to identify outliers in numerical features
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_num)
plt.xticks(rotation=45)
plt.title("Boxplots of Features to Identify Outliers")
plt.show()

# Correlation matrix heatmap
plt.figure(figsize=(10, 6))
corr_matrix = df_num.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

# Pairplot to visualize relationships between selected features and the target
sns.pairplot(df[['median_income', 'housing_median_age', 'total_bedrooms', 'Target']], diag_kind='kde')
plt.show()
