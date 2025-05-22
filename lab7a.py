import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
df=pd.read_csv("C:\\Users\\ACER\\OneDrive\\Documents\\datasets\\housing.csv")
print(df)
df['Target']=df.median_house_value
print(df.info())
print(df.head())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.describe())
plt.figure(figsize=(12,6))
df.hist(figsize=(12,6),bins=30,edgecolor='black')
plt.suptitle("Feature distributions")
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.title("Boxplots")
plt.show()
plt.figure(figsize=(12,6))
corr_matrix=df.corr()
sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',fmt='.2f')
plt.title("Feature correlation map")
sns.pairplot(df[['median_income','housing_median_age','total_bedrooms','Target']],diag_kind='kde')
plt.show()
