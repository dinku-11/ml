import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns     
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load Seaborn dataset names (optional)
sns.get_dataset_names()

# Load the dataset
data = sns.load_dataset('mpg')
df = data.copy()

# Fill missing values in 'horsepower' with median
df['horsepower'].fillna(df['horsepower'].median(), inplace=True)

# Data cleaning: check for missing values
df.isnull().sum()

# Separate numerical and categorical columns
numerical = df.select_dtypes(include=['int', 'float']).columns
categorical = df.select_dtypes(include=['object']).columns
print(numerical)
print(categorical)

# Correlation heatmap
corr_data = df[numerical].corr(method='pearson')
plt.figure(figsize=(10, 8))
sns.heatmap(corr_data, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.xticks(rotation=90, ha='right')
plt.yticks(rotation=0)
plt.title("Correlation Matrix Heatmap")
plt.show()

# Define input and target
X = df[['horsepower']]  # Independent variable
y = df['mpg']           # Dependent variable

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create polynomial features
degree = 2
poly = PolynomialFeatures(degree)
X_poly_train = poly.fit_transform(X_train)

# Train the model
model = LinearRegression()
model.fit(X_poly_train, y_train)

# Predict on test data
X_poly_test = poly.transform(X_test)
y_pred = model.predict(X_poly_test)

# Plot original data and polynomial regression line
plt.scatter(X, y, color='blue', label='Data')
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_range_poly = poly.transform(X_range)
y_range_pred = model.predict(X_range_poly)
plt.plot(X_range, y_range_pred, color='red', label='Polynomial Fit')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.legend()
plt.title(f'Polynomial Regression (degree {degree})')
plt.show()

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Display evaluation results
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'R-squared (R²): {r2:.2f}')
