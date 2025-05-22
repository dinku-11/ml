import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
# Load data
data = pd.read_csv("Breast Cancer Dataset.csv")
pd.set_option('display.max_columns', None)
# Basic exploration
print(data.head())
print(data.shape)
print(data.info())
print(data.isnull().sum())
# Preprocessing
df = data.drop(['id'], axis=1)
df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0}) # Malignant:1, Benign:0
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Model training
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, y_train)
# Functions for entropy, conditional entropy, information gain
import math
def entropy(column):
    counts = column.value_counts()
    probabilities = counts / len(column)
    return -sum(probabilities * probabilities.apply(math.log2))
def conditional_entropy(data, X, target):
    feature_values = data[X].unique()
    weighted_entropy = 0
    for value in feature_values:
        subset = data[data[X] == value] # <-- Corrected here
        weighted_entropy += (len(subset) / len(data)) * entropy(subset[target])
    return weighted_entropy
def information_gain(data, X, target):
    total_entropy = entropy(data[target])
    feature_conditional_entropy = conditional_entropy(data, X, target)
    return total_entropy - feature_conditional_entropy
# Calculate and print Information Gain for each feature
for feature in X.columns:
    ig = information_gain(df, feature, 'diagnosis')
    print(f"Information Gain for {feature}: {ig}")
# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=X.columns.tolist(), class_names=['Benign',
'Malignant'])
plt.show()
# Model prediction
y_pred = model.predict(X_test)
# Evaluate
accuracy = accuracy_score(y_test, y_pred) * 100
classification_rep = classification_report(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)
# Prediction for new sample
new = [[12.5, 19.2, 80.0, 500.0, 0.085, 0.1, 0.05, 0.02, 0.17, 0.06,
0.4, 1.0, 2.5, 40.0, 0.006, 0.02, 0.03, 0.01, 0.02, 0.003,
16.0, 25.0, 105.0, 900.0, 0.13, 0.25, 0.28, 0.12, 0.29, 0.08]]
# Convert to DataFrame to match training data
new_df = pd.DataFrame(new, columns=X.columns)
y_new_pred = model.predict(new_df)
if y_new_pred[0] == 0:
    print("Prediction: Benign")
else:
    print("Prediction: Malignant")
