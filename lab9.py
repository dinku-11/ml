import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Step 1: Load Olivetti Faces
data = fetch_olivetti_faces()
images = data.images
targets = data.target
# Step 2: Show 10 sample images
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
fig.suptitle("Sample Images from Olivetti Faces Dataset", fontsize=16)
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i], cmap='gray')
    ax.set_title(f"Person ID: {targets[i]}")
    ax.axis('off')
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()
# Step 3: Flatten the image data (64x64 -> 4096)
X = images.reshape((images.shape[0], -1))  # shape: (400, 4096)
y = targets
# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
# Step 5: Train Gaussian Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)
# Step 6: Predict on test data
y_pred = model.predict(X_test)
# Step 7: Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy of Naive Bayes: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# Create a DataFrame to compare actual vs predicted
comparison = pd.DataFrame({
    "Index": np.arange(len(y_test)),
    "Actual (y_test)": y_test,
    "Predicted (y_pred)": y_pred
})
print(comparison.head(20))  # Show first 20 rows
comparison["Match"] = comparison["Actual (y_test)"] == comparison["Predicted (y_pred)"]
print(comparison.head(20))
