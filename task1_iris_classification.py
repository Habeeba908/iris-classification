"""
Task 1: Iris Flower Classification
Classify iris flowers into Setosa, Versicolor, Virginica using a K-Nearest Neighbors classifier.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)

# 1. Load the dataset
iris = load_iris(as_frame=True)
df = iris.frame
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
print("First 5 rows:\n", df.head(), "\n")
print("Shape:", df.shape)
print("Missing values:\n", df.isnull().sum(), "\n")
print(df.describe(), "\n")

# 2. Visual exploration
sns.pairplot(df.drop(columns="target"), hue="species")
plt.savefig("iris_pairplot.png", dpi=150, bbox_inches="tight")
plt.show()

df.drop(columns="target").hist(figsize=(8, 6), edgecolor="black")
plt.suptitle("Iris Feature Distributions")
plt.savefig("iris_histograms.png", dpi=150, bbox_inches="tight")
plt.show()

# 3. Split into train/test sets
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Preprocess (scale features - helps KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train the classifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# 6. Evaluate
y_pred = model.predict(X_test_scaled)
print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred, average="macro"), 4))
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=iris.target_names))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
ConfusionMatrixDisplay(cm, display_labels=iris.target_names).plot(cmap="Blues")
plt.title("Iris - Confusion Matrix")
plt.savefig("iris_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()