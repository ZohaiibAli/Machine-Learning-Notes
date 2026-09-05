# Support Vector Machine (SVM)

## What is SVM?

Support Vector Machine (SVM) is a supervised machine learning algorithm used for **classification** and **regression** tasks. Its main goal is to find the best possible boundary (called a **hyperplane**) that separates data points belonging to different classes.

## Key Concepts

### 1. Hyperplane
A decision boundary that separates different classes in the feature space.
- In 2D, it's a line.
- In 3D, it's a plane.
- In higher dimensions, it's called a hyperplane.

### 2. Support Vectors
The data points closest to the hyperplane. These points directly influence the position and orientation of the hyperplane. Removing them would change the decision boundary.

### 3. Margin
The distance between the hyperplane and the nearest data points (support vectors) from each class. SVM tries to **maximize this margin** — a larger margin generally means better generalization on unseen data.

### 4. Kernel Trick
Real-world data is often not linearly separable. SVM uses **kernel functions** to transform data into a higher-dimensional space where a linear separation becomes possible.

Common kernels:
| Kernel | Use Case |
|--------|----------|
| Linear | When data is linearly separable |
| Polynomial | For curved decision boundaries |
| RBF (Radial Basis Function) | Most commonly used, handles non-linear data well |
| Sigmoid | Similar to neural network activation |

### 5. Hyperparameters
- **C (Regularization)**: Controls trade-off between maximizing margin and minimizing classification error. Low C = wider margin, more tolerance for misclassification. High C = stricter classification, smaller margin.
- **Gamma**: Defines how far the influence of a single data point reaches (used with RBF kernel). Low gamma = far reach, high gamma = close reach (can overfit).

## How SVM Works (Step-by-Step)

1. Plot data points in n-dimensional space (n = number of features).
2. Find the hyperplane that best separates the classes.
3. Maximize the margin between the hyperplane and the nearest points (support vectors).
4. If data isn't linearly separable, apply a kernel to project it into higher dimensions.
5. Classify new data points based on which side of the hyperplane they fall on.

## Types of SVM

- **Linear SVM**: Used when data can be separated with a straight line/plane.
- **Non-linear SVM**: Used when data requires kernel functions for separation.
- **SVR (Support Vector Regression)**: The regression variant of SVM.

## Uses / Applications of SVM

- **Text and Document Classification** — spam detection, sentiment analysis, topic categorization
- **Image Classification** — object detection, handwriting recognition (e.g., digit recognition)
- **Bioinformatics** — protein classification, cancer/tumor detection from gene expression data
- **Face Detection** — identifying faces vs. non-faces in images
- **Anomaly/Outlier Detection** — fraud detection in finance
- **Handwritten Character Recognition** — OCR systems
- **Bioinformatics & Medical Diagnosis** — disease prediction based on patient data

## Advantages

- Effective in high-dimensional spaces
- Works well when there's a clear margin of separation
- Memory efficient (uses only support vectors, not the whole dataset)
- Versatile via different kernel functions

## Disadvantages

- Not ideal for very large datasets (training can be slow)
- Performs poorly with overlapping classes / noisy data
- Choosing the right kernel and tuning hyperparameters (C, gamma) can be tricky
- Doesn't directly provide probability estimates (needs extra computation)

## Example (Python - scikit-learn)

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load dataset
X, y = load_iris(return_X_y=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train SVM model
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
print("Accuracy:", model.score(X_test, y_test))
```

## Summary

SVM is a powerful and versatile algorithm best suited for classification problems with clear margins of separation, especially in high-dimensional spaces like text and image data. The kernel trick makes it flexible enough to handle both linear and non-linear problems.
