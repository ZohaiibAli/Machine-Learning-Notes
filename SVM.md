# Support Vector Machines (SVM)

SVM is a **supervised learning algorithm** used mainly for **classification** (and also regression, called SVR). Its core idea: find the **best possible boundary (hyperplane)** that separates classes with the **maximum margin** — the widest possible gap between classes.

## Core Idea

> "Don't just find *a* line that separates the classes — find the line that separates them with the biggest possible safety margin."

- The **hyperplane** is the decision boundary (a line in 2D, a plane in 3D, a hyperplane in higher dimensions).
- **Support vectors** are the data points closest to the hyperplane — they "support" (define) where the boundary sits. Only these points matter for the final boundary; all other points can move around without changing it.
- **Margin** = the distance between the hyperplane and the nearest support vectors from each class. SVM tries to **maximize** this margin.

## Key Concepts

### 1. Hard Margin vs Soft Margin
- **Hard margin**: assumes data is perfectly separable — no misclassifications allowed. Rarely realistic.
- **Soft margin**: allows some misclassifications to get a better overall boundary (more robust to noisy/overlapping data). Controlled by the **`C` parameter**.

| C value | Effect |
|---|---|
| Small C | Wider margin, tolerates more misclassification (simpler boundary, less overfitting) |
| Large C | Narrower margin, tries hard to classify every point correctly (can overfit) |

### 2. The Kernel Trick
Real-world data is often **not linearly separable** (a straight line/plane can't separate the classes). SVM solves this using **kernels** — functions that project data into a higher-dimensional space where a linear separator *can* work, without actually computing that transformation explicitly (computationally efficient).

| Kernel | Use case |
|---|---|
| `linear` | Data is (roughly) linearly separable |
| `poly` (polynomial) | Curved boundaries, moderate complexity |
| `rbf` (Radial Basis Function) | Most common default — handles complex, non-linear boundaries well |
| `sigmoid` | Less common, behaves like a neural network activation |

### 3. Gamma (for `rbf`/`poly` kernels)
Controls how far the influence of a single training point reaches.

| Gamma value | Effect |
|---|---|
| Low gamma | Far-reaching influence, smoother boundary (can underfit) |
| High gamma | Very local influence, boundary hugs individual points closely (can overfit) |

## Why Feature Scaling Matters
Like KNN, SVM is **distance-based** (it relies on margins between points). Features on very different scales will distort the margin calculation, so always apply **StandardScaler** before fitting.

## Code Example (Python + scikit-learn)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------- Example dataset ----------
data = pd.DataFrame({
    'feature1': [1, 2, 2, 3, 6, 7, 8, 8, 9, 10],
    'feature2': [2, 3, 1, 2, 8, 7, 9, 6, 8, 9],
    'label':    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]   # binary classification target
})

X = data[['feature1', 'feature2']]
y = data['label']

# ---------- Train-test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- Feature scaling (important for SVM!) ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- Train SVM model ----------
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train_scaled, y_train)

# ---------- Predict & Evaluate ----------
y_pred = svm_model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ---------- Predict a new sample ----------
new_point = scaler.transform([[5, 5]])
print("Predicted class for new point:", svm_model.predict(new_point)[0])
```

### Tuning Hyperparameters with GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear']
}

grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_scaled, y_train)

print("Best Parameters:", grid.best_params_)
print("Best Cross-Val Accuracy:", grid.best_score_)
```
`GridSearchCV` tries every combination of the given parameters using cross-validation and returns the best-performing combination — much more reliable than guessing `C`/`gamma`/`kernel` manually.

## SVM for Regression (SVR)

```python
from sklearn.svm import SVR

svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr_model.fit(X_train_scaled, y_train)
y_pred = svr_model.predict(X_test_scaled)
```
`epsilon` defines a margin of tolerance where no penalty is given for errors — points within this margin around the predicted line don't count as errors.

## Key Characteristics

| Aspect | Detail |
|---|---|
| Learning type | Supervised (Classification & Regression) |
| Decision boundary | Maximum-margin hyperplane |
| Key points used | Only support vectors matter, not all data points |
| Sensitive to feature scale | Yes — always scale features first |
| Handles non-linear data | Yes, via kernel trick |
| Interpretability | Lower for non-linear kernels (harder to visualize in high dimensions) |
| Performance on large datasets | Can get slow — training complexity grows with dataset size |

## Advantages
- Effective in high-dimensional spaces (even when features > samples)
- Works well with clear margin of separation
- Kernel trick handles non-linear boundaries without manual feature engineering
- Robust to overfitting, especially in high-dimensional space (with proper `C`)

## Disadvantages
- Doesn't scale well to very large datasets (training can be slow)
- Choosing the right kernel/`C`/`gamma` requires tuning (e.g., GridSearchCV)
- Less interpretable than simpler models like Linear Regression
- Doesn't directly provide probability estimates (needs extra calibration, e.g., `probability=True`)

## When to Use SVM
- Medium-sized datasets with clear class separation
- High-dimensional data (e.g., text classification, gene data)
- When you need a robust classifier and can afford some hyperparameter tuning

## Tech Stack
- Python
- pandas, NumPy
- scikit-learn (`SVC`, `SVR`, `StandardScaler`, `GridSearchCV`)
- Matplotlib (for visualizing decision boundaries, if plotted in 2D)
