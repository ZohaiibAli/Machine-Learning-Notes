# K-Nearest Neighbors (KNN)

KNN is a **supervised learning algorithm** used for both **classification** and **regression**. Unlike models like Linear Regression that learn a fixed equation, KNN is a **lazy learner** — it doesn't "learn" a formula during training. Instead, it stores the entire training dataset and makes predictions by looking at the **k closest data points** (neighbors) to a new input.

## Core Idea

> "You are similar to the people/things closest to you."

- For **classification**: predict the **majority class** among the k nearest neighbors.
- For **regression**: predict the **average (or weighted average)** of the k nearest neighbors' values.

## How It Works (Step by Step)

1. Choose a value for **k** (number of neighbors to consider).
2. Calculate the **distance** between the new data point and every point in the training data (commonly **Euclidean distance**).
3. Select the **k closest points**.
4. **Classification** → take a majority vote among those k points' labels.
   **Regression** → take the mean of those k points' target values.

### Distance Formula (Euclidean — most common)
```
distance = sqrt( (x1 - x2)² + (y1 - y2)² + ... )
```

## Choosing K

| K value | Effect |
|---|---|
| Too small (e.g., k=1) | Very sensitive to noise, overfits — decision boundary is jagged |
| Too large (e.g., k=n) | Oversmoothed, underfits — ignores local patterns, biased toward majority class |
| Just right | Usually found via experimentation (e.g., trying a range of k and checking accuracy) |

- **k is usually chosen as odd** (for classification) to avoid tie votes between two classes.

## Why Feature Scaling Matters
KNN relies entirely on **distance calculations**. If one feature has a much larger scale than another (e.g., "salary" in thousands vs. "age" in tens), it will dominate the distance calculation and distort results. That's why **StandardScaler** (or MinMaxScaler) is almost always applied before using KNN.

## Code Example (Python + scikit-learn)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------- Example dataset ----------
data = pd.DataFrame({
    'feature1': [2, 4, 4, 6, 8, 8, 10, 12, 14, 16],
    'feature2': [3, 2, 5, 4, 6, 8, 9, 11, 10, 13],
    'label':    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]   # binary classification target
})

X = data[['feature1', 'feature2']]
y = data['label']

# ---------- Train-test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- Feature scaling (important for KNN!) ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- Train KNN model ----------
knn = KNeighborsClassifier(n_neighbors=3)  # k = 3
knn.fit(X_train_scaled, y_train)

# ---------- Predict & Evaluate ----------
y_pred = knn.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ---------- Predict a new sample ----------
new_point = scaler.transform([[7, 7]])
print("Predicted class for new point:", knn.predict(new_point)[0])
```

### Finding the Best K (Elbow Method)

The elbow method tests a range of K values and plots how error changes, so you can pick K empirically instead of guessing.

**Why it's needed:**
- **Too small K** (e.g., K=1) → overfits, very sensitive to noise
- **Too large K** → underfits, oversmoothed decision boundary
- The elbow method finds the middle ground.

**How it works:**
1. Loop through a range of K values (e.g., 1 to 30).
2. For each K, train a KNN model and calculate the **error rate** on the test/validation set.
3. Plot **K vs Error Rate**.
4. Look for the **"elbow point"** — where error stops dropping sharply and flattens out. That K is usually the best choice.

Think of it like a cost-benefit curve: error drops fast initially as K increases (less noise sensitivity), but after a point, increasing K further gives diminishing returns or starts hurting performance — that bend is the "elbow."

```python
import numpy as np
import matplotlib.pyplot as plt

error_rates = []
k_range = range(1, 30)

for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    error = 1 - accuracy_score(y_test, pred)
    error_rates.append(error)

plt.figure(figsize=(8, 5))
plt.plot(k_range, error_rates, marker='o', linestyle='--', color='blue')
plt.title('Elbow Method for Optimal K')
plt.xlabel('K value')
plt.ylabel('Error Rate')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# Automatically pick the K with lowest error
best_k = k_range[np.argmin(error_rates)]
print("Best K:", best_k)
```

**Reading the plot:**
- **Y-axis (Error Rate)** starts high at K=1 (overfitting/noisy), drops as K increases.
- At some point, the curve **flattens out** — that's the elbow.
- Picking K right at (or just after) the elbow gives a good bias-variance balance.
- If multiple K values give near-identical low error, prefer the **smaller K** (simpler, faster) or an **odd K** for binary classification (avoids tie votes).

**Alternative: Cross-Validation Instead of a Single Test Split**

A single train-test split can give a noisy elbow curve. A more robust approach averages results across multiple folds:

```python
from sklearn.model_selection import cross_val_score

cv_scores = []
for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append(1 - scores.mean())  # error = 1 - accuracy

best_k_cv = k_range[np.argmin(cv_scores)]
print("Best K (cross-validated):", best_k_cv)
```
This averages results across 5 folds, giving a more stable estimate of the true elbow point — less prone to a single lucky/unlucky train-test split.

## KNN for Regression

Same idea, but instead of majority vote, it averages neighbor values:

```python
from sklearn.neighbors import KNeighborsRegressor

knn_reg = KNeighborsRegressor(n_neighbors=3)
knn_reg.fit(X_train_scaled, y_train)
y_pred = knn_reg.predict(X_test_scaled)
```

## Key Characteristics

| Aspect | Detail |
|---|---|
| Learning type | Supervised (Classification & Regression) |
| Learning style | Lazy learner — no explicit training phase, just stores data |
| Sensitive to feature scale | Yes — always scale features first |
| Sensitive to irrelevant features | Yes — extra noisy features distort distance |
| Interpretability | Simple, intuitive |
| Prediction speed | Slow on large datasets (must compute distance to every point) |
| Works well with | Small-to-medium datasets, well-separated classes |

## Advantages
- Simple to understand and implement
- No training phase (fast to "fit")
- Naturally handles multi-class classification

## Disadvantages
- Slow prediction time on large datasets (distance to every point)
- Sensitive to irrelevant/unscaled features
- Struggles with high-dimensional data (**curse of dimensionality**)
- Requires storing the entire training dataset in memory

## When to Use KNN
- Small-to-medium sized datasets
- When decision boundaries are irregular/non-linear
- As a quick baseline model before trying more complex algorithms

## Tech Stack
- Python
- pandas, NumPy
- scikit-learn (`KNeighborsClassifier`, `KNeighborsRegressor`, `StandardScaler`)
- Matplotlib (for elbow method visualization)
