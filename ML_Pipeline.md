# Scikit-learn ML Pipelines

This explains the **pipeline pattern** used in most real-world ML workflows — not tied to any specific dataset. The same structure applies whether you're predicting house prices, customer churn, exam scores, or anything else with a mix of numeric and categorical features.

## Why Pipelines Exist

Raw data is almost never ready to feed into a model directly:
- Numeric columns may have **missing values** or **wildly different scales**.
- Categorical (text) columns need to be converted into numbers.
- If you preprocess manually and separately for train/test data, it's easy to make mistakes (e.g., accidentally leaking test data statistics into training — called **data leakage**).

A `Pipeline` chains every step (cleaning → transforming → modeling) into **one object**, so:
- `.fit()` runs every step in order, learning parameters (like scaling values) *only* from training data.
- `.predict()` automatically re-applies the exact same transformations to new data — no manual repetition, no leakage.

## 1. Data Loading & Cleaning (General Pattern)

```python
df = pd.read_csv("your_data.csv")
df.info()                 # column dtypes and non-null counts
df.isnull().sum()         # missing values per column
df.duplicated().sum()     # duplicate row count
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
```

**Why this matters:**
- `df.info()` / `df.isnull().sum()` — always check column types and completeness before transforming anything.
- Duplicate rows can bias a model by over-weighting repeated examples — drop them.
- `reset_index(drop=True)` re-numbers the index after dropping rows, avoiding alignment issues later (e.g., merging predictions back with original rows).

## 2. Feature/Target Split & Train-Test Split

```python
X = df.drop("target_column", axis=1)   # all feature columns
y = df["target_column"]                # what you want to predict

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

- `X` = **features** (independent variables), `y` = **target** (what the model learns to predict).
- `train_test_split` holds out a portion (commonly 20%) as an **unseen test set**, to check how well the model generalizes — not just how well it memorized training data.
- `random_state=42` fixes the random seed so the split is reproducible every time you run the code.

## 3. Preprocessing Pipelines (Numeric vs Categorical)

Linear models can't consume raw numeric-scale differences or text directly. Two sub-pipelines handle each type:

```python
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
])
```

### Numeric pipeline
1. **`SimpleImputer(strategy="median")`** — fills missing numeric values with the column's median. Median is preferred over mean because it's robust to outliers (one extreme value won't skew the fill).
2. **`StandardScaler()`** — rescales each feature to mean 0, std 1. Important for linear models: if one feature ranges in the thousands and another is a small integer count, the larger-scale feature can dominate the model unfairly and slow convergence.

### Categorical pipeline
1. **`SimpleImputer(strategy="most_frequent")`** — fills missing categories with the mode, since averages don't apply to text.
2. **`OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)`** — turns each category into binary (0/1) columns.
   - `drop="first"` avoids the **dummy variable trap** (perfect multicollinearity — one dropped column's value can always be inferred from the rest).
   - `handle_unknown="ignore"` prevents crashes if new/unseen categories appear later — they're just encoded as all zeros.
   - `sparse_output=False` returns a dense array, simpler to inspect and compatible with later steps like `PolynomialFeatures`.

## 4. Combining Both with `ColumnTransformer`

```python
preprocessor = ColumnTransformer([
    ("num", num_pipeline, make_column_selector(dtype_include=np.number)),
    ("cat", cat_pipeline, make_column_selector(dtype_include=object))
])
```

`ColumnTransformer` applies different preprocessing to different columns **in one step**, then combines the results into a single feature matrix.

- `make_column_selector(dtype_include=np.number)` — auto-selects all numeric columns.
- `make_column_selector(dtype_include=object)` — auto-selects all text/categorical columns.
- Using **selectors instead of hardcoded column names** makes the pipeline reusable across different datasets with similar structure — a key reason this pattern generalizes so well.

## 5. Model Comparison: Linear vs. Polynomial (Bias–Variance Tradeoff)

```python
results = {}

for degree in [1, 2, 3]:
    model_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("model", LinearRegression())
    ])
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[degree] = (r2, rmse)
    print(f"Degree {degree} -> R2: {r2:.4f}, RMSE: {rmse:.2f}")
```

- **`PolynomialFeatures(degree=d)`** generates polynomial/interaction terms (e.g., degree 2 adds squared terms and pairwise products) so a linear model can capture non-linear relationships.
- `include_bias=False` skips an extra intercept column, since `LinearRegression` fits its own intercept anyway.
- Comparing degrees this way illustrates the **bias–variance tradeoff**:
  - **Degree 1 (plain linear)** — simplest, may underfit if relationships are curved (high bias).
  - **Higher degrees (2, 3...)** — can fit training data very closely but often **overfit**: great training score, worse performance on the unseen test set (high variance).
- **General rule of thumb:** pick the model complexity that performs best on the **test set**, not the training set — that's the one that will generalize to real, unseen data.

## 6. Final Pipeline Assembly

```python
best_degree = 1  # whichever degree performed best on the test set

model_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("poly", PolynomialFeatures(degree=best_degree, include_bias=False)),
    ("model", LinearRegression())
])

model_pipeline.fit(X_train, y_train)
```

This bundles preprocessing + feature expansion + modeling into **one object**. Benefit: `.fit()` trains every step in the right order, and `.predict()` on new raw data automatically applies the same imputation, scaling, and encoding used during training — no manual preprocessing needed at inference time.

## 7. Model Evaluation

```python
y_pred = model_pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
```

- **R² (coefficient of determination)** — proportion of variance in the target explained by the model; closer to 1 is better.
- **RMSE (root mean squared error)** — average prediction error, in the same units as the target; lower is better.

A moderate R² (e.g., ~0.6–0.7) means the model captures a real trend but leaves room for improvement — common next steps: regularization, more/better features, or a non-linear model.

## 8. Predictive Function (General Pattern)

```python
def predict_target(feature1, feature2, feature3, ...):
    input_df = pd.DataFrame([{
        "feature1": feature1,
        "feature2": feature2,
        "feature3": feature3,
        # ...must match the exact column names used in training
    }])
    prediction = model_pipeline.predict(input_df)
    return prediction[0]
```

Key points that apply to **any** pipeline like this:
1. Wrap raw input values into a **single-row DataFrame**.
2. Column **names must exactly match** what the pipeline was trained on — `ColumnTransformer`/`OneHotEncoder` match by name/dtype, not position.
3. Passing it through `model_pipeline.predict()` automatically re-applies imputation, scaling, and encoding.
4. Returns a single scalar prediction (`prediction[0]`).

## Pipeline Architecture Summary (Generic)

```
Raw DataFrame (X)
      │
      ▼
ColumnTransformer ("preprocessing")
 ├── Numeric columns  → SimpleImputer(median) → StandardScaler
 └── Categorical cols → SimpleImputer(mode)   → OneHotEncoder(drop first)
      │
      ▼
PolynomialFeatures(degree=d)   (optional, for non-linear fit)
      │
      ▼
Model (e.g., LinearRegression)
      │
      ▼
Prediction
```

This exact skeleton works for **any tabular regression problem** — just swap the target column, feature columns, and optionally the final model (e.g., `Ridge`, `RandomForestRegressor`) without changing the overall structure.

## Key Concepts Reference Table

| Concept | Purpose |
|---|---|
| `train_test_split` | Separates data for unbiased evaluation on unseen samples |
| `SimpleImputer` | Handles missing values (median for numeric, mode for categorical) |
| `StandardScaler` | Normalizes numeric feature scales for linear models |
| `OneHotEncoder` | Converts categorical text into numeric binary columns |
| `ColumnTransformer` | Applies different preprocessing per column type in one step |
| `Pipeline` | Chains preprocessing + modeling into one fit/predict object, preventing data leakage |
| `PolynomialFeatures` | Adds non-linear terms so a linear model can fit curved relationships |
| `LinearRegression` | Fits a linear relationship between features and target |
| `R² / RMSE` | Metrics for judging prediction accuracy and error magnitude |

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn (`Pipeline`, `ColumnTransformer`, preprocessing, `LinearRegression`, metrics)
- Matplotlib (for visualization)

## Possible Extensions

- Try regularized linear models (Ridge/Lasso) to reduce overfitting risk, especially with polynomial features.
- Use cross-validation instead of a single train/test split for more reliable model comparison.
- Engineer new features (ratios, interactions) or try tree-based models (Random Forest, Gradient Boosting) for non-linear relationships.
- Persist the trained pipeline with `joblib`/`pickle` for reuse without retraining.
