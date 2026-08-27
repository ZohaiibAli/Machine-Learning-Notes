# Housing Prices Regression Pipeline

A scikit-learn pipeline that predicts house prices from structural and locational features (area, bedrooms, amenities, etc.) using Linear Regression, with a full preprocessing pipeline for numeric and categorical data.

## Dataset

The dataset (`Housing.csv`) contains 545 rows and 13 columns describing houses and their sale price:

| Column | Type | Description |
|---|---|---|
| `price` | numeric (target) | Sale price of the house |
| `area` | numeric | Plot/house area in square feet |
| `bedrooms`, `bathrooms`, `stories` | numeric | Counts of rooms/floors |
| `parking` | numeric | Number of parking spots |
| `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea` | categorical (yes/no) | Presence of a feature/amenity |
| `furnishingstatus` | categorical | `furnished`, `semi-furnished`, or `unfurnished` |

There were no missing values and the notebook still checks and removes duplicate rows before modeling, resetting the index afterward so downstream splitting/indexing behaves correctly.

## 1. Data Loading & Cleaning

```python
df = pd.read_csv("Housing.csv")
df.info()                 # column dtypes and non-null counts
df.isnull().sum()         # missing value check per column
df.duplicated().sum()     # count of duplicate rows
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
```

**Why this matters:**
- `df.info()` and `df.isnull().sum()` are standard exploratory steps to understand column types and confirm data completeness before any transformation.
- Duplicate rows can bias a model by over-weighting repeated examples, so they're dropped.
- `reset_index(drop=True)` re-numbers the DataFrame index from 0 after dropping rows, avoiding gaps that could cause alignment issues later (e.g., when merging predictions back with original rows).

## 2. Feature/Target Split & Train-Test Split

```python
X = df.drop("price", axis=1)   # all columns except the target
y = df["price"]                # the target variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

- `X` holds the **features** (independent variables), `y` holds the **target** (`price`), which is what the model learns to predict.
- `train_test_split` reserves 20% of the data as an unseen **test set** to evaluate the model's ability to generalize, while 80% (`X_train`, `y_train`) is used for fitting.
- `random_state=42` fixes the random seed so the split is reproducible — running the notebook again produces the exact same train/test rows.

## 3. Preprocessing Pipelines

Raw features can't go directly into a linear model — numeric features need to be on comparable scales, and categorical (text) features need to be converted to numbers. Two separate sub-pipelines handle this:

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
1. **`SimpleImputer(strategy="median")`** — fills any missing numeric values with the column's median. Median is preferred over mean because it's robust to outliers (e.g., an unusually large `area` value won't skew the fill value).
2. **`StandardScaler()`** — rescales each numeric feature to have mean 0 and standard deviation 1. This matters for linear models because features on very different scales (e.g., `area` in the thousands vs. `bedrooms` as small integers) can otherwise distort coefficient magnitudes and slow convergence.

### Categorical pipeline
1. **`SimpleImputer(strategy="most_frequent")`** — fills missing categorical values with the most common category (the mode), since medians/means don't apply to text categories.
2. **`OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)`** — converts each category into binary (0/1) columns so the model can use them numerically.
   - `drop="first"` removes one category per feature to avoid the **dummy variable trap** (perfect multicollinearity, since one column's value can be inferred from the rest).
   - `handle_unknown="ignore"` prevents the pipeline from crashing if the test set (or future input) contains a category never seen during training — it simply encodes it as all zeros instead of raising an error.
   - `sparse_output=False` returns a dense NumPy array instead of a sparse matrix, which is simpler to inspect and compatible with downstream steps like `PolynomialFeatures`.

## 4. Combining Pipelines with `ColumnTransformer`

```python
preprocessor = ColumnTransformer([
    ("num", num_pipeline, make_column_selector(dtype_include=np.number)),
    ("cat", cat_pipeline, make_column_selector(dtype_include=object))
])
```

`ColumnTransformer` applies different preprocessing to different columns of the same DataFrame **in one step**, then concatenates the results into a single feature matrix.

- `make_column_selector(dtype_include=np.number)` automatically selects all numeric columns (`area`, `bedrooms`, `bathrooms`, `stories`, `parking`) for the numeric pipeline.
- `make_column_selector(dtype_include=object)` automatically selects all text/categorical columns (`mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`, `furnishingstatus`) for the categorical pipeline.
- Using selectors instead of hardcoded column names makes the pipeline more robust to changes in the dataset's structure.

## 5. Model Comparison: Linear vs. Polynomial Regression

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

- **`PolynomialFeatures(degree=d)`** generates polynomial and interaction terms (e.g., for degree 2: squared terms and pairwise products of features) on top of the preprocessed features. This lets a linear model capture non-linear relationships.
- `include_bias=False` excludes the intercept column, since `LinearRegression` already fits its own intercept.
- The loop trains a full pipeline for degrees 1, 2, and 3, then compares them using two evaluation metrics:
  - **R² (coefficient of determination)** — the proportion of variance in `price` explained by the model; closer to 1 is better.
  - **RMSE (root mean squared error)** — the average prediction error, in the same units as `price`; lower is better.

### Result of the comparison

Degree 1 (plain Linear Regression) produced the best R² and lowest RMSE. Higher-degree polynomials (2 and 3) overfit the training data — they fit the training set well but generalize worse to the unseen test set, hurting real-world performance. Because of this, **degree 1 was chosen** as the final model.

This step demonstrates the classic **bias-variance tradeoff**: a simple model (degree 1) generalizes better here than a complex one (degree 2/3) that memorizes noise in the training data.

## 6. Final Model Pipeline

```python
best_degree = 1

model_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("poly", PolynomialFeatures(degree=best_degree, include_bias=False)),
    ("model", LinearRegression())
])

model_pipeline.fit(X_train, y_train)
```

This assembles the **entire workflow** — preprocessing, feature expansion, and modeling — into a single `Pipeline` object. Key benefit: calling `.fit()` once trains every step in the correct order, and calling `.predict()` on new raw data automatically applies the same imputation, scaling, and encoding used during training — no manual preprocessing needed at inference time.

## 7. Model Evaluation

```python
y_pred = model_pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
```

**Final results:**
- **R² Score:** ≈ 0.653 — the model explains about 65% of the variance in house prices.
- **RMSE:** ≈ 1,324,507 — on average, predictions deviate from actual prices by this amount (in the currency unit of the dataset).

These numbers indicate a moderate fit: useful as a baseline, but there's clear room for improvement (e.g., regularization, more features, or a non-linear model) if higher accuracy is needed.

## 8. Predictive Function

```python
def predict_price(area, bedrooms, bathrooms, stories, mainroad, guestroom,
                   basement, hotwaterheating, airconditioning, parking,
                   prefarea, furnishingstatus):
    input_df = pd.DataFrame([{
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "parking": parking,
        "prefarea": prefarea,
        "furnishingstatus": furnishingstatus
    }])
    prediction = model_pipeline.predict(input_df)
    return prediction[0]
```

This wraps the trained pipeline into a simple, reusable function:
1. Takes the raw feature values as individual arguments (matching what a user would naturally input).
2. Packages them into a single-row DataFrame with the **same column names** the pipeline was trained on — this is essential, since `ColumnTransformer` and `OneHotEncoder` match columns by name/dtype, not position.
3. Passes that DataFrame through the fitted `model_pipeline`, which automatically re-applies imputation, scaling, and encoding before predicting.
4. Returns a single scalar price prediction (`prediction[0]`).

The notebook demonstrates this two ways: once via interactive `input()` prompts (for manual testing), and once via a direct function call with hardcoded values, which returned a predicted price of **≈ 7,968,276** for a 7420 sq ft, 4-bedroom, 2-bathroom, 3-story house with air conditioning and a preferred-area location.

## Pipeline Architecture Summary

```
Raw DataFrame (X)
      │
      ▼
ColumnTransformer ("preprocessing")
 ├── Numeric columns  → SimpleImputer(median) → StandardScaler
 └── Categorical cols → SimpleImputer(mode)   → OneHotEncoder(drop first)
      │
      ▼
PolynomialFeatures(degree=1)
      │
      ▼
LinearRegression
      │
      ▼
Predicted price
```

## Key Concepts Used

| Concept | Purpose |
|---|---|
| `train_test_split` | Separates data for unbiased evaluation on unseen samples |
| `SimpleImputer` | Handles missing values (median for numeric, mode for categorical) |
| `StandardScaler` | Normalizes numeric feature scales for linear models |
| `OneHotEncoder` | Converts categorical text into numeric binary columns |
| `ColumnTransformer` | Applies different preprocessing per column type in one step |
| `Pipeline` | Chains preprocessing + modeling into one fit/predict object, preventing data leakage |
| `PolynomialFeatures` | Adds non-linear terms to let a linear model fit curved relationships |
| `LinearRegression` | Fits a linear relationship between features and price |
| `R² / RMSE` | Metrics for judging prediction accuracy and error magnitude |

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn (`Pipeline`, `ColumnTransformer`, preprocessing, `LinearRegression`, metrics)
- Matplotlib (imported for visualization)

## Possible Improvements

- Try regularized linear models (Ridge/Lasso) to reduce overfitting risk with polynomial features.
- Add cross-validation instead of a single train/test split for more reliable model comparison.
- Engineer additional features (e.g., area per bedroom) or try tree-based models (Random Forest, Gradient Boosting) for potentially better non-linear fit.
- Persist the trained pipeline with `joblib`/`pickle` for reuse without retraining.
