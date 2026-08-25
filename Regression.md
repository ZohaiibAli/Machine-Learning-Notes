# Linear Regression & Multiple Linear Regression

## 1. Simple Linear Regression
Ek **independent variable (X)** se ek **dependent variable (Y)** predict karna.

**Formula:**
```
y = m*x + b
```
- `m` = slope (coefficient)
- `b` = intercept

**Example:** Ghar ka size (sq ft) se price predict karna.

## 2. Multiple Linear Regression
**Do ya zyada independent variables** se ek dependent variable predict karna.

**Formula:**
```
y = b0 + b1*x1 + b2*x2 + ... + bn*xn
```

**Example:** Ghar ka size, bedrooms, aur age se price predict karna — teenon variables mil kar price decide karte hain.

### Key Differences

| Simple LR | Multiple LR |
|---|---|
| 1 independent variable | 2+ independent variables |
| Line fit hoti hai (2D) | Hyperplane fit hota hai (multi-dimensional) |
| `y = mx + b` | `y = b0 + b1x1 + b2x2 + ...` |

---

## 3. Code Example (Python + scikit-learn)

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ---------- Simple Linear Regression ----------
data_simple = pd.DataFrame({
    'size': [500, 750, 1000, 1250, 1500, 1750, 2000],
    'price': [50, 65, 80, 95, 110, 125, 140]
})

X_simple = data_simple[['size']]   # feature (2D shape needed)
y_simple = data_simple['price']    # target

model_simple = LinearRegression()
model_simple.fit(X_simple, y_simple)

print("Simple LR -> Slope:", model_simple.coef_[0])
print("Simple LR -> Intercept:", model_simple.intercept_)

# Predict price for 1300 sq ft house
predicted = model_simple.predict([[1300]])
print("Predicted price for 1300 sq ft:", predicted[0])


# ---------- Multiple Linear Regression ----------
data_multi = pd.DataFrame({
    'size': [500, 750, 1000, 1250, 1500, 1750, 2000],
    'bedrooms': [1, 2, 2, 3, 3, 4, 4],
    'age': [10, 8, 5, 6, 3, 2, 1],
    'price': [50, 68, 82, 100, 118, 135, 150]
})

X_multi = data_multi[['size', 'bedrooms', 'age']]
y_multi = data_multi['price']

X_train, X_test, y_train, y_test = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)

model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

print("\nMultiple LR -> Coefficients:", model_multi.coef_)
print("Multiple LR -> Intercept:", model_multi.intercept_)

y_pred = model_multi.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
```

---

## 4. Model Parameters: `coef_` and `intercept_`

### `model_simple.coef_[0]` → Slope (m)
- Batata hai `x` (size) mein **1 unit increase** hone se `y` (price) mein **kitna change** aayega.
- `.coef_` ek **array** hota hai (multiple features ho sakte hain), is liye `[0]` se pehla (simple LR mein sirf ek) coefficient nikalte hain.
- Example: agar value `0.062` ho, iska matlab size mein 1 sq ft badhne se price ~0.062 lakh barhta hai.

### `model_simple.intercept_` → Intercept (b)
- Ye woh value hai jab `x = 0` ho, yaani line **y-axis** ko kahan cut karti hai.
- `.intercept_` single number (scalar) hota hai, is liye `[0]` ki zaroorat nahi.

Dono mil kar equation banate hain:
```python
price = coef_[0] * size + intercept_
```

---

## 5. Evaluation Metrics: MSE & R² Score

### MSE (Mean Squared Error)
Predicted values aur actual values ke **error** ko measure karta hai.

**Formula:**
```
MSE = (1/n) * Σ(actual - predicted)²
```

- Har data point ka error nikalo (actual - predicted), usko square karo (negative/positive cancel na ho aur bade errors zyada penalize hon), phir average le lo.
- **Jitna kam MSE, utna accurate model.**
- MSE ki unit squared hoti hai (e.g. price²), is liye kabhi RMSE (root of MSE) use karte hain jo original unit mein wapas aata hai.

```python
actual =    [100, 150, 200]
predicted = [110, 140, 210]

errors = [(100-110)**2, (150-140)**2, (200-210)**2]  # [100, 100, 100]
mse = sum(errors) / len(errors)  # 100
```

### R² Score (Coefficient of Determination)
Batata hai model **kitna % variance** explain kar pa raha hai compared to sirf mean-based prediction.

**Formula:**
```
R² = 1 - (SS_residual / SS_total)
```
- `SS_residual` = model ke errors ka sum of squares
- `SS_total` = agar sirf mean use karte predictions ke liye, uska sum of squares

| R² Value | Matlab |
|---|---|
| 1.0 | Perfect fit — model sab variance explain kar raha hai |
| 0.8 | Model 80% variance explain kar raha hai (achha) |
| 0.0 | Model mean ke barabar predict kar raha hai (bekar) |
| Negative | Model mean se bhi bura fit (bohat bekar) |

### MSE vs R² Score

| MSE | R² Score |
|---|---|
| Actual error ka magnitude batata hai (squared units mein) | Model ki goodness of fit ko ratio/percentage mein batata hai |
| Lower is better | Higher (closer to 1) is better |
| Absolute number, cross-dataset comparison mushkil | Normalized (0-1), compare karna aasan |

**Real use:** Dono sath use karte hain — MSE se error ka real-term magnitude pata chalta hai, R² se overall fit quality.
