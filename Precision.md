# Precision (Machine Learning Metric)

Precision measures how many of the items your model **predicted as positive** are actually positive.

> "Out of everything I flagged as positive, how much did I get right?"

## Formula

```
Precision = TP / (TP + FP)
```

| Term | Meaning |
|------|---------|
| **TP** (True Positive) | Predicted positive, and it *was* positive |
| **FP** (False Positive) | Predicted positive, but it was *actually negative* |

Precision only looks at what you predicted positive — it ignores positives you missed (that's **recall**'s job).

## Example: Spam Email Classifier

Model checks 100 emails and flags 20 as "spam."

- 15 of those 20 are actually spam → **TP = 15**
- 5 of those 20 are legit emails, wrongly flagged → **FP = 5**

```
Precision = 15 / (15 + 5) = 0.75 → 75%
```

**Interpretation:** When the model says "spam," it's right 75% of the time.

## Why It Matters

Precision matters most when **false positives are costly**. Low precision here means real emails (job offers, important messages) keep landing in spam. Tuning for higher precision makes the model more "cautious" before calling something positive.

## Precision vs. Recall

| Metric | Question | Denominator |
|--------|----------|-------------|
| Precision | Of predicted positives, how many are correct? | TP + FP |
| Recall | Of actual positives, how many did we catch? | TP + FN |

There's usually a trade-off — this is the basis of the **precision-recall curve**.

## Code Example

```python
from sklearn.metrics import precision_score

y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

precision = precision_score(y_true, y_pred)
print(f"Precision: {precision:.2f}")
```
