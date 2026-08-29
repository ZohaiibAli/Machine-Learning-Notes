# Precision vs. Recall

Both are metrics for evaluating a classifier's positive predictions, but they answer different questions and are affected by different types of mistakes.

## Quick Definitions

| Metric | Question it answers | Formula |
|--------|----------------------|---------|
| **Precision** | Of everything predicted positive, how many are correct? | TP / (TP + FP) |
| **Recall** | Of everything actually positive, how many did we catch? | TP / (TP + FN) |

- **Precision** is hurt by **False Positives** (false alarms).
- **Recall** is hurt by **False Negatives** (missed cases).

## Example: Spam Email Classifier

Model checks 100 emails. Of the **20 actual spam emails**, it flags 15 correctly and misses 5. It also wrongly flags 5 legit emails as spam.

|                     | Predicted Spam | Predicted Not Spam |
|---------------------|:---------------:|:--------------------:|
| **Actually Spam**     | TP = 15        | FN = 5              |
| **Actually Not Spam** | FP = 5         | TN = 75             |

```
Precision = 15 / (15 + 5) = 0.75 → 75%
Recall    = 15 / (15 + 5) = 0.75 → 75%
```

(They happen to match here, but usually they don't — see below.)

## The Trade-off

Making a model more "aggressive" about predicting positive:
- **↑ Recall** — catches more real positives
- **↓ Precision** — but also raises more false alarms

Making a model more "cautious":
- **↑ Precision** — fewer false alarms
- **↓ Recall** — but misses more real positives

This is why tuning a classifier's decision threshold shifts one metric up and the other down — the basis of the **precision-recall curve**.

## When to Prioritize Which

| Prioritize **Precision** when... | Prioritize **Recall** when... |
|-----------------------------------|-------------------------------|
| False positives are costly | False negatives are costly |
| Example: spam filter (don't want to lose real emails) | Example: cancer screening (don't want to miss a real case) |
| Example: recommending content (bad recs annoy users) | Example: fraud detection (missing fraud is expensive) |

## Balancing Both: F1 Score

When you need a single number that balances the two:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

```
F1 = 2 × (0.75 × 0.75) / (0.75 + 0.75) = 0.75
```

F1 is high only when **both** precision and recall are reasonably high — it penalizes models that sacrifice one for the other.

## Code Example

```python
from sklearn.metrics import precision_score, recall_score, f1_score

y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))
```
