# Confusion Matrix (Machine Learning)

A confusion matrix is a table that shows how well a classification model's predictions match the actual labels. It breaks results down into four categories for a binary classifier.

## The Table

|                     | Predicted Positive | Predicted Negative |
|---------------------|:-------------------:|:-------------------:|
| **Actual Positive** | TP (True Positive)  | FN (False Negative) |
| **Actual Negative** | FP (False Positive) | TN (True Negative)  |

## The Four Outcomes

| Term | Meaning |
|------|---------|
| **TP** (True Positive)  | Model predicted positive, and it *was* positive |
| **TN** (True Negative)  | Model predicted negative, and it *was* negative |
| **FP** (False Positive) | Model predicted positive, but it was actually negative (a "false alarm") |
| **FN** (False Negative) | Model predicted negative, but it was actually positive (a "miss") |

## Example: Spam Email Classifier

Model checks 100 emails:

|                     | Predicted Spam | Predicted Not Spam |
|---------------------|:---------------:|:--------------------:|
| **Actually Spam**     | TP = 15        | FN = 5              |
| **Actually Not Spam** | FP = 5         | TN = 75             |

Reading it:
- **15 emails** correctly caught as spam
- **5 emails** were spam but slipped through (missed)
- **5 emails** were legit but wrongly flagged as spam
- **75 emails** correctly left alone

## Metrics Derived From It

| Metric | Formula | Question it answers |
|--------|---------|----------------------|
| Accuracy | (TP + TN) / Total | Overall, how often is the model right? |
| Precision | TP / (TP + FP) | Of predicted positives, how many are correct? |
| Recall (Sensitivity) | TP / (TP + FN) | Of actual positives, how many did we catch? |
| Specificity | TN / (TN + FP) | Of actual negatives, how many did we correctly reject? |
| F1 Score | 2 × (Precision × Recall) / (Precision + Recall) | Balance between precision and recall |

Using the spam example above:
```
Accuracy    = (15 + 75) / 100 = 0.90 → 90%
Precision   = 15 / (15 + 5)   = 0.75 → 75%
Recall      = 15 / (15 + 5)   = 0.75 → 75%
Specificity = 75 / (75 + 5)   = 0.94 → 94%
```

## Why It Matters

Accuracy alone can be misleading, especially with imbalanced classes (e.g., 95% non-spam). The confusion matrix shows exactly *where* a model fails — missing real positives (FN) vs. raising false alarms (FP) — so you know which type of error to reduce.

## Code Example

```python
from sklearn.metrics import confusion_matrix

y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

cm = confusion_matrix(y_true, y_pred)
print(cm)
# [[TN FP]
#  [FN TP]]
```
