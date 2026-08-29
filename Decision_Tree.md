# Decision Tree (Machine Learning)

A Decision Tree is a supervised learning algorithm that makes predictions by asking a series of **yes/no (or if-else) questions** about the data, splitting it step by step until it reaches a decision.

It looks like an actual upside-down tree: starting from a **root node**, branching into **internal nodes** (questions), and ending at **leaf nodes** (final predictions).

## Structure

```
                [Root Node: Question]
                 /              \
              Yes                No
              /                    \
      [Internal Node]          [Leaf: Prediction]
       /         \
     Yes          No
     /              \
[Leaf: A]       [Leaf: B]
```

| Term | Meaning |
|------|---------|
| **Root Node** | The first question, splits the entire dataset |
| **Internal Node** | A further question based on a feature |
| **Branch** | The outcome of a question (Yes/No, or a range of values) |
| **Leaf Node** | Final output — a class label (classification) or number (regression) |

## Example: Should You Play Tennis?

Suppose you have this small dataset:

| Outlook | Temperature | Windy | Play Tennis? |
|---------|-------------|-------|--------------|
| Sunny | Hot | No | No |
| Sunny | Hot | Yes | No |
| Overcast | Hot | No | Yes |
| Rainy | Mild | No | Yes |
| Rainy | Cool | Yes | No |

A decision tree trained on this might look like:

```
                [Outlook?]
              /     |      \
         Sunny  Overcast   Rainy
           |        |         |
      [Windy?]   [Yes]    [Windy?]
       /    \               /    \
     Yes    No            Yes    No
      |      |              |      |
    [No]   [No]           [No]   [Yes]
```

**How to read it:** If Outlook = Overcast → always Play Tennis. If Outlook = Sunny → don't play, regardless of wind. If Outlook = Rainy → only play if it's not windy.

To predict a new day like `(Outlook=Rainy, Windy=No)`, you just follow the branches: Outlook → Rainy → Windy? → No → **Play Tennis = Yes**.

## How Splits Are Chosen

At each node, the tree picks the feature/question that best separates the classes. Common metrics:

| Metric | Used For | Idea |
|--------|----------|------|
| **Gini Impurity** | Classification | How "mixed" the classes are after a split (0 = pure) |
| **Entropy / Information Gain** | Classification | How much "uncertainty" a split removes |
| **Variance Reduction / MSE** | Regression | How much a split reduces prediction error |

The algorithm greedily picks whichever question reduces impurity/error the most at each step.

## Why Decision Trees Are Popular

- **Easy to interpret** — you can literally trace the "reasoning" (unlike a black-box neural net)
- Handles both **classification** and **regression**
- Works with numeric and categorical data without much preprocessing
- No need to scale/normalize features

## Main Weakness: Overfitting

A tree grown too deep can memorize the training data (learn noise instead of patterns), performing poorly on new data.

**Fixes:**
- **Pruning** — cutting back branches that don't add real predictive value
- **Max depth / min samples per leaf** — limiting how deep or specific the tree can get
- **Ensembles** — combining many trees, e.g. **Random Forest** (many trees voting) or **Gradient Boosting** (trees built to fix previous trees' errors)

## Code Example

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Features: [Outlook(0=Sunny,1=Overcast,2=Rainy), Windy(0=No,1=Yes)]
X = [[0,0], [0,1], [1,0], [2,0], [2,1]]
y = ["No", "No", "Yes", "Yes", "No"]

clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf.fit(X, y)

# Predict a new case: Rainy, Not windy
print(clf.predict([[2, 0]]))  # → ['Yes']

plot_tree(clf, feature_names=["Outlook", "Windy"], class_names=clf.classes_, filled=True)
plt.show()
```
