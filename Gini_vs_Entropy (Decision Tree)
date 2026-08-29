# Gini Impurity vs. Entropy (Decision Trees)

Both **Gini Impurity** and **Entropy** are ways to measure how "mixed" or "impure" a group of data is at a node in a decision tree. The tree uses them to decide **which question/split is best** at each step.

A node is "pure" if all its samples belong to one class. A node is "impure" if it has a mix of classes.

## Gini Impurity

Measures the probability that a randomly picked sample would be **misclassified** if you randomly labeled it based on the class distribution at that node.

```
Gini = 1 - Σ(pᵢ)²
```

Where `pᵢ` = proportion of samples belonging to class `i`.

- **Gini = 0** → node is pure (only one class)
- **Gini = 0.5** → maximum impurity for 2 classes (50/50 split)

### Example

Node has 10 samples: 6 "Yes", 4 "No"

```
p(Yes) = 6/10 = 0.6
p(No)  = 4/10 = 0.4

Gini = 1 - (0.6² + 0.4²)
     = 1 - (0.36 + 0.16)
     = 1 - 0.52
     = 0.48
```

## Entropy (Information Gain)

Borrowed from information theory — measures the amount of **"disorder" or uncertainty** in the node.

```
Entropy = -Σ pᵢ × log₂(pᵢ)
```

- **Entropy = 0** → node is pure
- **Entropy = 1** → maximum impurity for 2 classes (50/50 split)

### Example

Same node: 6 "Yes", 4 "No"

```
Entropy = -(0.6 × log₂0.6) - (0.4 × log₂0.4)
        = -(0.6 × -0.737) - (0.4 × -1.322)
        = 0.442 + 0.529
        = 0.971
```

**Information Gain** = Entropy(parent) − weighted average Entropy(children). The tree picks the split that gives the **highest Information Gain** (biggest drop in entropy).

## Side-by-Side Comparison

| | Gini Impurity | Entropy |
|---|---|---|
| Formula | 1 - Σ(pᵢ)² | -Σ pᵢ log₂(pᵢ) |
| Range (2 classes) | 0 to 0.5 | 0 to 1 |
| Computation | Faster (no log) | Slower (log calculation) |
| Used by | CART (scikit-learn default) | ID3, C4.5 |
| Behavior | Tends to isolate the most frequent class | Tends to produce slightly more balanced splits |

In practice, **they usually pick very similar splits** — the difference in resulting trees is small. Gini is preferred when speed matters (large datasets); Entropy is sometimes preferred for its information-theory grounding.

## How This Fits Into Splitting

At each node, for every possible split, the tree calculates the **weighted impurity of the resulting children** and picks the split that reduces impurity the most.

```
Weighted Impurity = (n_left/n_total) × Impurity(left) + (n_right/n_total) × Impurity(right)
```

The split with the **lowest weighted impurity** (or highest Information Gain, in Entropy's case) wins and becomes that node's question.

## Code Example

```python
from sklearn.tree import DecisionTreeClassifier

X = [[0,0], [0,1], [1,0], [2,0], [2,1]]
y = ["No", "No", "Yes", "Yes", "No"]

# Using Gini (default)
tree_gini = DecisionTreeClassifier(criterion="gini")
tree_gini.fit(X, y)

# Using Entropy
tree_entropy = DecisionTreeClassifier(criterion="entropy")
tree_entropy.fit(X, y)
```
