# 02 · Decision Trees

[Notebook: decision_trees.ipynb](decision_trees.ipynb) · code: [`decision_stump.py`](decision_stump.py), [`decision_tree.py`](decision_tree.py)

## Key ideas

* **A stump is one question about one feature:** "is $x_j > t$?". A tree asks a stump, splits the data, and recursively asks new questions on each half.
* **Greedy search.** Each node picks its best split and never looks ahead.
* **The splitting rule matters.** Three stumps are implemented: equality on rounded values (a weak baseline), fewest training mistakes (error rate), and information gain
$$\mathrm{IG} = H(y) - \tfrac{n_{\text{yes}}}{n} H(y_{\text{yes}}) - \tfrac{n_{\text{no}}}{n} H(y_{\text{no}}), \qquad H(p) = -\textstyle\sum_c p_c \log p_c .$$

## Results

* The information-gain tree matches scikit-learn's `DecisionTreeClassifier(criterion="entropy")` at every depth.
* The error-rate tree gets stuck at about 11% training error from depth 5, because it won't make a split that doesn't immediately reduce mistakes. Information gain keeps finding purer splits and reaches 0% by depth 9, which is memorization.

## Connections

* **[01 k-NN](../01-k-nearest-neighbours/):** same data; k-NN wins on test error because the class borders are diagonal and irregular.
* **[04 Neural Networks](../04-neural-networks/):** a stump question uses one feature at a time. A neuron's question mixes *all* features with learned weights, so its boundary can be any tilted line, not just an axis-aligned one (see [structure of a neural network](../04-neural-networks/structure-of-a-neural-network.md)).
