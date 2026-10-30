# 01 · k-Nearest Neighbours

[Notebook: knn.ipynb](knn.ipynb) · code: [`knn.py`](knn.py), distances in [`../common/utils.py`](../common/utils.py)

## Key ideas

* **No training step.** `fit` stores the data. `predict` finds the $k$ closest training points and returns their most common label.
* **$k$ trades bias for variance.** Small $k$ follows the noise in the training set; large $k$ averages over more neighbours and gives a smoother boundary.
* **Vectorized distances.** All test-to-train distances come from one matrix operation, $\|a-b\|^2 = \|a\|^2 + \|b\|^2 - 2a^\top b$, instead of a double loop.
* **Cross-validation picks $k$** without touching the test set. It is written by hand here (10 folds).

## Results

* US cities (longitude, latitude → political alignment): $k=1$ is best, with 6.5% test error. The classes form compact regions, so a very local model works.
* Household debt (91 noisy features): $k=1$ is the worst choice. Cross-validation picks $k=17$ (73.0% test accuracy), within a point of the best $k$, against a 67.8% majority-class baseline.

## Connections

* **[02 Decision Trees](../02-decision-trees/):** compared on the same cities data. A tree can only cut the map into axis-aligned rectangles, so it gets more than twice the test error (16.1% at depth 5).
* **[04 Neural Networks](../04-neural-networks/):** k-NN uses raw distances between inputs. A network instead learns features (answers to learned questions) and measures things in that new space.
