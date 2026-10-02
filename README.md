# Machine Learning From Scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hamedheli/ml-from-scratch/blob/main/notebooks/ml-from-scratch.ipynb)

Implementations of core machine learning algorithms written with NumPy instead of high-level libraries: k-nearest neighbours, decision trees, gradient descent and SGD, a neural network with hand-written backpropagation, and PCA / robust PCA.

**Start with the notebook: [`notebooks/ml-from-scratch.ipynb`](notebooks/ml-from-scratch.ipynb).** It runs each model on real data, shows the results, and explains what they mean.

## Highlights

| Topic | What's implemented | Result |
|---|---|---|
| k-Nearest Neighbours | Vectorized pairwise distances, k-NN classifier, 10-fold cross-validation written by hand | 6.5% test error on US cities, vs. 16.1% for a depth-5 decision tree |
| Decision Trees | Equality, error-rate and information-gain splitting rules; recursive tree | Training error matches scikit-learn's entropy tree at every depth |
| Stochastic Gradient Descent | Gradient descent, Armijo line search, mini-batch SGD, four learning-rate schedules | Shows why a constant step never converges and why $c/t^2$ stops too early |
| Neural Networks | Multi-layer perceptron with sigmoid layers, softmax loss, backpropagation, L2 regularization | ~95% validation accuracy on MNIST; visualizes how hidden layers make non-linear data separable |
| PCA and Robust PCA | PCA via SVD; L1 robust PCA via alternating gradient descent | 2-D map of 50 animals from 85 traits; separates moving cars from the background in traffic video |

## Repository layout

```
src/
  knn.py                    k-nearest neighbours
  decision_stump.py         decision stumps (equality, error rate, information gain)
  decision_tree.py          recursive decision tree
  fun_obj.py                loss functions with analytic gradients, incl. neural-net backprop
  optimizers.py             gradient descent, line search, stochastic gradient
  learning_rate_getters.py  learning-rate schedules
  linear_models.py          linear regression / classification models
  neural_net.py             neural network (encoder + linear output layer)
  encoders.py               PCA, gradient-based (robust) PCA, multi-layer encoder
  utils.py                  data loading, distances, plotting
data/                       datasets used in the notebook (pickled NumPy arrays)
notebooks/
  ml-from-scratch.ipynb     walkthrough of everything above
```

## Running it

Click the Colab badge above, or run it locally:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/ml-from-scratch.ipynb
```

The whole notebook runs in about two minutes on a CPU. scikit-learn is used only to one-hot encode labels and as a reference to check the decision tree.

## Acknowledgements

These projects come from coursework in UBC's CPSC 340 (Machine Learning and Data Mining). The course provided the datasets and parts of the supporting framework (for example plotting helpers and the line-search optimizer). I implemented the algorithms themselves.
