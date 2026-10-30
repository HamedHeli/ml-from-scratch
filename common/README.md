# common

Helpers that every model folder uses.

* [`utils.py`](utils.py): `load_dataset` and `load_trainval` (they read from the repository's [`data/`](../data) folder wherever you run from), `euclidean_dist_squared`, `mode`, `standardize_cols`, and the plotting helpers `plot_classifier` and `evalModel`.
* [`fun_obj.py`](fun_obj.py): loss functions as objects whose `evaluate(w, X, y)` returns both the value and the gradient. Least squares, logistic, softmax, robust regression, the PCA / robust PCA factor losses, and the neural-network losses with hand-written backpropagation (`MLPLoss` and its subclasses). `FunObj.check_correctness` compares a gradient with finite differences.

The losses are plugged into the optimizers in [`03-gradient-descent/`](../03-gradient-descent), which only ever see a flat parameter vector, a value and a gradient.
