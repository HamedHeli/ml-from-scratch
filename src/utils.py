from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as plt
import statistics


def evalModel(model, X, y):
    # Create an instance of the model if a class is passed
    if hasattr(model, '__name__'):  
        model_instance = model()
        model_name = model.__name__
    else:  
        model_instance = model
        model_name = model.__class__.__name__
    print("Evaluating model:", model_name)

    model_instance.fit(X, y)
    y_pred = model_instance.predict(X)

    error = np.mean(y_pred != y)
    print("Error: %.3f" % error)

    # PLOT RESULT
    plot_classifier(model_instance, X, y)

    plt.show()


def load_dataset(filename):
    with open(Path("..", "data", filename).with_suffix(".pkl"), "rb") as f:
        return pickle.load(f)


def plot_classifier(model, X, y, need_argmax=False, ax=None):
    """plots the decision boundary of the model and the scatterpoints
       of the target values 'y'.

    Assumptions
    -----------
    X has two features, and y contains two classes (e.g. 0 and 1)

    Parameters
    ----------
    model : the trained model which has the predict function

    X : the N by D feature array

    y : the N element vector corresponding to the target values

    """
    x1 = X[:, 0]
    x2 = X[:, 1]

    x1_min, x1_max = int(x1.min()) - 1, int(x1.max()) + 1
    x2_min, x2_max = int(x2.min()) - 1, int(x2.max()) + 1

    x1_line = np.linspace(x1_min, x1_max, 200)
    x2_line = np.linspace(x2_min, x2_max, 200)

    x1_mesh, x2_mesh = np.meshgrid(x1_line, x2_line)

    mesh_data = np.c_[x1_mesh.ravel(), x2_mesh.ravel()]

    y_pred = model.predict(mesh_data)
    if need_argmax:
        y_pred = np.argmax(y_pred, axis=1)

    y_pred = np.reshape(y_pred, x1_mesh.shape)

    if ax is None:
        ax = plt.gca()
    ax.set_xlim([x1_mesh.min(), x1_mesh.max()])
    ax.set_ylim([x2_mesh.min(), x2_mesh.max()])

    ax.contourf(
        x1_mesh,
        x2_mesh,
        -y_pred.astype(int),  # unsigned int causes problems with negative sign... o_O
        cmap=plt.cm.RdBu,
        alpha=0.6,
    )

    y_vals = np.unique(y)
    for c, color in zip(y_vals, "br"):
        in_c = y == c
        ax.scatter(x1[in_c], x2[in_c], color=color, label=f"class {c:+d}")
    ax.legend()


def mode(y):
    """Computes the element with the maximum count

    Parameters
    ----------
    y : an input numpy array

    Returns
    -------
    y_mode :
        Returns the element with the maximum count
    """
    if len(y) == 0:
        return -1
    else:
        return statistics.mode(y.flatten())


def load_trainval(filename):
    d = load_dataset(filename)
    return d["X_train"], d["y_train"], d["X_valid"], d["y_valid"]


def ensure_1d(x):
    if x.ndim == 1:
        return x
    elif x.ndim == 2:
        return x.squeeze(axis=1)
    elif x.ndim == 0:
        return x[np.newaxis]
    else:
        raise ValueError(f"invalid shape {x.shape} for ensure_1d")


def euclidean_dist_squared(X, Xtest):
    """Computes the Euclidean distance between rows of 'X' and rows of 'Xtest'

    Parameters
    ----------
    X : an N by D numpy array
    Xtest: an T by D numpy array

    Returns an N by T array where D[i, j] is the squared distance between
    X[i] and Xtest[j]. Uses the identity ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    so that everything is done with vectorized matrix operations, no loops.
    """

    # add extra dimensions, so function still works for X and/or Xtest are 1-D arrays.
    if X.ndim == 1:
        X = X[np.newaxis, :]
    if Xtest.ndim == 1:
        Xtest = Xtest[np.newaxis, :]

    return (
        np.sum(X ** 2, axis=1)[:, np.newaxis]
        + np.sum(Xtest ** 2, axis=1)[np.newaxis, :]
        - 2 * X @ Xtest.T
    )


def standardize_cols(X, mu=None, sigma=None):
    # Standardize each column with mean 0 and variance 1
    n_rows, n_cols = X.shape

    if mu is None:
        mu = np.mean(X, axis=0)

    if sigma is None:
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-8] = 1.0

    return (X - mu) / sigma, mu, sigma
