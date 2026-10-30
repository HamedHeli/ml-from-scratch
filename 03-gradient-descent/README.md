# 03 · Gradient Descent & SGD

[Notebook: sgd.ipynb](sgd.ipynb) · code: [`optimizers.py`](optimizers.py), [`learning_rate_getters.py`](learning_rate_getters.py), [`linear_models.py`](linear_models.py), losses in [`../common/fun_obj.py`](../common/fun_obj.py)

## Key ideas

* **Gradient descent:** $w^{t+1} = w^t - \alpha^t \nabla f(w^t)$. `GradientDescentLineSearch` picks $\alpha$ with an Armijo backtracking line search.
* **SGD** uses the gradient on a random mini-batch: much cheaper per step and correct on average, but noisy.
* **Because of the noise, the step size has to shrink.** SGD converges when $\sum_t \alpha^t = \infty$ (steps can travel far enough) and $\sum_t (\alpha^t)^2 < \infty$ (the noise averages out).
* **Losses are objects** that return value and gradient, so any model can use any optimizer. `LinearModel` and `LinearModelMultiOutput` are the simplest users.

## Results

Least squares on the `dynamics` data, mini-batches of 10, $c = 0.1$:

| Schedule | Behaviour |
|---|---|
| $c$ | jumps between ~700 and ~2,700, never settles |
| $c/t$ | reaches ~699 within one epoch |
| $c/t^2$ | freezes at ~726: $\sum_t c/t^2$ is finite, so it stops early |
| $c/\sqrt{t}$ | ~700 with a little noise left |

## Connections

* **[04 Neural Networks](../04-neural-networks/):** the MNIST network trains with `StochasticGradient`, and the sinusoid experiments with `GradientDescentLineSearch`. `LinearModelMultiOutput` is the network's output layer.
* **[05 PCA](../05-pca/):** robust PCA has no closed form, so it is fit by alternating gradient descent on $W$ and $Z$ in `LinearEncoderGradient`.
