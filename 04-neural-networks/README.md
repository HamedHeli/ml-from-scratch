# 04 · Neural Networks

[Notebook: neural_networks.ipynb](neural_networks.ipynb) · code: [`neural_net.py`](neural_net.py), [`layers.py`](layers.py), backprop in [`../common/fun_obj.py`](../common/fun_obj.py)

**New: [Structure of a neural network](structure-of-a-neural-network.md)**, notes on the intuition: a neuron is one question that mixes all features, width is how many questions you ask, depth is follow-up questions, and channels and CNN filters are the same idea repeated at every pixel.

## Key ideas

* **Layers:** $z^{(l)} = h(W^{(l)} z^{(l-1)} + b^{(l)})$ with the sigmoid $h(a) = 1/(1+e^{-a})$, then a linear output layer and a softmax.
* **Loss:** softmax cross-entropy plus $\frac{\lambda}{2}\|w\|^2$.
* **No autodiff.** Backpropagation through every layer is written by hand, using $h'(a) = h(a)(1-h(a))$ (`MLPLogisticRegressionLossL2`).
* **Pieces:** `NonLinearEncoderMultiLayer` ([`layers.py`](layers.py)) is the stack of sigmoid layers, `LinearModelMultiOutput` ([`../03-gradient-descent/linear_models.py`](../03-gradient-descent/linear_models.py)) the output layer, and `NeuralNet` ([`neural_net.py`](neural_net.py)) ties the layers, loss and optimizer together.

## Results

* **MNIST**, 784 → 128 → 64 → 10 → 10, mini-batch SGD: about 95% validation accuracy, with a small train/validation gap.
* **Sinusoids**, a 2-D dataset no straight line can separate: 2 → **4** → 2 bends the data so the linear output layer can split it (0.2% validation error), while 2 → 2 → 2 and 2 → 2 → 2 → 2 fail (18–20%). Width matters more than depth here.

## Files

| File | What it is |
|---|---|
| [`neural_networks.ipynb`](neural_networks.ipynb) | forward pass by hand, MNIST, what the hidden layers learn |
| [`structure-of-a-neural-network.md`](structure-of-a-neural-network.md) | neurons as questions, width and depth, channels, CNNs, batches |
| [`shapes_demo.py`](shapes_demo.py) | checks the shapes in those notes with NumPy, incl. a CNN layer as a matrix product |
| [`figures/`](figures) | figures for the notes and the script that draws them |
| [`neural_net.py`](neural_net.py), [`layers.py`](layers.py) | the network |

## Connections

* **[03 Gradient Descent](../03-gradient-descent/):** the optimizers and the output layer.
* **[05 PCA](../05-pca/):** PCA is a linear encoder; a hidden layer is a linear encoder plus a non-linearity.
* **[02 Decision Trees](../02-decision-trees/):** a stump asks about one feature; a neuron asks about all of them at once.
* **Physics + AI:** the same structure (per-point MLPs plus a mixing step) underlies GNS, FNO and Transolver. See the last section of the notes.
