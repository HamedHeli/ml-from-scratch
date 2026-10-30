# 05 · PCA & Robust PCA

[Notebook: pca.ipynb](pca.ipynb) · code: [`encoders.py`](encoders.py), losses in [`../common/fun_obj.py`](../common/fun_obj.py)

## Key ideas

* **PCA** writes centered data as $X \approx ZW$: the rows of $W$ ($k \times d$) are the principal components, and $Z$ ($n \times k$) holds each example's coordinates. It minimizes $\|ZW - X\|_F^2$ and is solved with the SVD (`PCAEncoder`).
* **Robust PCA** swaps the squared error for the absolute error $\sum_{ij} |(ZW - X)_{ij}|$, so outliers pull much less. It is fit by alternating gradient descent (`LinearEncoderGradient`), with $|r| \approx \sqrt{r^2 + \epsilon}$ to keep it differentiable.
* **Encoders.** Both are "encoders" with `encode` and `decode`: they turn each example into $k$ numbers.

## Results

* Animals with Attributes (50 animals × 85 traits): PC1 separates land mammals from sea mammals and PC2 grazers from hunters. Two components keep 30% of the variance; five pass 50%.
* Highway camera: with $k = 5$, regular PCA leaves "ghost" cars in the background and a noisy foreground mask. Robust PCA reconstructs a nearly empty road and picks out the cars cleanly.

## Connections

* **[03 Gradient Descent](../03-gradient-descent/):** provides the optimizers for robust PCA.
* **[04 Neural Networks](../04-neural-networks/):** PCA is a *linear* encoder: each of the $k$ components is one question (a weighted sum of all features) with no activation. A neural network's hidden layer is the same thing followed by a non-linearity, and stacking those layers is what lets it bend the data.
