"""
Checks the tensor shapes from structure-of-a-neural-network.md with plain NumPy.

1. A hidden layer on houses: (20 houses, 10 features) @ (10, 32) -> (20, 32).
2. The same layer applied at every point of one item: (N, 3) @ (3, 32) -> (N, 32).
3. A 3x3 convolution with 64 filters is that house layer applied to every pixel's
   3x3x3 = 27-number neighbourhood ("mini-house"): (784, 27) @ (27, 64) -> (28, 28, 64).
   We compare it against a direct loop over pixels to show the two are the same thing.
4. Stride 2 with and without padding: 14x14 vs 13x13.

Run:  python shapes_demo.py
"""

import numpy as np

rng = np.random.default_rng(0)


def relu(a):
    return np.maximum(a, 0)


def conv_output_size(n, kernel, stride=1, pad=0):
    return (n + 2 * pad - kernel) // stride + 1


def patches(image, kernel=3, stride=1, pad=0):
    """Cut an (H, W, C) image into one row of kernel*kernel*C numbers per output pixel."""
    image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)))
    H, W, C = image.shape
    out_h = (H - kernel) // stride + 1
    out_w = (W - kernel) // stride + 1
    rows = []
    for i in range(0, out_h * stride, stride):
        for j in range(0, out_w * stride, stride):
            rows.append(image[i : i + kernel, j : j + kernel, :].reshape(-1))
    return np.array(rows), (out_h, out_w)


# 1. Houses: each of the 32 columns of W is one question that mixes all 10 features
X_houses = rng.normal(size=(20, 10))
W, b = rng.normal(size=(10, 32)), rng.normal(size=32)
Z = relu(X_houses @ W + b)
print("houses     ", X_houses.shape, "->", Z.shape)

# 2. One plate of N points with features (x, y, d): same 32 questions at every point
N = 500
X_points = rng.normal(size=(N, 3))
W_point = rng.normal(size=(3, 32))
print("points     ", X_points.shape, "->", (X_points @ W_point).shape)

# 3. A 28x28 RGB image and a conv layer with 64 filters of size 3x3x3
image = rng.normal(size=(28, 28, 3))
filters = rng.normal(size=(3, 3, 3, 64))  # (kernel row, kernel col, colour, question)
P, (out_h, out_w) = patches(image, pad=1)  # "same" padding keeps 28x28
W_conv = filters.reshape(27, 64)  # 64 questions, each mixing 27 numbers
out = (P @ W_conv).reshape(out_h, out_w, 64)
print("mini-houses", P.shape, "@", W_conv.shape, "->", out.shape)

# the same thing as an explicit loop: stand at each pixel and ask the 64 questions
padded = np.pad(image, ((1, 1), (1, 1), (0, 0)))
loop = np.zeros((28, 28, 64))
for i in range(28):
    for j in range(28):
        loop[i, j] = np.einsum("abc,abck->k", padded[i : i + 3, j : j + 3], filters)
assert np.allclose(out, loop)
print("matrix version == pixel-by-pixel loop")

# a batch of 20 pictures just adds a leading axis
print("batch of 20 ->", (20, *out.shape))

# the next 3x3 layer sees 3*3*64 = 576 numbers per pixel
P2, _ = patches(out, pad=1)
print("next layer's mini-house has", P2.shape[1], "numbers")

# 4. Stride 2
for pad in [1, 0]:
    P_s, shape = patches(image, stride=2, pad=pad)
    assert shape[0] == conv_output_size(28, 3, stride=2, pad=pad)
    print(f"stride 2, pad {pad}:", (*shape, 64))
