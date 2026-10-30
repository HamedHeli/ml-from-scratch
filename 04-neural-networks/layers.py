import numpy as np

"""
Fully-connected layers used by the neural network in neural_net.py.
Each layer is an "encoder" (an object with encode()), and NonLinearEncoderMultiLayer
chains them together. These used to live in encoders.py next to PCA; see ../05-pca/.
"""


class NonLinearEncoderMatMulActivation:
    """
    An "encoder" object encapsulating matrix multiplication and non-linear activation
    into its encode() term.
    This encoder in itself does not have fit() implemented, but it can be optimized
    as a part of multi-layer perceptron training via backpropagation.
    """

    def __init__(self, input_dim, output_dim, scale=1e-1, activate_yes=True):
        self.W = scale * np.random.randn(
            output_dim, input_dim
        )  # by convention we will right-multiply by W.T
        self.b = scale * np.random.randn(output_dim)
        self.activate_yes = activate_yes

    @property
    def size(self):
        return self.W.size + self.b.size

    def set_weights_and_biases(self, w):
        """
        Take a flattened parameter vector w and take it as weights and biases
        """
        assert w.size == self.W.size + self.b.size
        weights_flat = w[: -self.b.size]
        biases_flat = w[-self.b.size :]
        self.W = weights_flat.reshape(self.W.shape)
        self.b = biases_flat.reshape(self.b.shape)

    def get_parameters_flattened(self):
        return np.concatenate([self.W.reshape(-1), self.b.reshape(-1)])

    def encode(self, X):
        """
        Assume that X is an n-by-input_dim array,
        then we output a Z, an n-by-output_dim array.
        """
        Z = (X @ self.W.T) + self.b
        if self.activate_yes:
            return 1.0 / (1 + np.exp(-Z))
            # return Z * (Z >= 0)
        else:
            return Z

    def fit(self, X):
        raise RuntimeError("fit() should not be called.")


class NonLinearEncoderMultiLayer:
    """
    Encoder architecture used by fully-connected neural networks.
    Under the hood, we instantiate multiple NonLinearEncoderMatMulActivation
    instances and chain them together.
    layer_sizes is a list of integers specifying the input and output dimensions.
    Important: the first number should be 'd', the number of features.

    fit() is not implemented as the encoder parameters should be
    populated by a neural network's optimization via backpropagation.
    """

    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights_and_biases = None

        # Instantiate "child encoders" that will be chained together
        self.encoders = []
        for input_dim, output_dim in zip(
            layer_sizes, layer_sizes[1:]
        ):  # zip() trick to iterate consecutive pairs
            encoder = NonLinearEncoderMatMulActivation(input_dim, output_dim)
            self.encoders.append(encoder)

    @property
    def size(self):
        return self.get_parameters_flattened().size

    def set_weights_and_biases(self, w):
        """
        Receive parameters in a vector form. Parse into weights and biases.
        """
        encoder_sizes = [encoder.size for encoder in self.encoders]
        assert w.size == np.sum(encoder_sizes)
        ws = np.array_split(
            w, np.cumsum(encoder_sizes)
        )  # partition w into sub-arrays, corresponding to each encoder's size
        for w_for_encoder, encoder in zip(ws, self.encoders):
            encoder.set_weights_and_biases(w_for_encoder)

    def get_parameters_flattened(self):
        return np.concatenate(
            [encoder.get_parameters_flattened() for encoder in self.encoders]
        )

    def encode(self, X):
        """
        Multi-layer encoder's activations are used for gradient computation,
        so we will collect and return them.
        """
        activations = [X]
        for encoder in self.encoders:
            Z = encoder.encode(X)
            activations.append(Z)
            X = Z
        return Z, activations[:-1]

    def fit(self, X):
        raise RuntimeError("fit() should not be called.")
