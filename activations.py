import numpy as np
from abc import ABC, abstractmethod


class Activation(ABC):
    """Abstract class defining the common interface for all activation methods."""

    def __call__(self, Z):
        return self.forward(Z)

    @abstractmethod
    def forward(self, Z):
        pass


def initialize_activation(name: str) -> Activation:
    """Factory method to return an Activation object of the specified type."""
    if name == "linear":
        return Linear()
    elif name == "sigmoid":
        return Sigmoid()
    elif name == "tanh":
        return TanH()
    elif name == "relu":
        return ReLU()
    elif name == "softmax":
        return SoftMax()
    else:
        raise NotImplementedError("{} activation is not implemented".format(name))


class Linear(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for f(z) = z.

        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        return Z

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for f(z) = z.

        Parameters
        ----------
        Z   input to `forward` method
        dY  derivative of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        derivative of loss w.r.t. input of this layer
        """
        return dY


class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for relu activation:
        f(z) = z if z >= 0
               0 otherwise

        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        activated = np.array([[z if z >= 0 else 0 for z in Z[i]] for i in range(Z.shape[0])])
        return activated

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for relu activation.

        Parameters
        ----------
        Z   input to `forward` method
        dY  derivative of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        derivative of loss w.r.t. input of this layer
        """
        dYdZ = np.array([[1 if z >= 0 else 0 for z in Z[i]] for i in range(Z.shape[0])])
        dLdZ = np.array([np.multiply(dYdZ[i], dY[i]) for i in range(Z.shape[0])])
        return dLdZ


class SoftMax(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for softmax activation.
        Hint: The naive implementation might not be numerically stable.

        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        def softmax(x):
            m = np.max(x)
            n = np.exp(x - m)
            d = np.sum(n)
            return n / d
        
        return np.array([softmax(Z[i]) for i in range(Z.shape[0])])

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for softmax activation.

        Parameters
        ----------
        Z   input to `forward` method (m, out)
        dY  derivative of loss w.r.t. the output of this layer
            same shape as `Z` (m, out)

        Returns
        -------
        derivative of loss w.r.t. input of the forward method (m, out)
        """
        softmax_mat = self.forward(Z)
        output_dim = Z.shape[1]
        
        def jacobian(i):
            softmax = softmax_mat[i]
            I = np.identity(output_dim)
            mat = np.multiply(np.subtract(I, softmax), softmax.reshape(-1,1))
            return mat
        dLdZ = np.array([np.dot(dY[i], jacobian(i)) for i in range(Z.shape[0])]) # (m, out)
        return dLdZ # (m, out)


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for sigmoid function:
        f(z) = 1 / (1 + exp(-z))

        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        def sigmoid(z):
            return np.reciprocal(1.0+np.exp(-z))
        return np.array([sigmoid(Z[i]) for i in range(Z.shape[0])])

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for sigmoid.

        Parameters
        ----------
        Z   input to `forward` method
        dY  derivative of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        derivative of loss w.r.t. input of this layer
        """
        def sigmoid(z):
            return np.reciprocal(1.0+np.exp(-z))
        dLdZ = np.zeros(Z.shape)
        for i in range(Z.shape[0]):
            s = sigmoid(Z[i])
            dLdZ[i] = np.multiply(dY[i], np.multiply(s, 1 - s))
        return dLdZ


class TanH(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for f(z) = tanh(z).

        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        return np.tanh(Z)

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for f(z) = tanh(z).

        Parameters
        ----------
        Z   input to `forward` method
        dY  derivative of loss w.r.t. the output of this layer

        Returns
        -------
        derivative of loss w.r.t. input of this layer
        Zt = XtW + St-1*U + b

        dLdZt = dLdYt*dYtdZt
        dYtdZt = tanh'(Zt) =  1 - tanh**2(Zt)

        1 - tanh**2
        """
        tanh = np.tanh(Z)

        return np.multiply(dY, np.subtract(1, np.power(tanh, 2)))
