import numpy as np
from abc import ABC, abstractmethod

e = 1/np.power(10, 10)

class Loss(ABC):
    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass


def initialize_loss(name: str) -> Loss:
    if name == "cross_entropy":
        return CrossEntropy(name)
    elif name == "l2":
        return L2(name)
    elif name == "binary_cross_entropy":
        return BinaryCrossEntropy(name)
    else:
        raise NotImplementedError("{} loss is not implemented".format(name))


class CrossEntropy(Loss):
    """Cross entropy loss function."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, Y: np.ndarray, Y_hat: np.ndarray) -> float:
        return self.forward(Y, Y_hat)

    def forward(self, Y: np.ndarray, Y_hat: np.ndarray) -> float:
        """Computes the loss for predictions `Y_hat` given one-hot encoded labels
        `Y`.

        Parameters
        ----------
        Y      one-hot encoded labels of shape (batch_size, num_classes)
        Y_hat  model predictions in range (0, 1) of shape (batch_size, num_classes)

        Returns
        -------
        a single float representing the loss
        """
        ### YOUR CODE HERE ###
        Y_hat = np.where(Y_hat == 0, e, Y_hat)
        ln_Y_hat = np.log(Y_hat)
        cost = (-1/Y.shape[0])*np.sum(np.multiply(Y, ln_Y_hat))
        return cost

    def backward(self, Y: np.ndarray, Y_hat: np.ndarray) -> np.ndarray:
        """Backward pass of cross-entropy loss.

        Parameters
        ----------
        Y      one-hot encoded labels of shape (batch_size, num_classes)
        Y_hat  model predictions in range (0, 1) of shape (batch_size, num_classes)

        Returns
        -------
        the derivative of the cross-entropy loss with respect to the vector of
        predictions, `Y_hat`
        """
        Y_hat = np.where(Y_hat == 0, e, Y_hat)
        return -(1/Y.shape[0])*np.divide(Y, Y_hat)

class BinaryCrossEntropy(Loss):
    """Binary Cross entropy loss function."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, Y: np.ndarray, Y_hat: np.ndarray) -> float:
        return self.forward(Y, Y_hat)

    def forward(self, Y: np.ndarray, Y_hat: np.ndarray) -> float:
        """Computes the loss for predictions `Y_hat` given one-hot encoded labels
        `Y`.

            Parameters
            ----------
            Y      one-hot encoded labels of shape (batch_size, num_classes)
            Y_hat  model predictions in range (0, 1) of shape (batch_size, num_classes)

            Returns
            -------
            a single float representing the loss
        """
        Y_hat = np.where(Y_hat == 0, e, Y_hat)
        ln_Y_hat = np.log(Y_hat)
        cost = -np.sum([np.dot(Y[i], ln_Y_hat[i]) + np.dot((1 - Y[i]), np.log(1 - Y_hat[i])) for i in range(Y.shape[0])])
        return cost

    def backward(self, Y: np.ndarray, Y_hat: np.ndarray) -> np.ndarray:
        """Backward pass of cross-entropy loss.

        Parameters
        ----------
        Y      one-hot encoded labels of shape (batch_size, num_classes)
        Y_hat  model predictions in range (0, 1) of shape (batch_size, num_classes)

        Returns
        -------
        the derivative of the cross-entropy loss with respect to the vector of
        predictions, `Y_hat`
        """
        Y_hat = np.where(Y_hat == 0, e, Y_hat)
        return -(1/Y.shape[0])*np.divide(Y, Y_hat)


class L2(Loss):
    """Mean squared error loss."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, Y: np.ndarray, Y_hat: np.ndarray) -> float:
        return self.forward(Y, Y_hat)

    def forward(self, Y: np.ndarray, Y_hat: np.ndarray) -> float:
        """Compute the mean squared error loss for predictions `Y_hat` given
        regression targets `Y`.

        Parameters
        ----------
        Y      vector of regression targets of shape (batch_size, 1)
        Y_hat  vector of predictions of shape (batch_size, 1)

        Returns
        -------
        a single float representing the loss
        """
        
        return np.mean(np.power(np.subtract(Y, Y_hat), 2))

    def backward(self, Y: np.ndarray, Y_hat: np.ndarray) -> np.ndarray:
        """Backward pass for mean squared error loss.

        Parameters
        ----------
        Y      vector of regression targets of shape (batch_size, 1)
        Y_hat  vector of predictions of shape (batch_size, 1)

        Returns
        -------
        the derivative of the mean squared error with respect to the last layer
        of the neural network
        """
        return -2*np.mean(np.subtract(Y, Y_hat))
