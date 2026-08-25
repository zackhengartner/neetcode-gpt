import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        z = self.forward(x, w, b)
        y_hat = self.activation(z)
        np_loss = self.loss(y_hat, y_true)
        dw = (y_hat - y_true) * y_hat * (1 - y_hat) * x
        db = (y_hat - y_true) * y_hat * (1 - y_hat)
        return np.round(dw, 5), float(np.round(db,5))


    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float) -> NDArray[np.float64]:
        return x @ w + b

    def activation(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        return 1 / (1 + np.exp(-z))

    def loss(self, y_hat: NDArray[np.float64], y_true: float) -> NDArray[np.float64]:
        return 0.5 * (y_hat - y_true) ** 2