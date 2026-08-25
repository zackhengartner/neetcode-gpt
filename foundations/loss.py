import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        y_pred += 1e-7
        log_p1 = np.log(1-y_pred)
        log_p2 = np.log(y_pred)
        L = 0
        for i in range(len(y_true)):
            L += y_true[i] * log_p2[i] + (1 - y_true[i]) * log_p1[i]
        return round(L / -len(y_true),4)
        pass

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        y_pred += 1e-7
        L=0
        for true_row, pred_row in zip(y_true, y_pred):
            for y, p in zip(true_row,  pred_row):
                L += y * math.log(p)
        return round (L/-len(y_true),4)
        pass
