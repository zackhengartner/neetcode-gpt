import numpy as np
from numpy.typing import NDArray
import math


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        sub_arr = z - max(z)
        numerator = np.exp(sub_arr)
        total = sum(np.exp(num) for num in sub_arr)   
        return np.round(numerator / total,4)