import torch
from torchtyping import TensorType
from typing import Tuple

class Solution:
    def create_batches(self, data: TensorType[int], context_length: int, batch_size: int) -> Tuple[TensorType[int], TensorType[int]]:
        # data: 1D tensor of encoded text (integer token IDs)
        # context_length: number of tokens in each training example
        # batch_size: number of examples per batch
        #
        # Return (X, Y) where:
        # - X has shape (batch_size, context_length)
        # - Y has shape (batch_size, context_length)
        # - Y is X shifted right by 1 (Y[i][j] = data[start_i + j + 1])
        #
        # Use torch.manual_seed(0) before generating random start indices
        # Use torch.randint to pick random starting positions
        xs = []
        ys = []
        torch.manual_seed(0)
        rand_t = torch.randint(0, len(data) - context_length, (batch_size,)).tolist()
        for offset in rand_t:
            xs.append(data[offset: offset + context_length])
            ys.append(data[offset + 1: offset + context_length + 1])
        x_t = torch.stack(xs)
        y_t = torch.stack(ys)
        return (x_t, y_t)

