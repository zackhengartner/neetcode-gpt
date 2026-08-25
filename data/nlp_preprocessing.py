import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        combined = positive + negative
        vocab = []
        for sentence in combined:
            words = sentence.split()
            for word in words:
                if word not in vocab:
                    vocab.append(word)
        vocab.sort()
        word_to_id = {}

        for i, word in enumerate(vocab, start=1):
            word_to_id[word] = i

        tensors = []

        for sentence in positive + negative:
            ids = [word_to_id[word] for word in sentence.split()]
            tensors.append(torch.tensor(ids))

        padded = nn.utils.rnn.pad_sequence(
            tensors,
            batch_first=True
        )

        return padded