from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        unique_l = []
        text_l = list(text)
        for char in text_l:
            if char not in unique_l:
                unique_l.append(char)
        integer_l = range(len(unique_l))
        unique_l.sort()
        stoi = dict(zip(unique_l, integer_l))
        itos = dict(zip(integer_l, unique_l))
        return (stoi, itos)

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        result = []
        for i in range(len(text)):
            if stoi.get(text[i]) is not None:
                result.append(stoi[text[i]])
        return result
        

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        result_str = ""
        for i in range(len(ids)):
            if (itos.get(ids[i])) is not None:
                result_str += itos[ids[i]]
        return result_str
        
