from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        result = []
        for num in numbers:
            s = str(num)
            tokens = []
            i = 0
            n = len(s)
            while i < n:
                matched = False
                for j in range(n, i, -1):
                    if s[i:j] in vocab:
                        tokens.append(s[i:j])
                        i = j
                        matched = True
                        break
                if not matched:
                    tokens.append(s[i])
                    i += 1
            result.append(tokens)
        return result
    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        token_count = 0
        i = 0
        n = len(text)
        while i < n:
            matched = False
            for j in range(n, i, -1):
                if text[i:j] in vocab:
                    token_count += 1
                    i = j
                    matched = True
                    break
            if not matched:
                token_count += 1
                i += 1
        return token_count

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        words_l = text.split()
        return round(self.count_tokens(text,vocab) / len(words_l),4)
