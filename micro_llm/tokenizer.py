class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}

    def encode(self, text):
        return [self.char_to_idx[c] for c in text]

    def decode(self, tokens):
        return ''.join([self.idx_to_char[t] for t in tokens])
