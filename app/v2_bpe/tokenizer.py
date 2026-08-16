"""
tokenizer.py — A from-scratch byte-pair-encoding (BPE) tokenizer.

Char-level (v1) spends one token per character, so common substrings like
"the" get re-spelled every single time. BPE instead learns, from the corpus
itself, which character pairs occur most often and merges them into single
tokens — repeated: merge the new most-frequent pair again, num_merges times.
The result is a vocabulary of common chars, syllables, and whole short words.

Unlike char-level's flat stoi/itos dicts, encoding NEW text requires
replaying the learned merge rules in the exact order they were learned —
that's why this is a class with state (self.merges), not a pair of dicts.
"""


def get_pair_counts(tokens):
    """Count every adjacent pair in the token list."""
    counts = {}
    for pair in zip(tokens, tokens[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(tokens, pair, new_token):
    """Replace every adjacent occurrence of `pair` with `new_token`."""
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            new_tokens.append(new_token)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens


class BPETokenizer:
    """Trains on a corpus, then encodes/decodes text <-> token IDs.

    Same interface as the char tokenizer (encode/decode/vocab_size), so
    Yuti's model code needs ZERO changes to work with either version.
    """

    def __init__(self):
        self.merges = []       # ordered list of ((a, b), "ab") — order matters for encode()
        self.vocab = {}        # token string -> id
        self.inv_vocab = {}    # id -> token string

    def train(self, text, num_merges):
        tokens = list(text)
        base_chars = set(text)
        self.merges = []

        for _ in range(num_merges):
            counts = get_pair_counts(tokens)
            if not counts:
                break
            top = max(counts, key=counts.get)
            if counts[top] < 2:
                break
            new_tok = "".join(top)
            tokens = merge(tokens, top, new_tok)
            self.merges.append((top, new_tok))

        # Base chars get the first ids, then merged tokens in the order learned.
        self.vocab = {ch: i for i, ch in enumerate(sorted(base_chars))}
        for _, new_tok in self.merges:
            if new_tok not in self.vocab:
                self.vocab[new_tok] = len(self.vocab)
        self.inv_vocab = {i: t for t, i in self.vocab.items()}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text):
        tokens = list(text)
        for pair, new_tok in self.merges:
            tokens = merge(tokens, pair, new_tok)
        return [self.vocab[t] for t in tokens]

    def decode(self, ids):
        return "".join(self.inv_vocab[i] for i in ids)

    @classmethod
    def from_state(cls, vocab, inv_vocab, merges):
        """Rebuild a trained tokenizer from a saved checkpoint's state."""
        tok = cls()
        tok.vocab = vocab
        tok.inv_vocab = inv_vocab
        tok.merges = merges
        return tok
