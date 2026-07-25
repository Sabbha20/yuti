"""
data.py — Everything about turning raw text into model-ready batches.

Provides:
  - Dataset: loads text, builds the char-level tokenizer, holds train/val tensors
  - encode / decode: text <-> integer-list helpers
  - get_batch: random (x, y) batches for training

Char-level design: 1 character = 1 token. Simplest possible tokenizer,
zero hidden complexity — perfect for learning. (Real LLMs use BPE.)
"""

import torch
from config import config


class Dataset:
    def __init__(self, path=None):
        path = path or config["data_path"]
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # Vocabulary = every unique character in the corpus
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        # Lookup tables: string<->int
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        # Encode the ENTIRE corpus into one long tensor of ints
        data = torch.tensor(self.encode(text), dtype=torch.long)

        # 90/10 train/val split (sequential — order matters for text)
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def encode(self, s):
        """'hi' -> [46, 47]"""
        return [self.stoi[c] for c in s]

    def decode(self, tokens):
        """[46, 47] -> 'hi'"""
        return "".join(self.itos[i] for i in tokens)

    def get_batch(self, split):
        """Return a random (x, y) batch. y is x shifted right by one char."""
        data = self.train_data if split == "train" else self.val_data
        bs, T = config["batch_size"], config["block_size"]
        ix = torch.randint(len(data) - T, (bs,))
        x = torch.stack([data[i:i + T] for i in ix])
        y = torch.stack([data[i + 1:i + T + 1] for i in ix])
        return x.to(config["device"]), y.to(config["device"])
    
