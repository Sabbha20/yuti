"""
data.py — Everything about turning raw text into model-ready batches (v2, BPE).

Provides:
  - Dataset: loads text, trains the BPE tokenizer, holds train/val tensors
  - get_batch: random (x, y) batches for training
"""

import torch
from config import config
from tokenizer import BPETokenizer


class Dataset:
    def __init__(self, path=None):
        path = path or config["data_path"]
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        self.tokenizer = BPETokenizer()
        self.tokenizer.train(text, num_merges=config["num_merges"])
        self.vocab_size = self.tokenizer.vocab_size

        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def encode(self, s):
        return self.tokenizer.encode(s)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def get_batch(self, split):
        data = self.train_data if split == "train" else self.val_data
        bs, T = config["batch_size"], config["block_size"]
        ix = torch.randint(len(data) - T, (bs,))
        x = torch.stack([data[i:i + T] for i in ix])
        y = torch.stack([data[i + 1:i + T + 1] for i in ix])
        return x.to(config["device"]), y.to(config["device"])
