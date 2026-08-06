"""
config.py — Central configuration for Yuti.

All hyperparameters live here so experiments = editing ONE file.
Import this everywhere: `from config import config`.
"""

import torch

config = {
    # ── Data ──────────────────────────────────────────────
    "data_path":     "input.txt",

    # ── Model architecture ────────────────────────────────
    "block_size":    64,     # max context length (chars the model sees)
    "n_embd":        128,    # embedding dimension (model width)
    "n_head":        4,      # number of attention heads
    "n_layer":       4,      # number of transformer blocks (depth)
    "dropout":       0.1,    # regularization

    # ── Training ──────────────────────────────────────────
    "batch_size":    32,     # sequences processed in parallel
    "learning_rate": 3e-4,
    "max_iters":     3000,   # total training steps
    "eval_interval": 300,    # how often to measure val loss
    "eval_iters":    200,    # batches averaged for a stable loss estimate

    # ── Runtime ───────────────────────────────────────────
    "device":        "cuda" if torch.cuda.is_available() else "cpu",
    "seed":          1337,
    "checkpoint":    "yuti.pt",  # where trained weights are saved
}