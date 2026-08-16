"""
config.py — Central configuration for Yuti (v2, BPE tokenizer).

Same shape as v1_char/config.py, plus num_merges (BPE vocabulary size is
driven by how many merge rules we learn from the corpus).
"""

import torch

config = {
    # ── Data ──────────────────────────────────────────────
    "data_path":     "input.txt",
    "num_merges":    500,     # BPE merge rules to learn (-> ~500+ vocab size)

    # ── Model architecture ────────────────────────────────
    "block_size":    64,     # max context length (tokens the model sees)
    "n_embd":        128,    # embedding dimension (model width)
    "n_head":        4,      # number of attention heads
    "n_layer":       4,      # number of transformer blocks (depth)
    "dropout":       0.1,    # regularization

    # ── Training ──────────────────────────────────────────
    "batch_size":    32,     # sequences processed in parallel
    "learning_rate": 3e-4,
    "max_iters":     1500,   # total training steps
    "eval_interval": 300,    # how often to measure val loss
    "eval_iters":    200,    # batches averaged for a stable loss estimate

    # ── Runtime ───────────────────────────────────────────
    "device":        "cuda" if torch.cuda.is_available() else "cpu",
    "seed":          1337,
    "checkpoint":    "yuti.pt",  # where trained weights are saved
}
