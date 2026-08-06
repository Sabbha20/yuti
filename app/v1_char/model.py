"""
model.py — The Yuti GPT architecture, built bottom-up.

Component stack (small pieces compose into the whole):
    Head              one self-attention head (Q/K/V + causal mask)
    MultiHeadAttention  several heads in parallel, then projected
    FeedForward       per-position MLP (the "thinking" step)
    Block             attention + feed-forward, with residuals + layer norm
    YutiGPT           embeddings -> stacked blocks -> output head

Every tensor keeps the shape contract (B, T, C):
    B = batch (sequences in parallel)
    T = time  (positions / characters in the sequence)
    C = channels (vector size per position; n_embd internally, vocab at the end)

This is a decoder-only transformer — the same family as GPT/LLaMA/Claude.
The causal mask (see Head) is what makes it "decoder-only": each position
may attend to itself and the past, never the future.
"""

import torch
from torch import nn
from torch.nn import functional as F

from app.v1_char.config import config


class Head(nn.Module):
    """One head of self-attention.

    Each position emits a query ("what am I looking for?"), a key ("what do I
    contain?") and a value ("what I'll hand over if attended to"). A position's
    query is dotted against every earlier position's key to get attention
    weights; the output is the weighted sum of values.
    """

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(config["n_embd"], head_size, bias=False)
        self.query = nn.Linear(config["n_embd"], head_size, bias=False)
        self.value = nn.Linear(config["n_embd"], head_size, bias=False)

        # Lower-triangular causal mask. Registered as a buffer (not a trainable
        # parameter) so it moves with .to(device) but isn't updated by the optimizer.
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config["block_size"], config["block_size"])),
        )
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        _B, T, _C = x.shape
        k = self.key(x)     # (B, T, head_size)
        q = self.query(x)   # (B, T, head_size)

        # Attention scores, scaled by 1/sqrt(head_size) to keep variance stable.
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5   # (B, T, T)

        # Mask the future: set upper-triangle scores to -inf so softmax zeroes them.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)   # (B, T, head_size)
        return wei @ v      # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    """Several attention heads in parallel, concatenated and projected back.

    Each head learns a different kind of relationship; running them in parallel
    captures multiple patterns at once. head_size = n_embd // num_heads, so the
    concatenated output returns to n_embd.
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, config["n_embd"])
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embd)
        return self.dropout(self.proj(out))                  # (B, T, n_embd)


class FeedForward(nn.Module):
    """Per-position MLP. Attention mixes ACROSS positions; this thinks WITHIN
    each position. The 4x expansion gives the network room to compute before
    projecting back; ReLU provides the non-linearity that makes depth useful.
    """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(config["dropout"]),
        )

    def forward(self, x):
        return self.net(x)   # (B, T, n_embd)


class Block(nn.Module):
    """One transformer block: communication (attention) then computation (FF).

    The pattern x = x + sublayer(norm(x)) appears twice:
      - '+ x' is a residual connection: a gradient highway that lets deep
        stacks train, and lets each layer make small refinements.
      - norm(...) is pre-layer-norm: stabilizes the distribution entering each
        sublayer (modern GPT-2+ convention).
    """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))     # attend
        x = x + self.ffwd(self.ln2(x))   # think
        return x


class YutiGPT(nn.Module):
    """The full model: token + position embeddings -> blocks -> norm -> head."""

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        # Token embedding: token id -> meaning vector.
        self.token_embedding_table = nn.Embedding(vocab_size, config["n_embd"])
        # Position embedding: slot 0..block_size-1 -> position vector.
        self.position_embedding_table = nn.Embedding(config["block_size"], config["n_embd"])

        self.blocks = nn.Sequential(
            *[Block(config["n_embd"], config["n_head"]) for _ in range(config["n_layer"])]
        )
        self.ln_f = nn.LayerNorm(config["n_embd"])
        self.lm_head = nn.Linear(config["n_embd"], vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                                    # (B,T,n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T,n_embd)
        x = tok_emb + pos_emb        # inject identity AND order
        x = self.blocks(x)           # attention + feed-forward, n_layer times
        x = self.ln_f(x)
        logits = self.lm_head(x)     # (B, T, vocab_size)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Autoregressive generation: predict next char, append, repeat."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -config["block_size"]:]      # crop to context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                      # last position only
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
