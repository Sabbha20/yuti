# Yuti — a char-level GPT, from scratch

A tiny decoder-only transformer (nanoGPT-style) built for learning the
internals. CPU-friendly. Same architecture family as GPT/LLaMA/Claude.

## Files
- `config.py`   — all hyperparameters in one place
- `data.py`     — char-level tokenizer + batching (`Dataset`)
- `model.py`    — Head → MultiHeadAttention → FeedForward → Block → YutiGPT
- `train.py`    — training loop, saves `yuti.pt`
- `generate.py` — loads `yuti.pt`, generates text

## Setup (uv)
```bash
uv venv
uv add torch --index pytorch-cpu=https://download.pytorch.org/whl/cpu
```

## Get data
```bash
curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## Run
```bash
uv run python train.py
uv run python generate.py --prompt "ROMEO:" --tokens 500
```

## Notes
Yuti is a BASE model — it continues text, it does not chat. To make it
conversational you'd add instruction fine-tuning + RLHF on a larger model.
Scale `n_embd`, `n_layer`, `n_head`, `max_iters` in `config.py` for better output.