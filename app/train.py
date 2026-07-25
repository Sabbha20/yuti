"""
train.py — Train Yuti and save a checkpoint.

Run:  python train.py

The training loop is the whole game, five steps repeated MAX_ITERS times:
    1. get a batch
    2. forward pass -> loss
    3. zero old gradients
    4. loss.backward()  (autograd computes all gradients)
    5. optimizer.step() (nudge weights to reduce loss)

We periodically measure val loss (averaged over many batches for stability)
so we can watch for learning vs. overfitting. At the end we save the model
weights, its config, and the tokenizer vocab so generate.py can reload it.
"""

import torch

from config import config
from data import Dataset
from model import YutiGPT


@torch.no_grad()
def estimate_loss(model, dataset):
    """Average loss over EVAL_ITERS batches — a low-noise progress number."""
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(config["eval_iters"])
        for k in range(config["eval_iters"]):
            X, Y = dataset.get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    torch.manual_seed(config["seed"])
    print("Device:", config["device"])

    dataset = Dataset()
    print("Vocab size:", dataset.vocab_size)

    model = YutiGPT(dataset.vocab_size).to(config["device"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Yuti parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    history = {"iter": [], "train": [], "val": []}

    for it in range(config["max_iters"]):
        if it % config["eval_interval"] == 0 or it == config["max_iters"] - 1:
            losses = estimate_loss(model, dataset)
            history["iter"].append(it)
            history["train"].append(losses["train"])
            history["val"].append(losses["val"])
            print(f"step {it:4d} | train {losses['train']:.4f} | val {losses['val']:.4f}")

        xb, yb = dataset.get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save everything generate.py needs to rebuild and run the model.
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab_size": dataset.vocab_size,
            "stoi": dataset.stoi,
            "itos": dataset.itos,
            "config": config,
        },
        config["checkpoint"],
    )
    print(f"\nSaved checkpoint -> {config['checkpoint']}")


if __name__ == "__main__":
    main()