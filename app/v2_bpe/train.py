"""
train.py — Train Yuti (v2, BPE tokenizer) and save a checkpoint.

Run:  uv run python train.py   (from inside app/v2_bpe/)

Same five-step loop as v1_char. The only real difference is what gets saved:
BPE encoding needs the learned merge rules (in order) to encode new text,
not just a flat vocab dict — so the checkpoint carries vocab/inv_vocab/merges
instead of v1_char's stoi/itos.
"""

import torch
from config import config
from data import Dataset
from model import YutiGPT


@torch.no_grad()
def estimate_loss(model, dataset):
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
    print("Vocab size (BPE):", dataset.vocab_size)

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

    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab_size": dataset.vocab_size,
            "vocab": dataset.tokenizer.vocab,
            "inv_vocab": dataset.tokenizer.inv_vocab,
            "merges": dataset.tokenizer.merges,
            "config": config,
        },
        config["checkpoint"],
    )
    print(f"\nSaved checkpoint -> {config['checkpoint']}")


if __name__ == "__main__":
    main()
