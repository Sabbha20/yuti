"""
generate.py — Load a trained Yuti checkpoint and generate text.

Run:  python generate.py
      python generate.py --prompt "ROMEO:" --tokens 500

Yuti is a BASE model: it continues text, it does not chat. Give it a seed
(a "prompt") and it riffs on the pattern it learned. An empty seed starts
from a newline and free-writes.
"""

import argparse
import torch

from model import YutiGPT


def load_model(checkpoint_path):
    """Rebuild the model from a saved checkpoint and restore the tokenizer."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Restore the exact vocab so encode/decode line up with training.
    stoi, itos = ckpt["stoi"], ckpt["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda toks: "".join(itos[i] for i in toks)

    model = YutiGPT(ckpt["vocab_size"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    device = ckpt["config"]["device"]
    model.to(device)
    return model, encode, decode, device


def main():
    parser = argparse.ArgumentParser(description="Generate text with Yuti.")
    parser.add_argument("--checkpoint", default="yuti.pt")
    parser.add_argument("--prompt", default="", help="Seed text for Yuti to continue.")
    parser.add_argument("--tokens", type=int, default=500, help="How many chars to generate.")
    args = parser.parse_args()

    model, encode, decode, device = load_model(args.checkpoint)

    # Seed context: the prompt if given, else a single newline (token 0-ish).
    if args.prompt:
        idx = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)

    out = model.generate(idx, max_new_tokens=args.tokens)
    print(decode(out[0].tolist()))


if __name__ == "__main__":
    main()