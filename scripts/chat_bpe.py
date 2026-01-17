#!/usr/bin/env python3
# scripts/chat_bpe.py
"""
Interactive chat / prompt completion for a GPT checkpoint trained with GPT-2 BPE (tiktoken).
No Hugging Face dependency.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import tiktoken

from minimal_gpt import GPT


def pick_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]

    enc_name = cfg.get("encoding", "gpt2")
    enc = tiktoken.get_encoding(enc_name)
    vocab_size = cfg.get("vocab_size", enc.n_vocab)

    model = GPT(
        vocab_size=vocab_size,
        block_size=cfg["block_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        dropout=0.0,  # inference
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    return model, enc, cfg


def main():
    parser = argparse.ArgumentParser(description="Chat / completion with a BPE-trained minimal GPT checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="runs/wt2_bpe/ckpt_latest.pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--keep", type=int, default=256, help="how many recent tokens to keep as context")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    device = pick_device(args.device)
    print(f"[device] {device}")
    print(f"[load] {ckpt_path}")

    model, enc, cfg = load_checkpoint(ckpt_path, device)
    print(f"[model] block_size={model.block_size} n_layer={cfg['n_layer']} n_head={cfg['n_head']} n_embd={cfg['n_embd']}")
    print(f"[tokenizer] {cfg.get('encoding','gpt2')} vocab={enc.n_vocab}")
    print("=" * 60)
    print("Completion mode. Commands: /clear /temp <v> /topk <k> /quit")
    print("=" * 60)

    ctx_tokens: list[int] = []

    while True:
        try:
            user = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if not user.strip():
            continue

        if user.strip().lower() in ["/quit", "/exit", "quit", "exit"]:
            print("bye")
            break

        if user.startswith("/"):
            if user.strip() == "/clear":
                ctx_tokens = []
                print("context cleared")
                continue
            if user.startswith("/temp "):
                try:
                    args.temperature = float(user.split()[1])
                    print(f"temperature={args.temperature}")
                except Exception:
                    print("usage: /temp <float>")
                continue
            if user.startswith("/topk "):
                try:
                    args.top_k = int(user.split()[1])
                    print(f"top_k={args.top_k}")
                except Exception:
                    print("usage: /topk <int>")
                continue
            print("unknown command")
            continue

        # append user text (+ newline as a mild separator)
        user_tokens = enc.encode(user + "\n")
        ctx_tokens.extend(user_tokens)

        # keep recent context
        if len(ctx_tokens) > args.keep:
            ctx_tokens = ctx_tokens[-args.keep:]

        idx = torch.tensor([ctx_tokens], dtype=torch.long, device=device)

        with torch.no_grad():
            out = model.generate(
                idx,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )

        out_tokens = out[0].tolist()
        gen_tokens = out_tokens[len(ctx_tokens):]
        gen_text = enc.decode(gen_tokens)

        print(f"GPT: {gen_text}")

        # update context to include generated tokens, then crop
        ctx_tokens = out_tokens[-args.keep:]


if __name__ == "__main__":
    main()
