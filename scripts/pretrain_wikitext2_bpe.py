#!/usr/bin/env python3
# scripts/pretrain_wikitext2_bpe.py
"""
Milestone 3: GPT-2 BPE (tiktoken) + WikiText-2 (raw) pretraining
No Hugging Face dependency.

What you get:
- data download (with mirror + fallback)
- GPT-2 BPE tokenization + caching
- smoke tests: forward / causality / backward / param delta
- optional: overfit one fixed batch (correctness seal)
- train + val evaluation + checkpoint + sampling
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import torch

try:
    import tiktoken
except ImportError as e:
    raise SystemExit(
        "tiktoken is required for this script.\n"
        "Install: pip install -U tiktoken\n"
    ) from e

# Reuse your minimal GPT model implementation
from minimal_gpt import GPT


# -----------------------------
# Utils
# -----------------------------

def pick_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def download(url: str, dest: Path, timeout: int = 60) -> None:
    """Download URL to dest (atomic write)."""
    if dest.exists() and dest.stat().st_size > 0:
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    print(f"[download] {url}")
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def find_wikitext_files(root: Path) -> Dict[str, Path]:
    """
    Return dict with keys train/valid/test pointing to raw files.
    We accept either raw or tokens files.
    """
    patterns = [
        ("train", ["wiki.train.raw", "wiki.train.tokens"]),
        ("valid", ["wiki.valid.raw", "wiki.valid.tokens"]),
        ("test",  ["wiki.test.raw",  "wiki.test.tokens"]),
    ]

    found: Dict[str, Path] = {}
    for split, names in patterns:
        for name in names:
            hits = list(root.rglob(name))
            if hits:
                found[split] = hits[0]
                break

    missing = [k for k in ["train", "valid", "test"] if k not in found]
    if missing:
        raise FileNotFoundError(f"Cannot find files for splits: {missing} under {root}")

    return found


def ensure_wikitext2_raw(data_dir: Path) -> Dict[str, Path]:
    """
    Prefer Smerity mirror zip (raw). If that fails, fallback to a public raw-file mirror.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1) Try Smerity zip mirror (raw)
    zip_url = "https://wikitext.smerity.com/wikitext-2-raw-v1.zip"
    zip_path = data_dir / "wikitext-2-raw-v1.zip"

    extracted_marker = data_dir / ".extracted_ok"

    if not extracted_marker.exists():
        try:
            download(zip_url, zip_path)
            extract_zip(zip_path, data_dir)
            extracted_marker.write_text("ok")
        except Exception as e:
            print(f"[warn] failed to download/extract smerity zip: {e}")
            print("[warn] falling back to raw files mirror (no zip)")

            # 2) Fallback: direct raw files
            # (This mirror hosts wiki.train.raw/wiki.valid.raw/wiki.test.raw)
            base = "https://cosmo.zip/pub/datasets/wikitext-2-raw"
            for split in ["train", "valid", "test"]:
                url = f"{base}/wiki.{split}.raw"
                dest = data_dir / f"wiki.{split}.raw"
                download(url, dest)

            extracted_marker.write_text("ok_fallback")

    files = find_wikitext_files(data_dir)
    print("[data] files:")
    for k, p in files.items():
        print(f"  - {k}: {p}")
    return files


def encode_text(enc, text: str) -> torch.Tensor:
    # Add end-of-text token once at the end (optional but helpful as a boundary)
    eot = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
    ids = enc.encode(text) + [eot]
    return torch.tensor(ids, dtype=torch.int32)


def prepare_tokens(
    enc,
    files: Dict[str, Path],
    cache_dir: Path,
    force_rebuild: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize train/valid/test and cache as torch tensors on disk.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, torch.Tensor] = {}

    for split, path in files.items():
        cache_path = cache_dir / f"{split}.pt"
        if cache_path.exists() and not force_rebuild:
            data = torch.load(cache_path, map_location="cpu")
            out[split] = data
            continue

        print(f"[tokenize] split={split} reading {path.name}")
        text = path.read_text(encoding="utf-8", errors="replace")
        t0 = time.time()
        data = encode_text(enc, text)
        dt = time.time() - t0
        torch.save(data, cache_path)
        print(f"[tokenize] split={split} tokens={data.numel():,} saved={cache_path} ({dt:.2f}s)")
        out[split] = data

    return out


def get_batch(data_1d: torch.Tensor, block_size: int, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    data_1d: 1D tensor of token ids (int32 or int64)
    returns x,y of shape (B,T) in int64 on device
    """
    n = data_1d.size(0)
    if n <= block_size + 1:
        raise ValueError(f"Dataset too small: n={n}, block_size={block_size}")

    ix = torch.randint(0, n - block_size - 1, (batch_size,))
    x = torch.stack([data_1d[i:i + block_size] for i in ix]).long().to(device)
    y = torch.stack([data_1d[i + 1:i + block_size + 1] for i in ix]).long().to(device)
    return x, y


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
    eval_iters: int,
) -> Dict[str, float]:
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(eval_iters):
            xb, yb = get_batch(data, block_size, batch_size, device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = float(sum(losses) / len(losses))
    model.train()
    return out


def configure_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """
    AdamW with a typical decoupled weight decay pattern:
    - decay for 2D weights (matmul weights)
    - no decay for biases and LayerNorm/Embedding weights
    """
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2 and (".ln" not in name) and ("ln_" not in name) and (not name.endswith(".bias")):
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)
    return optimizer


def get_lr(step: int, *, base_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    if max_steps <= 0:
        return base_lr
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }
    torch.save(ckpt, path)


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Pretrain minimal GPT on WikiText-2 (raw) with GPT-2 BPE (tiktoken).")
    parser.add_argument("--run_name", type=str, default="wt2_bpe")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default=None, help="cuda|mps|cpu (default: auto)")

    # Model
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--batch_size", type=int, default=4, help="micro-batch size (keep small; logits is huge with 50k vocab)")
    parser.add_argument("--grad_accum", type=int, default=8, help="gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Eval / logging
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_iters", type=int, default=25)

    # Correctness seal
    parser.add_argument("--overfit_one_batch_steps", type=int, default=0, help="if >0: overfit a fixed batch for N steps")

    # Data
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--force_rebuild_tokens", action="store_true")

    # Sampling
    parser.add_argument("--sample_prompt", type=str, default="The history of")
    parser.add_argument("--sample_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=40)

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    print(f"[device] {device}")

    # Project root: repo_root/scripts/this_file.py -> repo_root
    repo_root = Path(__file__).resolve().parent.parent
    run_dir = repo_root / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir) if args.data_dir else (repo_root / "data" / "wikitext2_raw")
    cache_dir = repo_root / "data" / "cache_wt2_gpt2_bpe"

    # Tokenizer
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab  # should be 50257 for GPT-2
    print(f"[tokenizer] gpt2_bpe vocab_size={vocab_size}")

    # quick tokenizer sanity
    test_strings = ["GPT\\", "Hello world!", "中文也可以", "emoji🙂 test"]
    for s in test_strings:
        ids = enc.encode(s)
        s2 = enc.decode(ids)
        print(f"[tok_test] {s!r} -> {len(ids)} tokens -> roundtrip_ok={s2 == s}")

    # Data
    files = ensure_wikitext2_raw(data_dir)
    tokens = prepare_tokens(enc, files, cache_dir, force_rebuild=args.force_rebuild_tokens)

    train_data = tokens["train"]
    val_data = tokens["valid"]

    print(f"[data] train_tokens={train_data.numel():,} val_tokens={val_data.numel():,}")

    # Baseline check
    baseline = math.log(vocab_size)
    print(f"[baseline] ln(vocab_size)={baseline:.4f}")

    # Model
    model = GPT(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params/1e6:.3f}M")

    optimizer = configure_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    # -----------------------------
    # Smoke tests: forward / causality / backward / param delta
    # -----------------------------
    model.train()
    xb, yb = get_batch(train_data, args.block_size, args.batch_size, device)
    logits, loss = model(xb, yb)
    print(f"[smoke.forward] logits={tuple(logits.shape)} loss={loss.item():.4f}")

    # causality test in eval mode (so dropout won't affect)
    with torch.no_grad():
        model.eval()
        test = xb[:1].clone()
        logits1, _ = model(test)
        test2 = test.clone()
        test2[0, -1] = (test2[0, -1] + 1) % vocab_size
        logits2, _ = model(test2)
        diff = (logits1[:, :-1, :] - logits2[:, :-1, :]).abs().max().item()
        print(f"[smoke.causality] max_diff_on_past_positions={diff:.6f} (should be ~0)")
        model.train()

    optimizer.zero_grad(set_to_none=True)
    _, loss = model(xb, yb)
    loss.backward()

    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5
    print(f"[smoke.backward] grad_norm={grad_norm:.4f}")

    before = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
    optimizer.step()
    after = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    print(f"[smoke.update] param_delta_norm={(after - before).norm().item():.6f}")

    # -----------------------------
    # Optional: overfit one fixed batch (correctness seal)
    # -----------------------------
    if args.overfit_one_batch_steps > 0:
        print(f"\n[overfit-one-batch] steps={args.overfit_one_batch_steps}")
        model.train()
        xb_fix, yb_fix = get_batch(train_data, args.block_size, args.batch_size, device)
        optimizer = configure_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

        for i in range(args.overfit_one_batch_steps):
            _, l = model(xb_fix, yb_fix)
            optimizer.zero_grad(set_to_none=True)
            l.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if (i + 1) % 50 == 0 or i == 0:
                print(f"[overfit-one-batch] step {i+1} loss={l.item():.6f}")

    # -----------------------------
    # Train
    # -----------------------------
    if args.max_steps <= 0:
        print("\n[done] max_steps<=0, exit after smoke tests.")
        return

    print(f"\n[train] max_steps={args.max_steps} batch_size={args.batch_size} grad_accum={args.grad_accum} block_size={args.block_size}")
    model.train()
    optimizer = configure_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    t0 = time.time()
    for step in range(args.max_steps):
        # LR schedule
        lr = get_lr(
            step,
            base_lr=args.lr,
            min_lr=args.min_lr,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Eval + checkpoint + sample
        if step % args.eval_interval == 0 or step == args.max_steps - 1:
            losses = estimate_loss(
                model,
                train_data=train_data,
                val_data=val_data,
                block_size=args.block_size,
                batch_size=args.batch_size,
                device=device,
                eval_iters=args.eval_iters,
            )
            elapsed = time.time() - t0
            print(f"\n[eval] step={step} lr={lr:.2e} train_loss={losses['train']:.4f} val_loss={losses['val']:.4f} elapsed={elapsed:.1f}s")

            # sample
            with torch.no_grad():
                model.eval()
                prompt_ids = enc.encode(args.sample_prompt)
                idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                out = model.generate(idx, max_new_tokens=args.sample_tokens, temperature=args.temperature, top_k=args.top_k)
                text = enc.decode(out[0].tolist())
                model.train()
            print("[sample]")
            print(text)

            # save
            ckpt_path = run_dir / "ckpt_latest.pt"
            config = vars(args).copy()
            config.update({
                "vocab_size": vocab_size,
                "encoding": "gpt2",
            })
            save_checkpoint(ckpt_path, model, optimizer, step=step, config=config)
            # also save a step checkpoint occasionally
            if step % (args.eval_interval * 5) == 0:
                save_checkpoint(run_dir / f"ckpt_step{step:06d}.pt", model, optimizer, step=step, config=config)
            print(f"[ckpt] saved: {ckpt_path}")

        # gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for micro in range(args.grad_accum):
            xb, yb = get_batch(train_data, args.block_size, args.batch_size, device)
            _, loss = model(xb, yb)
            loss = loss / args.grad_accum
            loss.backward()
            loss_accum += loss.item()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        if step % args.log_interval == 0:
            print(f"[train] step={step} lr={lr:.2e} loss={loss_accum:.4f}")

    print("\n[done] training finished.")


if __name__ == "__main__":
    main()
