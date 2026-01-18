#!/usr/bin/env python3
# scripts/pretrain_wikitext2_bpe_repro.py
"""
Milestone 4: Reproducible pretraining runner
- config snapshot (config.json)
- jsonl metrics (metrics.jsonl)
- ckpt_latest + ckpt_best
- resume
- samples saved to files
- smoke tests do NOT change the true training start weights
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import torch

try:
    import tiktoken
except ImportError as e:
    raise SystemExit("Need tiktoken: pip install -U tiktoken") from e

from minimal_gpt import GPT


# -----------------------------
# Repro helpers
# -----------------------------

def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root))
        return out.decode().strip()
    except Exception:
        return None


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def pick_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_mem_stats(device: torch.device) -> dict:
    stats: dict = {"device": str(device)}
    if device.type == "cuda":
        stats.update({
            "cuda_allocated": int(torch.cuda.memory_allocated()),
            "cuda_reserved": int(torch.cuda.memory_reserved()),
            "cuda_max_allocated": int(torch.cuda.max_memory_allocated()),
        })
    elif device.type == "mps":
        # available on newer torch builds
        try:
            stats.update({
                "mps_allocated": int(torch.mps.current_allocated_memory()),
                "mps_driver_allocated": int(torch.mps.driver_allocated_memory()),
            })
        except Exception:
            pass
    return stats


# -----------------------------
# Data: WikiText-2 raw
# -----------------------------

def download(url: str, dest: Path, timeout: int = 60) -> None:
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
        raise FileNotFoundError(f"Missing splits: {missing} under {root}")
    return found


def ensure_wikitext2_raw(data_dir: Path) -> Dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_url = "https://wikitext.smerity.com/wikitext-2-raw-v1.zip"
    zip_path = data_dir / "wikitext-2-raw-v1.zip"
    extracted_marker = data_dir / ".extracted_ok"

    if not extracted_marker.exists():
        try:
            download(zip_url, zip_path)
            extract_zip(zip_path, data_dir)
            extracted_marker.write_text("ok")
        except Exception as e:
            print(f"[warn] smerity zip failed: {e}")
            print("[warn] fallback to raw files mirror (cosmo.zip)")
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
    eot = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
    ids = enc.encode(text) + [eot]
    return torch.tensor(ids, dtype=torch.int32)


def prepare_tokens(enc, files: Dict[str, Path], cache_dir: Path, force: bool) -> Dict[str, torch.Tensor]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, torch.Tensor] = {}

    for split, path in files.items():
        cache_path = cache_dir / f"{split}.pt"
        if cache_path.exists() and not force:
            out[split] = torch.load(cache_path, map_location="cpu")
            continue

        print(f"[tokenize] split={split} reading {path.name}")
        text = path.read_text(encoding="utf-8", errors="replace")
        t0 = time.time()
        data = encode_text(enc, text)
        torch.save(data, cache_path)
        print(f"[tokenize] split={split} tokens={data.numel():,} saved={cache_path} ({time.time()-t0:.2f}s)")
        out[split] = data

    return out


def get_batch(data_1d: torch.Tensor, block_size: int, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    n = data_1d.size(0)
    if n <= block_size + 1:
        raise ValueError(f"Dataset too small: n={n}, block_size={block_size}")
    ix = torch.randint(0, n - block_size - 1, (batch_size,))
    x = torch.stack([data_1d[i:i + block_size] for i in ix]).long().to(device)
    y = torch.stack([data_1d[i + 1:i + block_size + 1] for i in ix]).long().to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model: GPT, train_data: torch.Tensor, val_data: torch.Tensor,
                  block_size: int, batch_size: int, device: torch.device, eval_iters: int) -> dict:
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
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2 and (not name.endswith(".bias")) and ("ln" not in name.lower()):
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


def get_lr(step: int, *, base_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    if max_steps <= 0:
        return base_lr
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (base_lr - min_lr)


def save_checkpoint(path: Path, model: GPT, optimizer: torch.optim.Optimizer, step: int,
                    config: dict, best_val: float | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        "best_val": best_val,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }
    torch.save(ckpt, path)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device)


# -----------------------------
# Main
# -----------------------------

def main():
    p = argparse.ArgumentParser(description="Reproducible WikiText-2 GPT-2-BPE pretrain runner (Milestone 4).")
    p.add_argument("--run_name", type=str, default="wt2_bpe_repro")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--resume", action="store_true", help="resume from runs/<run_name>/ckpt_latest.pt if exists")

    # Model
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)

    # Train
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Eval / log
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--eval_interval", type=int, default=300)
    p.add_argument("--eval_iters", type=int, default=25)
    p.add_argument("--save_step_ckpt_every", type=int, default=1500)

    # Data
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--force_rebuild_tokens", action="store_true")

    # Sampling
    p.add_argument("--sample_prompt", type=str, default="The history of")
    p.add_argument("--sample_tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=40)

    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    run_dir = repo_root / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    print(f"[device] {device}")

    seed_all(args.seed)

    metrics_path = run_dir / "metrics.jsonl"
    config_path = run_dir / "config.json"
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    print(f"[tokenizer] gpt2_bpe vocab_size={vocab_size}")

    # Data
    data_dir = Path(args.data_dir) if args.data_dir else (repo_root / "data" / "wikitext2_raw")
    cache_dir = repo_root / "data" / "cache_wt2_gpt2_bpe"
    files = ensure_wikitext2_raw(data_dir)
    toks = prepare_tokens(enc, files, cache_dir, force=args.force_rebuild_tokens)
    train_data, val_data = toks["train"], toks["valid"]

    # Build run config snapshot (what you need for reproducibility)
    run_config = {
        "run_name": args.run_name,
        "time_start": now_iso(),
        "argv": sys.argv,
        "git_commit": get_git_commit(repo_root),
        "platform": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "versions": {
            "torch": torch.__version__,
            "tiktoken": getattr(tiktoken, "__version__", None),
        },
        "device": str(device),
        "data": {
            "train_tokens": int(train_data.numel()),
            "val_tokens": int(val_data.numel()),
            "data_dir": str(data_dir),
            "cache_dir": str(cache_dir),
        },
        "model": {
            "vocab_size": vocab_size,
            "block_size": args.block_size,
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "n_embd": args.n_embd,
            "dropout": args.dropout,
        },
        "train": {
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "max_steps": args.max_steps,
            "lr": args.lr,
            "min_lr": args.min_lr,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "log_interval": args.log_interval,
            "eval_interval": args.eval_interval,
            "eval_iters": args.eval_iters,
        },
        "sample": {
            "prompt": args.sample_prompt,
            "sample_tokens": args.sample_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
        },
    }
    write_json(config_path, run_config)

    # Build / Resume
    ckpt_latest_path = run_dir / "ckpt_latest.pt"
    ckpt_best_path = run_dir / "ckpt_best.pt"

    start_step = 0
    best_val = float("inf")

    model = GPT(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    ).to(device)

    optimizer = configure_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    if args.resume:
        if not ckpt_latest_path.exists():
            raise FileNotFoundError(f"--resume but no checkpoint: {ckpt_latest_path}")
        ckpt = load_checkpoint(ckpt_latest_path, device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = int(ckpt["step"]) + 1
        best_val = float(ckpt.get("best_val", best_val))
        print(f"[resume] from step={start_step} best_val={best_val:.4f}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params/1e6:.3f}M")
    print(f"[baseline] ln(vocab)={math.log(vocab_size):.4f}")

    # -----------------------------
    # Smoke tests (do NOT change training start)
    # -----------------------------
    # Save initial weights on CPU (small enough at 16M params)
    init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.train()
    xb, yb = get_batch(train_data, args.block_size, args.batch_size, device)
    logits, loss = model(xb, yb)
    print(f"[smoke.forward] logits={tuple(logits.shape)} loss={loss.item():.4f}")

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
    gnorm = 0.0
    for p_ in model.parameters():
        if p_.grad is not None:
            gnorm += p_.grad.data.norm(2).item() ** 2
    gnorm = gnorm ** 0.5
    print(f"[smoke.backward] grad_norm={gnorm:.4f}")

    # param delta test with a TEMP step, then restore
    before = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu()
    tmp_opt = configure_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    tmp_opt.zero_grad(set_to_none=True)
    _, l = model(xb, yb)
    l.backward()
    tmp_opt.step()
    after = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu()
    print(f"[smoke.update] param_delta_norm={(after - before).norm().item():.6f}")

    # Restore original init weights (so training is reproducible)
    model.load_state_dict(init_state)

    # -----------------------------
    # Train loop
    # -----------------------------
    if args.max_steps <= 0:
        print("[done] max_steps<=0, exit after smoke tests.")
        return

    tokens_per_step = args.batch_size * args.grad_accum * args.block_size
    total_tokens = start_step * tokens_per_step
    t0 = time.time()

    for step in range(start_step, args.max_steps):
        lr = get_lr(step, base_lr=args.lr, min_lr=args.min_lr,
                    warmup_steps=args.warmup_steps, max_steps=args.max_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Eval / ckpt / sample
        if step % args.eval_interval == 0 or step == args.max_steps - 1:
            losses = estimate_loss(
                model, train_data, val_data,
                block_size=args.block_size,
                batch_size=args.batch_size,
                device=device,
                eval_iters=args.eval_iters,
            )
            train_loss = losses["train"]
            val_loss = losses["val"]
            train_ppl = float(math.exp(train_loss))
            val_ppl = float(math.exp(val_loss))
            elapsed = time.time() - t0

            rec = {
                "type": "eval",
                "time": now_iso(),
                "step": step,
                "lr": lr,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_ppl": train_ppl,
                "val_ppl": val_ppl,
                "total_tokens": total_tokens,
                "elapsed_s": elapsed,
                "mem": get_mem_stats(device),
            }
            append_jsonl(metrics_path, rec)

            print(f"\n[eval] step={step} lr={lr:.2e} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_ppl={val_ppl:.1f} elapsed={elapsed:.1f}s")

            # sample & save
            with torch.no_grad():
                model.eval()
                prompt_ids = enc.encode(args.sample_prompt)
                idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                out = model.generate(idx, max_new_tokens=args.sample_tokens,
                                     temperature=args.temperature, top_k=args.top_k)
                text = enc.decode(out[0].tolist())
                model.train()

            sample_path = samples_dir / f"step{step:06d}.txt"
            sample_path.write_text(text, encoding="utf-8")
            print("[sample]")
            print(text)

            # checkpoint latest
            save_checkpoint(ckpt_latest_path, model, optimizer, step, run_config, best_val)

            # checkpoint best
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(ckpt_best_path, model, optimizer, step, run_config, best_val)
                print(f"[ckpt] best updated: val_loss={best_val:.4f} -> {ckpt_best_path}")

            # periodic step checkpoint
            if args.save_step_ckpt_every > 0 and step % args.save_step_ckpt_every == 0 and step != 0:
                save_checkpoint(run_dir / f"ckpt_step{step:06d}.pt", model, optimizer, step, run_config, best_val)

            print(f"[ckpt] latest saved: {ckpt_latest_path}")

        # Train step with grad accumulation
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        step_start = time.time()
        for _ in range(args.grad_accum):
            xb, yb = get_batch(train_data, args.block_size, args.batch_size, device)
            _, loss = model(xb, yb)
            loss = loss / args.grad_accum
            loss.backward()
            loss_accum += loss.item()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        # Update counters
        total_tokens += tokens_per_step
        dt_step = time.time() - step_start
        toks_per_s = tokens_per_step / max(dt_step, 1e-8)

        if step % args.log_interval == 0:
            rec = {
                "type": "train",
                "time": now_iso(),
                "step": step,
                "lr": lr,
                "loss": float(loss_accum),
                "total_tokens": total_tokens,
                "tokens_per_s": float(toks_per_s),
                "elapsed_s": float(time.time() - t0),
            }
            append_jsonl(metrics_path, rec)
            print(f"[train] step={step} lr={lr:.2e} loss={loss_accum:.4f} tokens/s={toks_per_s:.0f}")

    print("\n[done] training finished.")


if __name__ == "__main__":
    main()
