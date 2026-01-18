#!/usr/bin/env python3
# scripts/summarize_run.py

from __future__ import annotations
import argparse
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True, help="e.g. runs/wt2_bpe_repro")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    metrics = run_dir / "metrics.jsonl"
    if not metrics.exists():
        raise FileNotFoundError(metrics)

    best = None
    last_eval = None
    n_train = 0
    n_eval = 0

    with open(metrics, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("type") == "train":
                n_train += 1
            elif rec.get("type") == "eval":
                n_eval += 1
                last_eval = rec
                val = rec["val_loss"]
                if (best is None) or (val < best["val_loss"]):
                    best = rec

    print(f"[run] {run_dir}")
    print(f"[counts] train_logs={n_train} eval_logs={n_eval}")
    if best:
        print(f"[best] step={best['step']} val_loss={best['val_loss']:.4f} val_ppl={best['val_ppl']:.2f}")
    if last_eval:
        print(f"[last] step={last_eval['step']} val_loss={last_eval['val_loss']:.4f} val_ppl={last_eval['val_ppl']:.2f}")

if __name__ == "__main__":
    main()
