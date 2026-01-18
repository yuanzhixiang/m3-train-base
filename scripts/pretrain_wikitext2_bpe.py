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
    # 创建命令行参数解析器，用于配置训练参数
    parser = argparse.ArgumentParser(description="Pretrain minimal GPT on WikiText-2 (raw) with GPT-2 BPE (tiktoken).")
    # 实验运行名称，用于保存检查点和日志
    parser.add_argument("--run_name", type=str, default="wt2_bpe")
    # 随机种子，用于可复现性
    parser.add_argument("--seed", type=int, default=1337)
    # 设备选择：cuda(NVIDIA GPU) / mps(Apple Silicon) / cpu，None表示自动选择
    parser.add_argument("--device", type=str, default=None, help="cuda|mps|cpu (default: auto)")

    # Model 模型架构参数
    # Transformer 层数（每层包含一个自注意力块和一个前馈网络）
    parser.add_argument("--n_layer", type=int, default=4)
    # 多头注意力的头数
    parser.add_argument("--n_head", type=int, default=4)
    # 嵌入维度（每个 token 的向量表示维度）
    parser.add_argument("--n_embd", type=int, default=256)
    # 上下文窗口大小（模型一次能处理的最大 token 数）
    parser.add_argument("--block_size", type=int, default=128)
    # Dropout 比例，用于防止过拟合
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training 训练参数
    # 微批次大小（每次前向传播处理的样本数，保持较小以节省显存，因为 50k 词汇表的 logits 很大）
    parser.add_argument("--batch_size", type=int, default=4, help="micro-batch size (keep small; logits is huge with 50k vocab)")
    # 梯度累积步数（模拟更大的批次：有效批次 = batch_size * grad_accum）
    parser.add_argument("--grad_accum", type=int, default=8, help="gradient accumulation steps")
    # 最大训练步数
    parser.add_argument("--max_steps", type=int, default=2000)
    # 基础学习率（峰值学习率）
    parser.add_argument("--lr", type=float, default=3e-4)
    # 最小学习率（余弦退火后的最终学习率）
    parser.add_argument("--min_lr", type=float, default=3e-5)
    # 学习率预热步数（从 0 线性增加到 base_lr）
    parser.add_argument("--warmup_steps", type=int, default=100)
    # 权重衰减系数（L2 正则化，AdamW 中的解耦权重衰减）
    parser.add_argument("--weight_decay", type=float, default=0.1)
    # 梯度裁剪阈值（防止梯度爆炸）
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Eval / logging 评估和日志记录
    # 训练日志打印间隔（每 N 步打印一次训练 loss）
    parser.add_argument("--log_interval", type=int, default=50)
    # 评估间隔（每 N 步在训练集和验证集上评估一次）
    parser.add_argument("--eval_interval", type=int, default=200)
    # 评估时的迭代次数（计算平均 loss 的批次数）
    parser.add_argument("--eval_iters", type=int, default=25)

    # Correctness seal 正确性验证
    # 如果 >0：在固定批次上过拟合 N 步（用于验证模型是否能正常学习）
    parser.add_argument("--overfit_one_batch_steps", type=int, default=0, help="if >0: overfit a fixed batch for N steps")

    # Data 数据相关
    # 数据集目录，None 表示使用默认路径
    parser.add_argument("--data_dir", type=str, default=None)
    # 是否强制重新 tokenize（否则会使用缓存）
    parser.add_argument("--force_rebuild_tokens", action="store_true")

    # Sampling 生成采样参数
    # 生成文本时的提示词（prompt）
    parser.add_argument("--sample_prompt", type=str, default="The history of")
    # 生成的 token 数量
    parser.add_argument("--sample_tokens", type=int, default=120)
    # 采样温度（越高越随机，越低越确定性）
    parser.add_argument("--temperature", type=float, default=0.9)
    # Top-K 采样（只从概率最高的 K 个 token 中采样）
    parser.add_argument("--top_k", type=int, default=40)

    # 解析命令行参数
    args = parser.parse_args()

    # 设置随机种子，确保实验可复现
    torch.manual_seed(args.seed)

    # 根据参数或自动检测选择计算设备（GPU/CPU）
    device = pick_device(args.device)
    print(f"[device] {device}")

    # Project root: repo_root/scripts/this_file.py -> repo_root
    # 获取项目根目录（当前文件在 scripts/ 下，父目录的父目录是项目根目录）
    repo_root = Path(__file__).resolve().parent.parent
    # 创建运行目录，用于保存检查点和日志
    run_dir = repo_root / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # 设置数据目录：如果指定了 data_dir 参数则使用它，否则使用默认路径
    data_dir = Path(args.data_dir) if args.data_dir else (repo_root / "data" / "wikitext2_raw")
    # tokenize 后的缓存目录
    cache_dir = repo_root / "data" / "cache_wt2_gpt2_bpe"

    # Tokenizer 初始化 GPT-2 的 BPE tokenizer
    enc = tiktoken.get_encoding("gpt2")
    # GPT-2 的词汇表大小应该是 50257
    vocab_size = enc.n_vocab  # should be 50257 for GPT-2
    print(f"[tokenizer] gpt2_bpe vocab_size={vocab_size}")

    # quick tokenizer sanity 快速验证 tokenizer 是否正常工作
    # 测试不同类型的字符串：转义字符、英文、中文、emoji
    test_strings = ["GPT\\", "Hello world!", "中文也可以", "emoji🙂 test"]
    for s in test_strings:
        # 编码：文本 -> token IDs
        ids = enc.encode(s)
        # 解码：token IDs -> 文本
        s2 = enc.decode(ids)
        # 验证往返转换是否一致
        print(f"[tok_test] {s!r} -> {len(ids)} tokens -> roundtrip_ok={s2 == s}")

    # Data 下载并准备数据
    # 确保 WikiText-2 数据集已下载（如果没有则自动下载）
    files = ensure_wikitext2_raw(data_dir)
    # 对文本进行 tokenize 并缓存为 .pt 文件（如果缓存存在则直接加载）
    tokens = prepare_tokens(enc, files, cache_dir, force_rebuild=args.force_rebuild_tokens)

    # 获取训练集和验证集的 token 序列
    train_data = tokens["train"]
    val_data = tokens["valid"]

    # 打印数据集统计信息
    print(f"[data] train_tokens={train_data.numel():,} val_tokens={val_data.numel():,}")

    # Baseline check 基线检查
    # 随机猜测的 loss 是 ln(vocab_size)，模型的 loss 应该低于这个值
    baseline = math.log(vocab_size)
    print(f"[baseline] ln(vocab_size)={baseline:.4f}")

    # Model 创建模型
    model = GPT(
        vocab_size=vocab_size,        # 词汇表大小
        block_size=args.block_size,   # 上下文长度
        n_layer=args.n_layer,         # Transformer 层数
        n_head=args.n_head,           # 注意力头数
        n_embd=args.n_embd,           # 嵌入维度
        dropout=args.dropout,         # Dropout 比例
    ).to(device)  # 将模型移到指定设备（GPU/CPU）

    # 计算模型参数总数
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params/1e6:.3f}M")

    # 配置优化器（AdamW，带权重衰减）
    optimizer = configure_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    # -----------------------------
    # Smoke tests: forward / causality / backward / param delta
    # 冒烟测试：验证模型的基本功能是否正常
    # -----------------------------
    # 设置模型为训练模式（启用 dropout 等）
    model.train()
    # 从训练集获取一个批次数据
    xb, yb = get_batch(train_data, args.block_size, args.batch_size, device)
    # 前向传播：计算 logits 和 loss
    logits, loss = model(xb, yb)
    print(f"[smoke.forward] logits={tuple(logits.shape)} loss={loss.item():.4f}")

    # causality test in eval mode (so dropout won't affect)
    # 因果性测试：验证模型是否满足因果约束（token 只依赖过去，不依赖未来）
    with torch.no_grad():  # 不计算梯度（推理模式）
        model.eval()  # 评估模式（关闭 dropout）
        # 取批次中的第一个样本
        test = xb[:1].clone()
        # 计算原始输入的 logits
        logits1, _ = model(test)
        # 修改最后一个 token
        test2 = test.clone()
        test2[0, -1] = (test2[0, -1] + 1) % vocab_size
        # 计算修改后的 logits
        logits2, _ = model(test2)
        # 比较前面位置的 logits 差异（应该为 0，因为它们不应该依赖最后一个 token）
        diff = (logits1[:, :-1, :] - logits2[:, :-1, :]).abs().max().item()
        print(f"[smoke.causality] max_diff_on_past_positions={diff:.6f} (should be ~0)")
        model.train()  # 恢复训练模式

    # 反向传播测试：验证梯度计算是否正常
    # 清空优化器中的梯度
    optimizer.zero_grad(set_to_none=True)
    # 前向传播计算 loss
    _, loss = model(xb, yb)
    # 反向传播计算梯度
    loss.backward()

    # 计算梯度的 L2 范数（用于检查梯度是否正常）
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5
    print(f"[smoke.backward] grad_norm={grad_norm:.4f}")

    # 参数更新测试：验证优化器是否正常更新参数
    # 记录更新前的参数
    before = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
    # 执行一步优化（更新参数）
    optimizer.step()
    # 记录更新后的参数
    after = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    # 计算参数变化的范数（应该 >0，说明参数确实被更新了）
    print(f"[smoke.update] param_delta_norm={(after - before).norm().item():.6f}")

    # -----------------------------
    # Optional: overfit one fixed batch (correctness seal)
    # 可选：在固定批次上过拟合（正确性验证）
    # -----------------------------
    # 如果设置了 overfit_one_batch_steps > 0，则执行过拟合测试
    if args.overfit_one_batch_steps > 0:
        print(f"\n[overfit-one-batch] steps={args.overfit_one_batch_steps}")
        model.train()
        # 获取一个固定的批次（不变的数据）
        xb_fix, yb_fix = get_batch(train_data, args.block_size, args.batch_size, device)
        # 重新初始化优化器
        optimizer = configure_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

        # 在固定批次上反复训练（如果模型能学习，loss 应该快速下降到接近 0）
        for i in range(args.overfit_one_batch_steps):
            # 前向传播
            _, l = model(xb_fix, yb_fix)
            # 清空梯度
            optimizer.zero_grad(set_to_none=True)
            # 反向传播
            l.backward()
            # 如果设置了梯度裁剪，则裁剪梯度
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 更新参数
            optimizer.step()
            # 每 50 步或第一步打印 loss
            if (i + 1) % 50 == 0 or i == 0:
                print(f"[overfit-one-batch] step {i+1} loss={l.item():.6f}")

    # -----------------------------
    # Train
    # 正式训练循环
    # -----------------------------
    # 如果 max_steps <= 0，则跳过训练，只执行冒烟测试后退出
    if args.max_steps <= 0:
        print("\n[done] max_steps<=0, exit after smoke tests.")
        return

    print(f"\n[train] max_steps={args.max_steps} batch_size={args.batch_size} grad_accum={args.grad_accum} block_size={args.block_size}")
    model.train()
    # 重新初始化优化器（如果之前进行了 overfit 测试，需要重置优化器状态）
    optimizer = configure_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    # 记录训练开始时间
    t0 = time.time()
    # 主训练循环
    for step in range(args.max_steps):
        # LR schedule 学习率调度
        # 根据当前步数计算学习率（warmup + cosine decay）
        lr = get_lr(
            step,
            base_lr=args.lr,
            min_lr=args.min_lr,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
        )
        # 更新优化器中所有参数组的学习率
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Eval + checkpoint + sample 评估 + 保存检查点 + 生成样本
        # 每隔 eval_interval 步或最后一步执行评估
        if step % args.eval_interval == 0 or step == args.max_steps - 1:
            # 在训练集和验证集上评估模型
            losses = estimate_loss(
                model,
                train_data=train_data,
                val_data=val_data,
                block_size=args.block_size,
                batch_size=args.batch_size,
                device=device,
                eval_iters=args.eval_iters,
            )
            # 计算已训练时间
            elapsed = time.time() - t0
            print(f"\n[eval] step={step} lr={lr:.2e} train_loss={losses['train']:.4f} val_loss={losses['val']:.4f} elapsed={elapsed:.1f}s")

            # sample 生成文本样本
            with torch.no_grad():  # 不计算梯度
                model.eval()  # 评估模式
                # 将提示词编码为 token IDs
                prompt_ids = enc.encode(args.sample_prompt)
                # 转换为 tensor 并移到设备上
                idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                # 生成文本
                out = model.generate(idx, max_new_tokens=args.sample_tokens, temperature=args.temperature, top_k=args.top_k)
                # 解码生成的 token IDs 为文本
                text = enc.decode(out[0].tolist())
                model.train()  # 恢复训练模式
            print("[sample]")
            print(text)

            # save 保存检查点
            ckpt_path = run_dir / "ckpt_latest.pt"
            # 构造配置字典（包含所有超参数）
            config = vars(args).copy()
            config.update({
                "vocab_size": vocab_size,
                "encoding": "gpt2",
            })
            # 保存最新检查点
            save_checkpoint(ckpt_path, model, optimizer, step=step, config=config)
            # also save a step checkpoint occasionally 偶尔保存特定步数的检查点
            if step % (args.eval_interval * 5) == 0:
                save_checkpoint(run_dir / f"ckpt_step{step:06d}.pt", model, optimizer, step=step, config=config)
            print(f"[ckpt] saved: {ckpt_path}")

        # gradient accumulation 梯度累积
        # 清空梯度
        optimizer.zero_grad(set_to_none=True)
        # 累积的 loss
        loss_accum = 0.0

        # 执行多个微批次的梯度累积（模拟更大的批次）
        for micro in range(args.grad_accum):
            # 获取一个微批次
            xb, yb = get_batch(train_data, args.block_size, args.batch_size, device)
            # 前向传播
            _, loss = model(xb, yb)
            # 将 loss 除以累积步数（相当于求平均）
            loss = loss / args.grad_accum
            # 反向传播（梯度会累积）
            loss.backward()
            # 累加 loss 用于日志
            loss_accum += loss.item()

        # 如果设置了梯度裁剪，则裁剪梯度（防止梯度爆炸）
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # 执行一步优化（更新所有参数）
        optimizer.step()

        # 每隔 log_interval 步打印训练日志
        if step % args.log_interval == 0:
            print(f"[train] step={step} lr={lr:.2e} loss={loss_accum:.4f}")

    print("\n[done] training finished.")


if __name__ == "__main__":
    main()
