# 命令行参数解析
import argparse
# JSON 序列化，用于保存日志
import json
# 操作系统接口
import os
# 时间测量
import time
# 路径处理
from pathlib import Path

# PyTorch 深度学习框架
import torch
# 数据加载工具
from torch.utils.data import DataLoader, Dataset
# Hugging Face 模型和分词器
from transformers import AutoModelForCausalLM, AutoTokenizer

# 继承 PyTorch 的 Dataset 类，用于创建自定义数据集
class TinyTextDataset(Dataset):
    def __init__(self, tokenizer, texts, seq_len: int):
        # TODO 这个 examples 的数据格式是什么是由谁决定的？
        self.examples = []
        for t in texts:
            # 对每条文本进行分词
            enc = tokenizer(
                t,
                # 超过长度截断
                truncation=True,
                # 不足长度填充
                padding="max_length",
                # 最大序列长度
                max_length=seq_len,
                # 返回 PyTorch 张量
                return_tensors="pt",
            )
            # token ID，去掉 batch 维度
            input_ids = enc["input_ids"].squeeze(0)
            # 注意力掩码（1=有效，0=填充）
            attention_mask = enc["attention_mask"].squeeze(0)
            # Causal LM: labels = input_ids（标准做法）
            # 语言模型的目标是预测下一个 token，所以标签就是输入本身
            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    # 复制一份作为标签
                    "labels": input_ids.clone(),
                }
            )

    def __len__(self):
        # 返回数据集大小
        return len(self.examples)

    def __getitem__(self, i):
        # 返回第 i 个样本
        return self.examples[i]


def pick_device():
    if torch.backends.mps.is_available():
        # 优先用 Apple MPS（GPU 加速）
        return torch.device("mps")
    # 否则用 CPU
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    # 训练步数
    parser.add_argument("--steps", type=int, default=120)
    # 批大小
    parser.add_argument("--batch_size", type=int, default=4)
    # 序列长度
    parser.add_argument("--seq_len", type=int, default=128)
    # 学习率
    parser.add_argument("--lr", type=float, default=5e-4)
    # 输出目录
    parser.add_argument("--out_dir", type=str, default="runs/m0_smoke")
    # 每多少步记录一次日志
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    # 创建输出目录（如不存在）
    out_dir.mkdir(parents=True, exist_ok=True)
    # 日志文件路径
    log_path = out_dir / "log.jsonl"

    device = pick_device()
    print("Using device:", device)

    # 使用一个很小的 GPT-2 模型，便于快速验证
    model_name = "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # GPT-2 分词器默认没有 pad_token
    if tokenizer.pad_token is None:
        # 用 eos_token 作为 pad_token
        tokenizer.pad_token = tokenizer.eos_token

    # 从 HF 下载模型
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # 将模型移到 MPS/CPU
    model.to(device)
    # 设置为训练模式
    model.train()

    # 小型数据集（重复以获得足够多的批次）
    base_texts = [
        "Hello from M3 Max. This is a smoke test for training on MPS.",
        "We want to verify forward, backward, optimizer step, and checkpoint saving.",
        "Loss should decrease a bit, but we don't care about final quality here.",
        "If MPS is missing ops, fallback to CPU may happen when enabled.",
    ]
    # 扩大到 1024 条样本
    texts = base_texts * 256
    ds = TinyTextDataset(tokenizer, texts, seq_len=args.seq_len)

    # 自定义 collate 函数：将多个样本堆叠成一个批次
    def collate(batch):
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            "labels": torch.stack([x["labels"] for x in batch]),
        }

    # 创建 DataLoader
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    # AdamW 优化器
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 创建迭代器
    # 简单循环：无限迭代 dataloader
    it = iter(dl)

    # 记录开始时间
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        # 获取下一个批次
        try:
            batch = next(it)
        except StopIteration:
            # 如果 dataloader 耗尽，重新创建迭代器
            it = iter(dl)
            batch = next(it)

        # 将批次数据移到设备上
        batch = {k: v.to(device) for k, v in batch.items()}

        # MPS 同步（可选）：确保计时更准确
        if device.type == "mps":
            torch.mps.synchronize()

        start = time.perf_counter()
        # 前向传播
        out = model(**batch)
        loss = out.loss

        # 反向传播
        loss.backward()

        # 优化器更新参数
        opt.step()
        # 清零梯度（set_to_none=True 更高效）
        opt.zero_grad(set_to_none=True)

        # MPS 同步
        if device.type == "mps":
            torch.mps.synchronize()

        end = time.perf_counter()

        # 每隔 log_every 步记录一次
        if step % args.log_every == 0 or step == 1:
            # 本步处理的 token 数
            tokens = args.batch_size * args.seq_len
            dt = end - start
            # 每秒处理的 token 数
            toks_per_s = tokens / max(dt, 1e-9)

            rec = {
                "step": step,
                "loss": float(loss.detach().cpu().item()),
                "tokens_per_s": toks_per_s,
                "device": str(device),
            }
            print(rec)
            # 追加写入 JSONL 日志
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    t1 = time.perf_counter()
    print(f"Done. Total seconds: {t1 - t0:.2f}")

    # 保存模型检查点
    ckpt_dir = out_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir, safe_serialization=True)
    tokenizer.save_pretrained(ckpt_dir)
    print("Saved checkpoint to:", ckpt_dir)

    # 验证 checkpoint 可以正常加载
    model2 = AutoModelForCausalLM.from_pretrained(ckpt_dir).to(device)
    model2.eval()

    # 不计算梯度
    with torch.no_grad():
        test = tokenizer("Reload test.", return_tensors="pt", padding=True)
        test = {k: v.to(device) for k, v in test.items()}
        out2 = model2(**test)
    print("Reload forward ok. logits shape:", tuple(out2.logits.shape))


if __name__ == "__main__":
    # 设置环境变量，减少分词器警告
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
