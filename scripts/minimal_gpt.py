# minimal_gpt.py
# A minimal GPT that can: forward, generate, do one backward step.
# Dependency: pip install torch

## 导入依赖，只依赖 torch 和数学核心库
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Building blocks
# -----------------------------

## 因果自注意力机制
## 之所以叫做因果，因为 token 只能依赖过去和当前的信息，不能依赖未来信息，符合因果关系
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.block_size = block_size

        # One projection to get q, k, v (like GPT-2)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=True)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=True)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask (lower triangular)
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("bias", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape
        assert T <= self.block_size, f"Sequence length {T} > block_size {self.block_size}"

        qkv = self.c_attn(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)

        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Attention scores: (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # Apply causal mask: disallow attending to future positions
        mask = self.bias[:, :, :T, :T]  # (1,1,T,T)
        att = att.masked_fill(mask == 0, torch.finfo(att.dtype).min)

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # back to (B, T, C)

        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y

## 多层感知机（Multi_Layer Perceptron），也叫做前馈神经网络（Feed-Forward Network）
class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=True)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

## 一个完整的 Transformer 块，组合注意力层和 MLP 层
class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


## 完整的 GPT 模型，这是最顶层的类
class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int = 2,
        n_head: int = 2,
        n_embd: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.wte = nn.Embedding(vocab_size, n_embd)      # token embeddings
        self.wpe = nn.Embedding(block_size, n_embd)      # position embeddings

        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying (common in GPT)
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        # idx: (B, T)
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} > block_size {self.block_size}"

        pos = torch.arange(0, T, device=idx.device)  # (T,)
        tok_emb = self.wte(idx)                      # (B, T, C)
        pos_emb = self.wpe(pos)                      # (T, C) -> broadcast to (B, T, C)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)                     # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten for cross-entropy: (B*T, vocab)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]  # crop context if needed
            logits, _ = self(idx_cond)            # (B, T, vocab)
            logits = logits[:, -1, :] / max(temperature, 1e-8)  # (B, vocab)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                cutoff = v[:, [-1]]
                logits = logits.masked_fill(logits < cutoff, torch.finfo(logits.dtype).min)

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)   # (B, 1)
            idx = torch.cat([idx, next_id], dim=1)              # append
        return idx


# -----------------------------
# Tiny char-level tokenizer
# -----------------------------

def build_char_vocab(text: str):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def encode(s: str, stoi: dict) -> list[int]:
    return [stoi[c] for c in s]

def decode(ids: list[int], itos: dict) -> str:
    return "".join(itos[i] for i in ids)

def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    # Sample random subsequences
    ix = torch.randint(0, data.size(0) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


# -----------------------------
# Main: smoke tests + tiny train
# -----------------------------

def main():
    # 设置随机种子为 1337，让程序每次运行的随机结果都一样，方案调试和复现
    # 1337 是一个常见的彩蛋数字
    torch.manual_seed(1337)

    # 选择计算设备，优先使用 GPU（如果可用），否则使用 CPU
    device = torch.device("mps")
    print(f"[device] {device}")

    # A tiny corpus (keep vocab small; repeat to get enough training data)
    # 准备训练数据，数据量较小，所以最后 * 200 来扩充数据量
    corpus = (
        "最小GPT从零到一。\n"
        "我们先让模型学会预测下一个字符。\n"
        "注意：这是一个教学用的极小模型。\n"
        "从forward到generate再到backward，一次打通。\n"
    ) * 200

    # 构建字符词汇表
    # stoi: string to integer 是字符到数字的映射
    # 例如: {'最': 0, '小': 1, 'G': 2, ...}
    # itos: integer to string 是数字到字符的映射
    # 例如: {0: '最', 1: '小', 2: 'G', ...}
    # 这样就可以把文本转换成模型能处理的数字
    stoi, itos = build_char_vocab(corpus)
    # 计算词汇表大小
    vocab_size = len(stoi)
    print(f"[vocab] size={vocab_size}")

    # 将文本编码成数字张量
    data = torch.tensor(encode(corpus, stoi), dtype=torch.long)

    # Hyperparams: deliberately tiny so it runs anywhere
    # 设置模型超参数
    block_size = 64
    # 批次大小，一次处理多少样本，更大的 batch 更稳定但需要更多内存
    batch_size = 16
    # Transformer 层数
    n_layer = 2
    # 注意力头数量
    n_head = 2
    # 嵌入维度
    n_embd = 64
    # Dropout 概率
    # Dropout 是一种防止过拟合的技术
    dropout = 0.0

    # 创建模型
    model = GPT(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
    ).to(device)

    # 计算模型的总参数量
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params/1e6:.3f}M")

    # --------
    # 1) Forward test
    # --------
    # 设置模型为训练模式
    model.train()
    # 获取一个训练批次
    xb, yb = get_batch(data, block_size, batch_size, device)
    # 执行前向传播
    logits, loss = model(xb, yb)
    # 打印前向传播结果
    print(f"[forward] logits={tuple(logits.shape)} loss={loss.item():.4f}")

    # --------
    # 2) Causality test (mask correctness)
    #    Change the last token and see whether earlier-position logits change.
    # --------
    # 禁用梯度计算
    with torch.no_grad():
        # 取第一个样本的副本
        test = xb[:1, :].clone()
        # 第一次前向传播
        logits1, _ = model(test)
        # 创建 text 的另一个副本
        test2 = test.clone()
        # 修改最后一个 token
        # Flip the last token to a different id
        test2[0, -1] = (test2[0, -1] + 1) % vocab_size
        # 第二次前向传播
        logits2, _ = model(test2)

        # Compare logits for positions [0 .. T-2]
        diff = (logits1[:, :-1, :] - logits2[:, :-1, :]).abs().max().item()
        print(f"[causality] max_diff_on_past_positions={diff:.6f} (should be ~0)")

    # --------
    # 3) One backward step
    # --------
    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # 清空梯度
    optimizer.zero_grad(set_to_none=True)
    # 前向传播计算损失
    logits, loss = model(xb, yb)
    # 反向传播计算梯度
    loss.backward()

    # Grad norm (sanity)
    # 计算梯度的 L2 范数
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    print(f"[backward] grad_norm={total_norm:.4f}")

    # 更多参数
    optimizer.step()

    # Recompute loss on same batch just to show it runs end-to-end
    # 重新计算更新后的损失
    with torch.no_grad():
        _, loss2 = model(xb, yb)
    # 对比更新前后的损失
    print(f"[update] loss_before={loss.item():.4f} loss_after={loss2.item():.4f}")

    # --------
    # 4) Generate before/after a tiny bit of training
    # --------
    # 生成测试
    # 设置提示词
    prompt = "最小GPT"
    # 编码提示词
    idx = torch.tensor([encode(prompt, stoi)], dtype=torch.long, device=device)

    # 训练前生成文本
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=80, temperature=1.0, top_k=20)

    # 打印训练前生成的文本
    print("\n[generate before train]")
    print(decode(out[0].tolist(), itos))

    # Tiny training (so generation looks less random)
    # 开始训练
    # 训练 200 步
    steps = 200
    # 设置模型到训练模式
    model.train()
    # 训练循环
    for step in range(steps):
        # 每步随机采样一个新批次
        xb, yb = get_batch(data, block_size, batch_size, device)
        # 前向传播
        _, loss = model(xb, yb)
        # 清空梯度
        optimizer.zero_grad(set_to_none=True)
        # 反向传播
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # 更新参数
        optimizer.step()
        # 每 50 步打印一次损失
        if (step + 1) % 50 == 0:
            print(f"[train] step {step+1}/{steps} loss={loss.item():.4f}")

    # 训练后生成文本
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=120, temperature=0.9, top_k=30)

    # 打印训练后生成的文本
    print("\n[generate after train]")
    print(decode(out[0].tolist(), itos))

    # --------
    # 5) Save model checkpoint
    # --------
    # 保存模型检查点
    from pathlib import Path
    save_dir = Path("runs/minimal_gpt")
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / "model.pt"

    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "block_size": block_size,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "stoi": stoi,
        "itos": itos,
    }, checkpoint_path)
    print(f"\n[checkpoint] saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
