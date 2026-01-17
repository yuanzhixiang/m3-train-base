# minimal_gpt.py
# A minimal GPT that can: forward, generate, do one backward step.
# Dependency: pip install torch

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Building blocks
# -----------------------------

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
    torch.manual_seed(1337)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # A tiny corpus (keep vocab small; repeat to get enough training data)
    corpus = (
        "最小GPT从零到一。\n"
        "我们先让模型学会预测下一个字符。\n"
        "注意：这是一个教学用的极小模型。\n"
        "从forward到generate再到backward，一次打通。\n"
    ) * 200

    stoi, itos = build_char_vocab(corpus)
    vocab_size = len(stoi)
    print(f"[vocab] size={vocab_size}")

    data = torch.tensor(encode(corpus, stoi), dtype=torch.long)

    # Hyperparams: deliberately tiny so it runs anywhere
    block_size = 64
    batch_size = 16
    n_layer = 2
    n_head = 2
    n_embd = 64
    dropout = 0.0

    model = GPT(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params/1e6:.3f}M")

    # --------
    # 1) Forward test
    # --------
    model.train()
    xb, yb = get_batch(data, block_size, batch_size, device)
    logits, loss = model(xb, yb)
    print(f"[forward] logits={tuple(logits.shape)} loss={loss.item():.4f}")

    # --------
    # 2) Causality test (mask correctness)
    #    Change the last token and see whether earlier-position logits change.
    # --------
    with torch.no_grad():
        test = xb[:1, :].clone()
        logits1, _ = model(test)
        test2 = test.clone()
        # Flip the last token to a different id
        test2[0, -1] = (test2[0, -1] + 1) % vocab_size
        logits2, _ = model(test2)

        # Compare logits for positions [0 .. T-2]
        diff = (logits1[:, :-1, :] - logits2[:, :-1, :]).abs().max().item()
        print(f"[causality] max_diff_on_past_positions={diff:.6f} (should be ~0)")

    # --------
    # 3) One backward step
    # --------
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    optimizer.zero_grad(set_to_none=True)
    logits, loss = model(xb, yb)
    loss.backward()

    # Grad norm (sanity)
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    print(f"[backward] grad_norm={total_norm:.4f}")

    optimizer.step()

    # Recompute loss on same batch just to show it runs end-to-end
    with torch.no_grad():
        _, loss2 = model(xb, yb)
    print(f"[update] loss_before={loss.item():.4f} loss_after={loss2.item():.4f}")

    # --------
    # 4) Generate before/after a tiny bit of training
    # --------
    prompt = "最小GPT"
    idx = torch.tensor([encode(prompt, stoi)], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=80, temperature=1.0, top_k=20)
    print("\n[generate before train]")
    print(decode(out[0].tolist(), itos))

    # Tiny training (so generation looks less random)
    steps = 200
    model.train()
    for step in range(steps):
        xb, yb = get_batch(data, block_size, batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 50 == 0:
            print(f"[train] step {step+1}/{steps} loss={loss.item():.4f}")

    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=120, temperature=0.9, top_k=30)
    print("\n[generate after train]")
    print(decode(out[0].tolist(), itos))


if __name__ == "__main__":
    main()
