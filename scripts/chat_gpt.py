#!/usr/bin/env python3
"""
交互式 chat 工具，用于与训练好的 minimal GPT 模型对话
"""

import argparse
import sys
from pathlib import Path

import torch

# 导入 minimal_gpt 模块
from minimal_gpt import GPT, build_char_vocab, encode, decode


def load_model_and_vocab(checkpoint_path: str, device: torch.device):
    """加载模型和词汇表"""
    ckpt = torch.load(checkpoint_path, map_location=device)

    # 重建词汇表
    stoi = ckpt["stoi"]
    itos = ckpt["itos"]
    vocab_size = len(stoi)

    # 重建模型
    model = GPT(
        vocab_size=vocab_size,
        block_size=ckpt["block_size"],
        n_layer=ckpt["n_layer"],
        n_head=ckpt["n_head"],
        n_embd=ckpt["n_embd"],
        dropout=0.0,  # 推理时不使用 dropout
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, stoi, itos


def chat_loop(model: GPT, stoi: dict, itos: dict, device: torch.device,
              max_new_tokens: int = 150, temperature: float = 0.8, top_k: int = 40):
    """交互式对话循环"""
    print("=" * 60)
    print("GPT Chat - 输入 'quit' 或 'exit' 退出")
    print("=" * 60)
    print()

    context = ""  # 保存对话上下文

    while True:
        # 获取用户输入
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n再见！")
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "退出"]:
            print("再见！")
            break

        # 如果是特殊命令
        if user_input.startswith("/"):
            if user_input == "/clear":
                context = ""
                print("上下文已清空")
                continue
            elif user_input == "/context":
                print(f"当前上下文: {repr(context)}")
                continue
            elif user_input.startswith("/temp "):
                try:
                    temperature = float(user_input.split()[1])
                    print(f"Temperature 设置为: {temperature}")
                except (ValueError, IndexError):
                    print("用法: /temp <float>")
                continue
            elif user_input.startswith("/topk "):
                try:
                    top_k = int(user_input.split()[1])
                    print(f"Top-k 设置为: {top_k}")
                except (ValueError, IndexError):
                    print("用法: /topk <int>")
                continue
            else:
                print("未知命令。可用命令: /clear, /context, /temp <value>, /topk <value>")
                continue

        # 构建输入（保留一定上下文）
        prompt = context + user_input

        # 确保 prompt 中的所有字符都在词汇表中
        try:
            prompt_ids = encode(prompt, stoi)
        except KeyError as e:
            print(f"错误: 输入包含未知字符 {e}")
            continue

        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        # 生成回复
        try:
            with torch.no_grad():
                out = model.generate(
                    idx,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k
                )

            response = decode(out[0].tolist(), itos)

            # 只显示新生成的部分
            generated_text = response[len(prompt):]
            print(f"GPT: {generated_text}")
            print()

            # 更新上下文（保留最后一部分）
            context = response[-model.block_size//2:] if len(response) > model.block_size//2 else response

        except Exception as e:
            print(f"生成时出错: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="与训练好的 minimal GPT 模型聊天")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/minimal_gpt/model.pt",
        help="模型 checkpoint 路径"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=150,
        help="最大生成 token 数"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="采样温度（越高越随机）"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Top-k 采样"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备: cuda, mps, cpu (默认自动选择)"
    )

    args = parser.parse_args()

    # 检查 checkpoint 是否存在
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"错误: checkpoint 文件不存在: {ckpt_path}")
        print("请先运行 minimal_gpt.py 训练模型并保存 checkpoint")
        sys.exit(1)

    # 选择设备
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    print(f"使用设备: {device}")
    print(f"加载模型: {ckpt_path}")

    # 加载模型
    try:
        model, stoi, itos = load_model_and_vocab(str(ckpt_path), device)
        print(f"模型加载成功！词汇表大小: {len(stoi)}")
        print()
    except Exception as e:
        print(f"加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 进入对话循环
    chat_loop(
        model,
        stoi,
        itos,
        device,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k
    )


if __name__ == "__main__":
    main()
