"""
GPT-2续写生成脚本（演示/验收用）

加载预训练检查点 + BPE分词器，从命令行prompt生成中文续写。

用法：
  python generate_gpt2.py --ckpt models/gpt2_pretrain/best.pt --prompt "人工智能"
  python generate_gpt2.py --prompt "深度学习" --num-samples 3 --temperature 0.8 --top-k 200
  # 交互模式：不带--prompt时进入循环，逐行输入
  python generate_gpt2.py --ckpt models/gpt2_pretrain/best.pt
"""

import argparse
import os

import torch

from gpt2_model import GPT, GPTConfig
from bpe_tokenizer import BPETokenizer


def pick_device(override=None):
    """自动选择设备：mps > cuda > cpu"""
    if override:
        return override
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(ckpt_path, device):
    """从检查点重建模型并加载权重"""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到检查点：{ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    step = ckpt.get("step", "?")
    print(f"已加载模型：{ckpt_path}（step={step}，参数量"
          f"{model.num_params() / 1e6:.2f}M，vocab={config.vocab_size}）")
    return model


def generate_once(model, tok, prompt, device, max_new_tokens=200,
                  temperature=0.8, top_k=200, top_p=None):
    """对单个prompt生成一段续写文本"""
    ids = tok.encode(prompt)
    if not ids:
        ids = [tok.eot_id]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens, temperature=temperature,
                         top_k=top_k, top_p=top_p, eot_id=tok.eot_id)
    text = tok.decode(out[0].tolist())
    return text


def main():
    parser = argparse.ArgumentParser(description="GPT-2中文续写生成")
    parser.add_argument("--ckpt", default="models/gpt2_pretrain/best.pt")
    parser.add_argument("--tokenizer-dir", default="models/bpe_tokenizer")
    parser.add_argument("--prompt", default=None, help="不提供则进入交互模式")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f"使用设备：{device}")

    tok = BPETokenizer.load(args.tokenizer_dir)
    model = load_model(args.ckpt, device)

    def run(prompt):
        for i in range(args.num_samples):
            text = generate_once(
                model, tok, prompt, device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k, top_p=args.top_p)
            print(f"\n=== 样例 {i + 1}/{args.num_samples} ===")
            print(text)

    if args.prompt is not None:
        run(args.prompt)
    else:
        # 交互模式：逐行读入prompt，空行或quit退出
        print("进入交互模式，输入prompt回车生成（quit/exit退出）")
        while True:
            try:
                prompt = input("\nprompt> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见")
                break
            if prompt.lower() in ("quit", "exit"):
                break
            if not prompt:
                continue
            run(prompt)


if __name__ == "__main__":
    main()
