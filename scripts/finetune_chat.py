"""
对话微调脚本（v0.4）

在预训练好的Nova基座模型（models/gpt2_pretrain/best.pt）上用对话语料微调，
让"续写"模型学会"问答"格式：

    问：你好
    答：你好呀！很高兴见到你。<|endoftext|>

要点：
1. 损失掩码：只在"答"部分计算损失（提问token的target置-1被忽略），
   梯度集中于"看到问题→给出回答"的映射
2. 语料回放（replay）：每个batch混入一定比例的预训练语料随机窗口，
   缓解小数据微调导致的灾难性遗忘（丢失中文流畅度）
3. 数据增强：问题的无标点/小写变体也训练，容忍随意的输入习惯

用法：
    python3 finetune_chat.py                       # 默认参数即可
    python3 finetune_chat.py --steps 800 --lr 5e-5 # 自定义
"""

import argparse
import math
import os
import random
import time

import numpy as np
import torch

import os
import sys
# 使脚本可从任意目录直接运行（把项目根目录加入模块搜索路径）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nova.bpe_tokenizer import BPETokenizer
from nova.gpt2_model import GPT, GPTConfig

PROMPT_PREFIX = "问："
ANSWER_PREFIX = "\n答："


def pick_device(override=None):
    if override:
        return override
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_conversation_pairs(filepath="data/conversation_data.txt"):
    """解析对话数据：空行分隔的块，每块第一行是问题，其余行是回答"""
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    for block in content.split("\n\n"):
        lines = [ln.strip() for ln in block.strip().split("\n") if ln.strip()]
        if len(lines) >= 2:
            pairs.append((lines[0], " ".join(lines[1:])))
    return pairs


def augment_pairs(pairs):
    """数据增强：问题去尾部标点/小写的变体（回答不变）"""
    seen = {q for q, _ in pairs}
    out = list(pairs)
    for q, a in pairs:
        stripped = q.rstrip(" ?!.？！。")
        for variant in (stripped, q.lower(), stripped.lower()):
            if variant and variant not in seen:
                seen.add(variant)
                out.append((variant, a))
    return out


def load_knowledge_pairs(filepath, limit=200):
    """加载挖掘的知识TSV（实体\t定义句），为每个实体生成提问变体

    limit：只取前limit个实体。知识样本的作用是教会模型
    "实体→定义句"的回答格式，不是逼模型背事实——事实类回答
    由推理端的知识库检索保证（chat_inference.py），几千个实体
    在几百步微调里每条只能被看到零点几次，背也背不下来。
    """
    pairs = []
    if not os.path.exists(filepath):
        return pairs
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if len(pairs) >= limit * 3:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            entity, ans = parts
            # 三种问法都训练：裸实体名 / 介绍一下X / X是什么
            pairs.append((entity, ans))
            pairs.append((f"介绍一下{entity}", ans))
            pairs.append((f"{entity}是什么？", ans))
    return pairs


def build_examples(pairs, tok, max_len):
    """问答对 -> (input_ids, target_ids)，问题部分target=-1不计损失"""
    examples = []
    for q, a in pairs:
        prompt_ids = tok.encode(PROMPT_PREFIX + q + ANSWER_PREFIX)
        answer_ids = tok.encode(a) + [tok.eot_id]
        ids = (prompt_ids + answer_ids)[:max_len]
        if len(ids) <= len(prompt_ids):  # 回答被完全截断，丢弃
            continue
        inp = ids[:-1]
        tgt = ids[1:]
        # 问题部分不计损失：预测第i+1个token的位置是i，
        # 前len(prompt_ids)-1个位置预测的都还是问题token
        mask_len = min(len(prompt_ids) - 1, len(tgt))
        tgt = [-1] * mask_len + tgt[mask_len:]
        examples.append((inp, tgt))
    return examples


def batchify(samples, device):
    """按batch内最长序列padding；input补0（任意id均可），target补-1"""
    max_len = max(len(inp) for inp, _ in samples)
    x = torch.zeros(len(samples), max_len, dtype=torch.long)
    y = torch.full((len(samples), max_len), -1, dtype=torch.long)
    for i, (inp, tgt) in enumerate(samples):
        x[i, :len(inp)] = torch.tensor(inp, dtype=torch.long)
        y[i, :len(tgt)] = torch.tensor(tgt, dtype=torch.long)
    return x.to(device), y.to(device)


def replay_sample(data, window, count):
    """从预训练语料memmap中随机取count个窗口，全序列计损失"""
    samples = []
    for _ in range(count):
        i = random.randint(0, len(data) - window - 2)
        chunk = data[i:i + window + 1].astype(np.int64)
        samples.append((chunk[:-1].tolist(), chunk[1:].tolist()))
    return samples


@torch.no_grad()
def sample_replies(model, tok, prompts, device, max_new_tokens=60):
    """用当前模型对几个问题生成回答（观察训练进度用）"""
    model.eval()
    outs = []
    for q in prompts:
        ids = tok.encode(PROMPT_PREFIX + q + ANSWER_PREFIX)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        y = model.generate(x, max_new_tokens, temperature=0.3, top_k=20,
                           eot_id=tok.eot_id)
        reply = tok.decode(y[0, len(ids):].tolist()).strip()
        outs.append((q, reply))
    model.train()
    return outs


def main():
    ap = argparse.ArgumentParser(description="Nova对话模型微调")
    ap.add_argument("--pretrained", default="models/gpt2_pretrain/best.pt")
    ap.add_argument("--tokenizer-dir", default="models/bpe_tokenizer")
    ap.add_argument("--data", default="data/conversation_data.txt")
    ap.add_argument("--knowledge", default="data/knowledge_qa.tsv",
                    help="挖掘的实体定义TSV，存在时混入训练")
    ap.add_argument("--replay-bin", default="data/pretrain/train.bin",
                    help="预训练语料，存在时按--replay-frac比例混入")
    ap.add_argument("--out-dir", default="models/gpt2_chat")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--knowledge-limit", type=int, default=200,
                    help="参与训练的知识实体数（只教格式，事实靠检索）")
    ap.add_argument("--knowledge-frac", type=float, default=0.25,
                    help="每batch中知识问答样本的占比")
    ap.add_argument("--replay-frac", type=float, default=0.25)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-interval", type=int, default=20)
    ap.add_argument("--sample-interval", type=int, default=150)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f"使用设备：{device}")

    tok = BPETokenizer.load(args.tokenizer_dir)

    # 加载预训练模型
    ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.train()
    print(f"已加载预训练模型：{args.pretrained}"
          f"（step={ckpt.get('step', '?')}，参数量{model.num_params()/1e6:.2f}M）")

    # 准备对话样本
    pairs = load_conversation_pairs(args.data)
    augmented = augment_pairs(pairs)
    examples = build_examples(augmented, tok, args.max_len)
    print(f"对话数据：{len(pairs)}组原始问答，增强后{len(augmented)}组，"
          f"有效样本{len(examples)}条")

    # 挖掘的知识问答（实体→维基定义句）
    know_pairs = load_knowledge_pairs(args.knowledge, limit=args.knowledge_limit)
    know_examples = build_examples(know_pairs, tok, args.max_len)
    if know_examples:
        print(f"知识问答：{len(know_pairs)}条（含提问变体），"
              f"有效样本{len(know_examples)}条")

    # 预训练语料回放
    replay_data = None
    if args.replay_frac > 0 and os.path.exists(args.replay_bin):
        replay_data = np.memmap(args.replay_bin, dtype=np.uint16, mode="r")
        print(f"语料回放：每batch混入{args.replay_frac:.0%}预训练文本窗口")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95), weight_decay=0.01)

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        t = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * t))

    os.makedirs(args.out_dir, exist_ok=True)
    watch_prompts = ["你好", "你叫什么名字？", "北京大学", "介绍一下长城"]

    # 每batch的固定配比：回放 / 知识问答 / 对话
    n_replay = int(args.batch_size * args.replay_frac) if replay_data is not None else 0
    n_know = int(args.batch_size * args.knowledge_frac) if know_examples else 0
    n_chat = args.batch_size - n_replay - n_know
    print(f"batch配比：对话{n_chat} + 知识{n_know} + 回放{n_replay}")
    t0 = time.time()

    for step in range(args.steps):
        lr = lr_at(step)
        for g in optimizer.param_groups:
            g["lr"] = lr

        batch = random.sample(examples, min(n_chat, len(examples)))
        if n_know:
            batch += random.sample(know_examples, min(n_know, len(know_examples)))
        if n_replay:
            batch += replay_sample(replay_data, args.max_len - 1, n_replay)
        x, y = batchify(batch, device)

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_interval == 0 or step == args.steps - 1:
            dt = (time.time() - t0) / (step + 1)
            print(f"step {step:4d} | loss {loss.item():.4f} | lr {lr:.2e} "
                  f"| {dt*1000:.0f}ms/step")

        if (step + 1) % args.sample_interval == 0 or step == args.steps - 1:
            print("---- 生成样例 ----")
            for q, r in sample_replies(model, tok, watch_prompts, device):
                print(f"  问：{q}  答：{r}")
            print("-----------------")

    out_path = os.path.join(args.out_dir, "chat.pt")
    torch.save({
        "model": model.state_dict(),
        "config": ckpt["config"],
        "step": args.steps,
        "base": args.pretrained,
        "format": {"prompt_prefix": PROMPT_PREFIX, "answer_prefix": ANSWER_PREFIX},
    }, out_path)
    print(f"✅ 微调完成，模型已保存到：{out_path}")


if __name__ == "__main__":
    main()
