"""
Nova基座模型中文预训练脚本（GPT-2风格结构，nanoGPT式训练流程）

功能：
  - 数据加载：np.memmap从train.bin/val.bin随机采样block_size长度窗口
  - 设备自动选择 mps > cuda > cpu；默认fp32，--amp开autocast fp16
  - AdamW(β=0.9/0.95, weight_decay=0.1)，bias与LayerNorm不做weight decay
  - 学习率：线性warmup + 余弦衰减到min_lr
  - 梯度裁剪1.0、梯度累积
  - 断点续训：定期保存 ckpt.pt（model+optimizer+step+config），--resume恢复
  - 另存验证损失最优的 best.pt
  - 定期打印 step/train loss/lr/tokens_per_sec；每隔若干步跑val loss并生成一段中文样例

用法示例：
  python pretrain.py --preset gpt2-mini --batch-size 16 --grad-accum 8 --max-steps 5000
  python pretrain.py --preset gpt2-mini --resume         # 断点续训
"""

import argparse
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch

import os
import sys
# 使脚本可从任意目录直接运行（把项目根目录加入模块搜索路径）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nova.gpt2_model import GPT, GPTConfig, PRESETS
from nova.bpe_tokenizer import BPETokenizer


# ---------------------------------------------------------------------------
# 设备与数据
# ---------------------------------------------------------------------------
def pick_device(override=None):
    """自动选择设备：mps > cuda > cpu"""
    if override:
        return override
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class DataLoader:
    """从二进制token文件按窗口随机采样的极简数据加载器"""

    def __init__(self, data_dir, block_size, device, device_type):
        self.block_size = block_size
        self.device = device
        self.device_type = device_type
        train_path = os.path.join(data_dir, "train.bin")
        val_path = os.path.join(data_dir, "val.bin")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"缺少{train_path}，请先运行encode_corpus.py")
        # memmap只在需要时按页读盘，不会把整份数据载入内存
        self.train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
        self.val_data = (np.memmap(val_path, dtype=np.uint16, mode="r")
                         if os.path.exists(val_path) else None)

    def get_batch(self, split, batch_size):
        """随机采样一个batch的(x, y)，y是x右移一位的预测目标"""
        data = self.train_data if split == "train" else self.val_data
        if data is None:
            data = self.train_data
        # 随机起点，保证能取到block_size+1个token
        ix = torch.randint(len(data) - self.block_size - 1, (batch_size,))
        x = torch.stack([
            torch.from_numpy(data[i:i + self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([
            torch.from_numpy(data[i + 1:i + 1 + self.block_size].astype(np.int64)) for i in ix])
        if self.device_type == "cuda":
            # CUDA下固定内存 + 异步搬运，减少数据搬运对训练的阻塞
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            # MPS/CPU直接搬运（pin_memory是CUDA专属优化）
            x, y = x.to(self.device), y.to(self.device)
        return x, y


# ---------------------------------------------------------------------------
# 优化器与学习率
# ---------------------------------------------------------------------------
def configure_optimizer(model, weight_decay, lr, betas, device_type):
    """
    构造AdamW，并把参数分成两组：
      - 维度>=2的权重矩阵/嵌入 -> 施加weight decay
      - 维度<2的bias、LayerNorm -> 不做weight decay
    """
    params = [p for p in model.parameters() if p.requires_grad]
    decay_params = [p for p in params if p.dim() >= 2]
    nodecay_params = [p for p in params if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    n_decay = sum(p.numel() for p in decay_params)
    n_nodecay = sum(p.numel() for p in nodecay_params)
    print(f"优化器分组：decay张量{len(decay_params)}个({n_decay:,}参数)，"
          f"no-decay张量{len(nodecay_params)}个({n_nodecay:,}参数)")
    # CUDA上可用fused AdamW加速；MPS/CPU用普通实现
    use_fused = (device_type == "cuda")
    extra = dict(fused=True) if use_fused else dict()
    return torch.optim.AdamW(optim_groups, lr=lr, betas=betas, **extra)


def get_lr(it, warmup_steps, max_steps, lr, min_lr):
    """线性warmup + 余弦衰减的学习率调度"""
    if it < warmup_steps:
        return lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    # 余弦从lr衰减到min_lr
    ratio = (it - warmup_steps) / max(1, (max_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))  # 1 -> 0
    return min_lr + coeff * (lr - min_lr)


# ---------------------------------------------------------------------------
# 评估与采样
# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model, loader, batch_size, eval_iters, ctx):
    """在train/val上各跑eval_iters个batch估计平均损失"""
    out = {}
    model.eval()
    for split in ("train", "val"):
        if split == "val" and loader.val_data is None:
            continue
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = loader.get_batch(split, batch_size)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


@torch.no_grad()
def sample_text(model, tok, prompt, max_new_tokens, device, top_k=200,
                temperature=0.8):
    """用当前模型生成一段中文样例，返回字符串"""
    model.eval()
    ids = tok.encode(prompt)
    if not ids:
        ids = [tok.eot_id]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens, temperature=temperature,
                         top_k=top_k, eot_id=tok.eot_id)
    text = tok.decode(out[0].tolist())
    model.train()
    return text


# ---------------------------------------------------------------------------
# 训练主流程
# ---------------------------------------------------------------------------
def train(args):
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    device_type = "cuda" if device.startswith("cuda") else \
        ("mps" if device == "mps" else "cpu")
    print(f"使用设备：{device}")

    # 分词器：模型词表大小对齐分词器实际词表
    tok = BPETokenizer.load(args.tokenizer_dir)
    vocab_size = tok.vocab_size
    print(f"分词器词表大小：{vocab_size}，eot_id={tok.eot_id}")

    # 数据
    loader = DataLoader(args.data_dir, args.block_size, device, device_type)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    best_path = os.path.join(args.out_dir, "best.pt")

    # autocast上下文（--amp时启用fp16）
    if args.amp and device_type in ("cuda", "mps"):
        ctx = torch.autocast(device_type=device_type, dtype=torch.float16)
        print("已启用AMP（autocast fp16）")
    else:
        ctx = nullcontext()
    # GradScaler仅在CUDA+fp16下需要；MPS的fp16直接autocast即可
    scaler = torch.amp.GradScaler(enabled=(args.amp and device_type == "cuda"))

    # 构建模型（vocab_size用分词器实际值覆盖预设默认16384）
    preset = dict(PRESETS[args.preset])
    preset["vocab_size"] = vocab_size
    preset["block_size"] = args.block_size
    preset["dropout"] = args.dropout
    config = GPTConfig(**preset)

    start_step = 0
    best_val = float("inf")

    model = GPT(config)
    model.to(device)
    optimizer = configure_optimizer(
        model, args.weight_decay, args.lr, (args.beta1, args.beta2), device_type)

    # 断点续训
    if args.resume:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"--resume但找不到{ckpt_path}")
        print(f"从检查点恢复：{ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        # 用检查点里的config重建模型，保证结构一致
        config = GPTConfig(**ckpt["config"])
        model = GPT(config)
        model.to(device)
        model.load_state_dict(ckpt["model"])
        optimizer = configure_optimizer(
            model, args.weight_decay, args.lr, (args.beta1, args.beta2), device_type)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        best_val = ckpt.get("best_val", float("inf"))
        print(f"恢复到step={start_step}，best_val={best_val:.4f}")

    print(f"模型参数量：{model.num_params() / 1e6:.2f}M")
    print(f"每步token数：{args.batch_size * args.block_size * args.grad_accum:,} "
          f"(batch{args.batch_size} x block{args.block_size} x accum{args.grad_accum})")

    model.train()
    x, y = loader.get_batch("train", args.batch_size)  # 预取第一个batch
    t0 = time.time()
    tokens_per_iter = args.batch_size * args.block_size * args.grad_accum

    for step in range(start_step, args.max_steps):
        # 设置本步学习率
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr) \
            if args.decay_lr else args.lr
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # 梯度累积：累加grad_accum个micro batch再更新一次
        for micro in range(args.grad_accum):
            with ctx:
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            # 在反向传播的同时异步预取下一个batch
            x, y = loader.get_batch("train", args.batch_size)
            scaler.scale(loss).backward()

        # 梯度裁剪
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # 周期性日志
        if step % args.log_interval == 0:
            if device_type == "mps":
                torch.mps.synchronize()
            elif device_type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            # 本区间处理的token数 = 每步token数 * 区间步数
            steps_done = args.log_interval if step > start_step else 1
            tokens_per_sec = tokens_per_iter * steps_done / dt
            lossf = loss.item() * args.grad_accum
            print(f"step {step:>6} | loss {lossf:.4f} | lr {lr:.2e} | "
                  f"{dt / steps_done * 1000:.0f}ms/step | {tokens_per_sec:,.0f} tok/s")
            t0 = time.time()

        # 周期性评估 + 生成样例
        if step > 0 and step % args.eval_interval == 0:
            losses = estimate_loss(model, loader, args.batch_size, args.eval_iters, ctx)
            msg = f"[eval] step {step} | train {losses['train']:.4f}"
            if "val" in losses:
                msg += f" | val {losses['val']:.4f}"
            print(msg)

            sample = sample_text(model, tok, args.sample_prompt,
                                 args.sample_tokens, device)
            print(f"[sample] {sample!r}")

            # 保存验证损失最优模型
            cur_val = losses.get("val", losses["train"])
            if cur_val < best_val:
                best_val = cur_val
                _save_ckpt(best_path, model, optimizer, step, best_val, config)
                print(f"[best] 保存最优模型 val={best_val:.4f} -> {best_path}")
            t0 = time.time()

        # 周期性保存断点
        if step > 0 and step % args.ckpt_interval == 0:
            _save_ckpt(ckpt_path, model, optimizer, step, best_val, config)
            print(f"[ckpt] step {step} 已保存 -> {ckpt_path}")
            t0 = time.time()

    # 训练结束保存最终断点
    _save_ckpt(ckpt_path, model, optimizer, args.max_steps, best_val, config)
    print(f"训练结束，最终断点 -> {ckpt_path}")


def _save_ckpt(path, model, optimizer, step, best_val, config):
    """保存检查点：模型、优化器、step、config、best_val"""
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val": best_val,
        "config": vars(config),
    }, path)


def build_argparser():
    p = argparse.ArgumentParser(description="Nova基座模型中文预训练（GPT-2风格结构）")
    p.add_argument("--preset", default="gpt2-mini", choices=list(PRESETS.keys()))
    p.add_argument("--data-dir", default="data/pretrain")
    p.add_argument("--tokenizer-dir", default="models/bpe_tokenizer")
    p.add_argument("--out-dir", default="models/gpt2_pretrain")
    # 16GB内存建议micro batch从16x512起步，显存不够再调小
    p.add_argument("--batch-size", type=int, default=16, help="micro batch大小")
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--grad-accum", type=int, default=8, help="梯度累积步数")
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min-lr", type=float, default=6e-5)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--decay-lr", action="store_true", default=True)
    p.add_argument("--no-decay-lr", dest="decay_lr", action="store_false")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--eval-interval", type=int, default=250, help="每M步评估+采样")
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--ckpt-interval", type=int, default=500, help="每N步存断点")
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--amp", action="store_true", help="启用autocast fp16")
    p.add_argument("--resume", action="store_true", help="从ckpt.pt恢复")
    p.add_argument("--device", default=None, help="强制指定设备，如cpu/mps/cuda")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--sample-prompt", default="人工智能")
    p.add_argument("--sample-tokens", type=int, default=100)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
