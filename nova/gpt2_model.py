"""
GPT-2结构复刻模块（缩小版）

忠实复刻GPT-2 small的核心结构，只是把层数/宽度缩小到10M~50M参数，便于在
单张M1上做中文预训练教学：
  - Pre-LN Transformer block（LayerNorm在子层之前）
  - 可学习的位置编码（wpe），而非正弦位置编码
  - GELU激活
  - 因果自注意力，使用F.scaled_dot_product_attention(is_causal=True)
  - 词嵌入(wte)与输出层(lm_head)权重共享
  - LayerNorm带bias（对齐GPT-2）
  - 初始化normal(0, 0.02)，残差投影按1/sqrt(2*n_layer)缩放
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """GPT模型的结构超参数"""
    block_size: int = 512      # 最大上下文长度（位置编码长度）
    vocab_size: int = 16384    # 词表大小（对齐BPE tokenizer默认值）
    n_layer: int = 6           # Transformer block层数
    n_head: int = 6            # 注意力头数
    n_embd: int = 384          # 隐藏维度
    dropout: float = 0.0       # dropout概率（预训练通常设0）
    bias: bool = True          # 线性层与LayerNorm是否带bias（GPT-2带bias）


class LayerNorm(nn.Module):
    """带可选bias的LayerNorm（PyTorch的nn.LayerNorm不支持bias=False，这里自己实现以对齐GPT-2）"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """因果多头自注意力"""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd必须能被n_head整除"
        # 一次性算出Q/K/V三个投影（合并成一个大矩阵，效率更高）
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 输出投影（属于残差路径，初始化时会做缩放）
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout_p = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()  # batch、序列长度、隐藏维度

        # 计算Q/K/V并拆分成多头：(B, n_head, T, head_dim)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # 使用PyTorch内置的缩放点积注意力，is_causal=True自动加因果掩码
        # 训练时启用注意力dropout，推理时(eval)自动关闭
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=True,
        )

        # 合并多头：(B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 输出投影 + 残差dropout
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """position-wise前馈网络（4倍升维 + GELU + 降维）"""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        # 降维投影同样属于残差路径
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Pre-LN Transformer block：先LayerNorm再进子层，子层输出以残差加回"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        # Pre-LN：x + Attn(LN(x))，x + MLP(LN(x))
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT-2风格的因果语言模型"""

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # 主干：词嵌入wte、位置嵌入wpe、若干Transformer block、末端LayerNorm
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        # 语言模型输出头（不带bias，GPT-2的lm_head即为共享的词嵌入）
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重共享：词嵌入与输出层共用同一份权重
        self.transformer.wte.weight = self.lm_head.weight

        # 初始化所有权重
        self.apply(self._init_weights)
        # 对残差路径上的投影(c_proj)做特殊缩放：std = 0.02 / sqrt(2 * n_layer)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        """GPT-2初始化：Linear/Embedding权重normal(0, 0.02)，bias置0，LayerNorm权重置1"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        前向传播。
        idx:     (B, T) 输入token id
        targets: (B, T) 训练目标；不为None时同时返回交叉熵损失
        返回:     (logits, loss)
        """
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, \
            f"序列长度{T}超过block_size {self.config.block_size}"

        # 位置索引0..T-1
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)   # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)   # (T, n_embd)，广播到batch
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # 训练：对全序列计算logits与损失
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # 推理：只需最后一个位置的logits，省显存
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def num_params(self, non_embedding=False):
        """
        统计参数量。
        non_embedding=True时按nanoGPT惯例减去位置嵌入wpe
        （词嵌入wte因与lm_head共享，本就只算一次）。
        """
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None,
                 eot_id=None):
        """
        自回归逐token生成。
        idx: (B, T) 起始上下文
        temperature: 温度，越大越随机；<=0时退化为贪心
        top_k:  只在概率最高的k个token中采样
        top_p:  核采样，累积概率达到p的最小集合内采样
        eot_id: 若提供，B=1时遇到该token提前停止
        """
        self.eval()
        for _ in range(max_new_tokens):
            # 上下文超长时裁剪到最后block_size个token
            idx_cond = idx if idx.size(1) <= self.config.block_size \
                else idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # 取最后一步 (B, vocab)

            if temperature <= 0:
                # 贪心解码
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                # Top-k过滤（top_k为None或<=0表示不启用）
                if top_k is not None and top_k > 0:
                    k = min(top_k, logits.size(-1))
                    v, _ = torch.topk(logits, k)
                    logits[logits < v[:, [-1]]] = float("-inf")

                # Top-p（核）过滤
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    # 标记累积概率超过top_p的token（保留刚好越过阈值的那个）
                    sorted_mask = cum_probs > top_p
                    sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                    sorted_mask[:, 0] = False
                    # 映射回原始顺序
                    mask = sorted_mask.scatter(1, sorted_idx, sorted_mask)
                    logits[mask] = float("-inf")

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

            # 单条序列遇到结束符则停止
            if eot_id is not None and idx.size(0) == 1 and idx_next.item() == eot_id:
                break

        return idx


# 预设配置：名称 -> 结构超参
PRESETS = {
    # 约17M参数（含词嵌入），适合快速验证与小语料
    "gpt2-mini": dict(n_layer=6, n_head=6, n_embd=384, block_size=512, vocab_size=16384),
    # 约34M参数，教学演示的主力规模
    "gpt2-demo": dict(n_layer=8, n_head=8, n_embd=512, block_size=512, vocab_size=16384),
    # 极小配置，仅用于冒烟测试（不在任务要求的两个预设内）
    "gpt2-tiny": dict(n_layer=2, n_head=4, n_embd=128, block_size=256, vocab_size=16384),
}


def build_model(preset="gpt2-mini", vocab_size=None, block_size=None, dropout=0.0):
    """按预设名构建GPT模型，可覆盖vocab_size/block_size/dropout"""
    if preset not in PRESETS:
        raise ValueError(f"未知预设 {preset}，可选：{list(PRESETS.keys())}")
    cfg_kwargs = dict(PRESETS[preset])
    if vocab_size is not None:
        cfg_kwargs["vocab_size"] = vocab_size
    if block_size is not None:
        cfg_kwargs["block_size"] = block_size
    cfg_kwargs["dropout"] = dropout
    config = GPTConfig(**cfg_kwargs)
    return GPT(config)


def _format_params(n):
    """把参数量格式化成人类可读字符串"""
    return f"{n / 1e6:.2f}M ({n:,})"


if __name__ == "__main__":
    # 直接运行时打印各预设的参数量，便于核对规模
    for name in ("gpt2-mini", "gpt2-demo", "gpt2-tiny"):
        m = build_model(name)
        total = m.num_params()
        non_emb = m.num_params(non_embedding=True)
        print(f"[{name}] 总参数量 {_format_params(total)} | "
              f"非嵌入参数量 {_format_params(non_emb)}")
