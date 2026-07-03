"""
Nova对话模型推理封装（v0.4，GPT-2风格结构）

加载微调后的对话模型（models/gpt2_chat/chat.pt），提供reply()单轮问答接口。
推理流程：
  1. 内置技能优先（时间/日期/算术等实时信息，语言模型不可靠）
  2. 知识库检索（data/knowledge_qa.tsv，从维基语料挖掘的实体定义）：
     命中实体则直接返回定义原文——34M模型记不住几千条事实，
     检索保证事实类回答零幻觉
  3. 模型生成：构造 "问：{用户输入}\n答：" 提示，自回归生成到
     <|endoftext|>为止（自带重复惩罚采样，抑制"复读机"退化）
  4. 清理：截断自问自答、折叠残余的循环重复、空回复兜底

命令行快速测试：python3 chat_inference.py
"""

import os
import re

import torch
import torch.nn.functional as F

import os
import sys
# 使脚本可从任意目录直接运行（把项目根目录加入模块搜索路径）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nova.bpe_tokenizer import BPETokenizer
from nova.gpt2_model import GPT, GPTConfig
from legacy.inference import LLMInference

PROMPT_PREFIX = "问："
ANSWER_PREFIX = "\n答："
KNOWLEDGE_TSV = "data/knowledge_qa.tsv"

# 事实类提问的模式：提取出实体后查知识库
KNOWLEDGE_PATTERNS = (
    re.compile(r"^(?:请)?介绍一下(.+)$"),
    re.compile(r"^什么是(.+)$"),
    re.compile(r"^(.+?)是什么$"),
    re.compile(r"^(.+)$"),  # 裸实体名
)


def _pick_device(override=None):
    if override:
        return override
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ChatBot:
    """微调Nova模型的单轮对话推理器"""

    def __init__(self, ckpt_path="models/gpt2_chat/chat.pt",
                 tokenizer_dir="models/bpe_tokenizer", device=None):
        self.device = _pick_device(device)
        self.tok = BPETokenizer.load(tokenizer_dir)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model = GPT(GPTConfig(**ckpt["config"]))
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device)
        self.model.eval()
        print(f"已加载对话模型：{ckpt_path}"
              f"（参数量{self.model.num_params()/1e6:.2f}M，设备{self.device}）")

        # 内置技能（时间/日期/算术）复用inference.py的规则实现。
        # 用__new__跳过__init__：这些方法不依赖任何实例状态，
        # 无需构建LLMInference里的旧词表模型
        self._skills = LLMInference.__new__(LLMInference)

        # 知识库：实体 -> 维基定义句（挖掘自预训练语料）
        self.knowledge = {}
        if os.path.exists(KNOWLEDGE_TSV):
            with open(KNOWLEDGE_TSV, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) == 2:
                        self.knowledge[parts[0]] = parts[1]
            print(f"知识库已加载：{len(self.knowledge)}条实体定义")

    def _knowledge_lookup(self, message):
        """事实类提问的检索：剥掉提问套话后精确匹配实体名"""
        msg = message.strip().rstrip("？?。！!，, ")
        for pat in KNOWLEDGE_PATTERNS:
            m = pat.match(msg)
            if m:
                entity = m.group(1).strip()
                if entity in self.knowledge:
                    return self.knowledge[entity]
        return None

    def reply(self, message, max_new_tokens=80, temperature=0.3, top_k=20,
              repetition_penalty=1.2):
        """生成单轮回复"""
        message = message.strip()
        if not message:
            return "请输入内容哦~"

        # 1. 内置技能优先
        builtin = LLMInference._builtin_response(self._skills, message)
        if builtin:
            return builtin

        # 2. 知识库检索（事实类问题不靠模型记忆，检索保证准确）
        known = self._knowledge_lookup(message)
        if known:
            return known

        # 3. 模型生成（带重复惩罚）
        ids = self.tok.encode(PROMPT_PREFIX + message + ANSWER_PREFIX)
        new_ids = self._generate(ids, max_new_tokens, temperature, top_k,
                                 repetition_penalty)
        text = self.tok.decode(new_ids)

        # 3. 清理
        return self._postprocess(text, message)

    @torch.no_grad()
    def _generate(self, prompt_ids, max_new_tokens, temperature, top_k,
                  repetition_penalty, penalty_window=64):
        """自回归采样，带重复惩罚：最近penalty_window个已生成token的
        logit按CTRL论文方式打折（正logit除以惩罚，负logit乘以惩罚），
        大幅降低小模型陷入短语循环（复读机）的概率
        """
        block_size = self.model.config.block_size
        x = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        generated = []

        for _ in range(max_new_tokens):
            x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
            logits, _ = self.model(x_cond)  # targets=None时只返回最后一步
            logits = logits[0, -1, :]

            if repetition_penalty and repetition_penalty > 1.0 and generated:
                recent = torch.tensor(sorted(set(generated[-penalty_window:])),
                                      dtype=torch.long, device=self.device)
                picked = logits[recent]
                logits[recent] = torch.where(
                    picked > 0, picked / repetition_penalty,
                    picked * repetition_penalty)

            if temperature <= 0:
                next_id = int(torch.argmax(logits))
            else:
                logits = logits / temperature
                if top_k and top_k > 0:
                    k = min(top_k, logits.size(-1))
                    v, _ = torch.topk(logits, k)
                    logits[logits < v[-1]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1))

            if next_id == self.tok.eot_id:
                break
            generated.append(next_id)
            x = torch.cat(
                [x, torch.tensor([[next_id]], dtype=torch.long,
                                 device=self.device)], dim=1)

        return generated

    def _postprocess(self, text, message):
        # 模型可能在回答后继续编造"问：xxx"，从第一个再现的问答标记处截断
        for marker in ("\n问", "问：", "\n答", "答："):
            pos = text.find(marker)
            if pos > 0:
                text = text[:pos]

        # 维基列表体残留：条目分隔符没有对话价值，直接去掉
        text = text.replace("*", " ")

        # 折叠循环重复：同一片段(2~30字符)连续出现2次以上时只保留一份
        # （重复惩罚已大幅降低出现概率，这里是最后一道兜底）
        text = re.sub(r"(.{2,30}?)(?:\s*\1){2,}", r"\1", text)

        text = re.sub(r"\s+", " ", text).strip()

        if len(text) < 2:
            if re.search(r"[一-鿿]", message):
                return "我在呢！有什么可以帮你的吗？"
            return "I'm here to help! What can I assist you with today?"
        return text


def _interactive():
    bot = ChatBot()
    print("进入对话模式（quit/exit退出）")
    while True:
        try:
            msg = input("\n👤 您: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg.lower() in ("quit", "exit", ""):
            break
        print(f"🤖 {bot.reply(msg)}")


if __name__ == "__main__":
    _interactive()
