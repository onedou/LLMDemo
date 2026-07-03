"""
Byte-Level BPE分词器模块

基于HuggingFace的tokenizers库训练一个字节级BPE分词器，用于中文预训练。
字节级(Byte-Level)的好处：把文本先编码成UTF-8字节再做BPE，任意字符（含中文、
生僻字、emoji）都能被无损表示，encode/decode严格往返一致，无需预先准备字典。

用法：
  # 训练
  python bpe_tokenizer.py train --files data/pretrain/corpus/*.txt --vocab-size 16384
  # 代码中使用
  tok = BPETokenizer.load("models/bpe_tokenizer")
  ids = tok.encode("你好，世界")
  text = tok.decode(ids)
"""

import argparse
import glob
import os

from tokenizers import ByteLevelBPETokenizer
from tokenizers import Tokenizer

# 文档结束符：GPT-2沿用的特殊token，用于分隔不同文档
EOT_TOKEN = "<|endoftext|>"
# 分词器文件名（存到save_dir下）
TOKENIZER_FILE = "tokenizer.json"


def train_bpe(files, vocab_size=16384, save_dir="models/bpe_tokenizer",
              min_frequency=2):
    """
    训练字节级BPE分词器。
    files:      语料文件路径列表（每行一篇文档的纯文本）
    vocab_size: 目标词表大小
    save_dir:   分词器保存目录
    """
    files = list(files)
    if not files:
        raise ValueError("没有可用于训练的语料文件")
    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"语料文件不存在：{f}")

    print(f"开始训练BPE：{len(files)}个文件，目标词表{vocab_size}")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=files,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        # 只保留文档结束符一个特殊token，让它拿到稳定的id
        special_tokens=[EOT_TOKEN],
    )

    os.makedirs(save_dir, exist_ok=True)
    # 用统一的tokenizer.json格式保存（后续用Tokenizer.from_file加载最省事）
    out_path = os.path.join(save_dir, TOKENIZER_FILE)
    tokenizer.save(out_path)

    actual_vocab = tokenizer.get_vocab_size()
    eot_id = tokenizer.token_to_id(EOT_TOKEN)
    print(f"训练完成：实际词表大小{actual_vocab}，"
          f"{EOT_TOKEN} 的id={eot_id}")
    print(f"已保存到：{out_path}")
    return save_dir


class BPETokenizer:
    """字节级BPE分词器的加载/编码/解码封装"""

    def __init__(self, tokenizer):
        self._tok = tokenizer
        self.eot_token = EOT_TOKEN
        self.eot_id = tokenizer.token_to_id(EOT_TOKEN)
        # 词表大小（供构建模型时对齐vocab_size使用）
        self.vocab_size = tokenizer.get_vocab_size()

    @classmethod
    def load(cls, save_dir="models/bpe_tokenizer"):
        """从目录加载分词器"""
        path = os.path.join(save_dir, TOKENIZER_FILE)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"未找到分词器文件：{path}，请先运行 train 训练")
        tokenizer = Tokenizer.from_file(path)
        return cls(tokenizer)

    def encode(self, text, add_eot=False):
        """
        文本 -> token id列表。
        add_eot=True时在末尾追加文档结束符（编码整篇文档时用）。
        """
        ids = self._tok.encode(text).ids
        if add_eot:
            ids = ids + [self.eot_id]
        return ids

    def decode(self, ids):
        """token id列表 -> 文本（字节级保证往返无损）"""
        return self._tok.decode(ids)


def _expand_files(patterns):
    """把命令行传入的glob模式展开成实际文件列表"""
    files = []
    for pat in patterns:
        matched = sorted(glob.glob(pat))
        if matched:
            files.extend(matched)
        elif os.path.exists(pat):
            files.append(pat)
    return files


def _cli():
    parser = argparse.ArgumentParser(description="字节级BPE分词器训练/测试")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="训练分词器")
    p_train.add_argument("--files", nargs="+", required=True,
                         help="语料文件或glob，如 data/pretrain/corpus/*.txt")
    p_train.add_argument("--vocab-size", type=int, default=16384)
    p_train.add_argument("--save-dir", default="models/bpe_tokenizer")
    p_train.add_argument("--min-frequency", type=int, default=2)

    p_test = sub.add_parser("test", help="加载并测试一段文本的往返")
    p_test.add_argument("--save-dir", default="models/bpe_tokenizer")
    p_test.add_argument("--text", default="你好，世界！这是一次BPE分词往返测试。")

    args = parser.parse_args()

    if args.cmd == "train":
        files = _expand_files(args.files)
        train_bpe(files, vocab_size=args.vocab_size,
                  save_dir=args.save_dir, min_frequency=args.min_frequency)
    elif args.cmd == "test":
        tok = BPETokenizer.load(args.save_dir)
        ids = tok.encode(args.text)
        back = tok.decode(ids)
        print(f"原文：{args.text}")
        print(f"token数：{len(ids)}  词表大小：{tok.vocab_size}  eot_id：{tok.eot_id}")
        print(f"ids：{ids}")
        print(f"还原：{back}")
        print("往返一致" if back == args.text else "往返不一致（异常）")


if __name__ == "__main__":
    _cli()
