"""
语料编码模块：把清洗后的中文语料编码成二进制token文件

流程：
  1. 流式逐行读取corpus文件（每行一篇文档），不把整份语料读进内存
  2. 每篇文档用BPE编码后追加一个<|endoftext|>，把不同文档隔开
  3. token以uint16写入（词表16384 < 65536，uint16安全且省一半磁盘）
  4. 先写到临时的all.bin，统计总token数后再按99:1切成train.bin / val.bin

用法：
  python encode_corpus.py --corpus "data/pretrain/corpus/*.txt" \
      --tokenizer-dir models/bpe_tokenizer --out-dir data/pretrain
"""

import argparse
import glob
import os

import numpy as np
from tqdm import tqdm

from bpe_tokenizer import BPETokenizer

# uint16上限，超过说明词表选择有误
UINT16_MAX = 65535
# 累积多少token后刷一次盘，控制内存占用
FLUSH_EVERY = 1_000_000


def _expand(patterns):
    """展开glob模式为文件列表"""
    files = []
    for pat in patterns:
        m = sorted(glob.glob(pat))
        files.extend(m if m else ([pat] if os.path.exists(pat) else []))
    return files


def encode_corpus(corpus_files, tokenizer_dir="models/bpe_tokenizer",
                  out_dir="data/pretrain", val_ratio=0.01):
    """
    把语料编码为train.bin / val.bin。
    返回 (总token数, 训练token数, 验证token数)。
    """
    corpus_files = _expand(corpus_files)
    if not corpus_files:
        raise ValueError("没有匹配到任何语料文件")

    tok = BPETokenizer.load(tokenizer_dir)
    assert tok.vocab_size <= UINT16_MAX, \
        f"词表{tok.vocab_size}超过uint16上限，请改用uint32"

    os.makedirs(out_dir, exist_ok=True)
    tmp_path = os.path.join(out_dir, "_all.tmp.bin")

    print(f"编码{len(corpus_files)}个语料文件，分词器词表={tok.vocab_size}，"
          f"eot_id={tok.eot_id}")

    n_docs = 0
    total_tokens = 0
    buf = []  # 累积token的缓冲区，满FLUSH_EVERY后落盘

    with open(tmp_path, "wb") as fout:
        for path in corpus_files:
            with open(path, "r", encoding="utf-8") as fin:
                for line in tqdm(fin, desc=os.path.basename(path), unit="doc"):
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    # 每篇文档末尾追加eot，作为文档边界
                    ids = tok.encode(line, add_eot=True)
                    buf.extend(ids)
                    n_docs += 1
                    total_tokens += len(ids)

                    if len(buf) >= FLUSH_EVERY:
                        np.asarray(buf, dtype=np.uint16).tofile(fout)
                        buf = []
        # 收尾刷盘
        if buf:
            np.asarray(buf, dtype=np.uint16).tofile(fout)
            buf = []

    if total_tokens == 0:
        os.remove(tmp_path)
        raise ValueError("语料为空，没有产生任何token")

    # 按token数做99:1切分
    n_val = int(total_tokens * val_ratio)
    n_train = total_tokens - n_val

    train_path = os.path.join(out_dir, "train.bin")
    val_path = os.path.join(out_dir, "val.bin")

    # 源文件与目标文件都用memmap，分块拷贝，避免一次性载入内存
    src = np.memmap(tmp_path, dtype=np.uint16, mode="r", shape=(total_tokens,))
    train_mm = np.memmap(train_path, dtype=np.uint16, mode="w+", shape=(n_train,))
    val_mm = np.memmap(val_path, dtype=np.uint16, mode="w+", shape=(n_val,))

    _chunked_copy(src, train_mm, 0, n_train)
    _chunked_copy(src, val_mm, n_train, n_val)

    train_mm.flush()
    val_mm.flush()
    del src, train_mm, val_mm
    os.remove(tmp_path)

    print("-" * 50)
    print(f"文档数：{n_docs:,}")
    print(f"总token数：{total_tokens:,}")
    print(f"train.bin：{n_train:,} tokens -> {train_path}")
    print(f"val.bin  ：{n_val:,} tokens -> {val_path}")
    print(f"平均每篇文档 {total_tokens / max(n_docs, 1):.1f} tokens")
    return total_tokens, n_train, n_val


def _chunked_copy(src, dst, src_offset, count, chunk=1_000_000):
    """从src[src_offset:src_offset+count]分块拷贝到dst[0:count]"""
    done = 0
    while done < count:
        n = min(chunk, count - done)
        dst[done:done + n] = src[src_offset + done:src_offset + done + n]
        done += n


def _cli():
    parser = argparse.ArgumentParser(description="语料 -> uint16二进制token文件")
    parser.add_argument("--corpus", nargs="+",
                        default=["data/pretrain/corpus/*.txt"],
                        help="语料文件或glob")
    parser.add_argument("--tokenizer-dir", default="models/bpe_tokenizer")
    parser.add_argument("--out-dir", default="data/pretrain")
    parser.add_argument("--val-ratio", type=float, default=0.01)
    args = parser.parse_args()

    encode_corpus(args.corpus, tokenizer_dir=args.tokenizer_dir,
                  out_dir=args.out_dir, val_ratio=args.val_ratio)


if __name__ == "__main__":
    _cli()
