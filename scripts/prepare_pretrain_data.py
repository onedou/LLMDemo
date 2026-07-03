#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_pretrain_data.py
========================
中文预训练语料的「下载 + 清洗 + 统计」一体化可复现脚本。

本轮迭代目标：为 10M~50M 参数的 Nova 基座模型（GPT-2 风格结构）准备中文预训练语料。

语料来源（默认）
----------------
HuggingFace 数据集：pleisto/wikipedia-cn-20230720-filtered
  - 基于中文维基 2023-07-20 dump，保留约 25.4 万条高质量词条
  - 已做简繁转换 + 大陆习惯用词转换（即已是简体中文）
  - 单个 JSON 文件，约 500MB
  - 许可证：CC BY-SA 3.0
  - JSON 结构：[{"completion": "<正文>", "source": "..."}, ...]

清洗规则
--------
  1. 去除 HTML 标签、常见 wiki 标记（[[..]] {{..}} '' == ==）、引用角标（[1]、[来源请求] 等）
  2. 把文档内部换行/制表符合并成空格 —— 每行一篇文档
  3. 规范化空白（全角空格/连续空白 -> 单个半角空格）
  4. 繁体转简体（opencc t2s，作为兜底；本数据集大多已是简体）
  5. 丢弃清洗后长度 < MIN_CHARS（默认 30）的文档
  6. 按整行精确去重（blake2b 指纹）
  7. 分片输出，每片约 SHARD_MB（默认 50）MB：corpus_000.txt, corpus_001.txt ...

产出
----
  data/pretrain/raw/     原始下载文件
  data/pretrain/corpus/  清洗后的分片纯文本（UTF-8，每行一篇文档）

依赖
----
  pip install huggingface_hub ijson opencc-python-reimplemented
  （transformers 4.57.x 已自带 huggingface_hub；ijson 用于流式解析大 JSON；
    opencc 用于繁转简，缺失时会跳过繁转简并给出提示。）

用法
----
  # 一条龙：下载 -> 清洗 -> 统计（最常用）
  python prepare_pretrain_data.py --all

  # 仅下载原始语料到 data/pretrain/raw/
  python prepare_pretrain_data.py --download

  # 仅清洗（假设已下载）
  python prepare_pretrain_data.py --clean

  # 仅对已生成的 corpus/ 做统计与抽样展示
  python prepare_pretrain_data.py --stats

  # 常用可调参数
  python prepare_pretrain_data.py --all --shard-mb 50 --min-chars 30
  # 原始文件 > --rm-raw-threshold-gb（默认 1GB）时，清洗后自动删除以省磁盘
  python prepare_pretrain_data.py --all --keep-raw            # 强制保留原始文件
"""

import argparse
import hashlib
import os
import re
import sys

# ----------------------------------------------------------------------------
# 默认配置
# ----------------------------------------------------------------------------
REPO_ID = "pleisto/wikipedia-cn-20230720-filtered"
REPO_TYPE = "dataset"
RAW_FILENAME = "wikipedia-cn-20230720-filtered.json"
JSON_TEXT_FIELD = "completion"  # 每条记录里正文所在字段

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(ROOT, "data", "pretrain", "raw")
CORPUS_DIR = os.path.join(ROOT, "data", "pretrain", "corpus")

DEFAULT_SHARD_MB = 50
DEFAULT_MIN_CHARS = 30
DEFAULT_RM_RAW_GB = 1.0  # 原始文件超过该阈值(GB)则在清洗后删除

# ----------------------------------------------------------------------------
# 清洗用正则（预编译）
# ----------------------------------------------------------------------------
_RE_HTML_TAG = re.compile(r"<[^>]+>")                # HTML/XML 标签
_RE_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)   # HTML 注释
_RE_WIKI_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")    # {{...}} 模板
_RE_WIKI_HEADING = re.compile(r"={2,}\s*([^=]+?)\s*={2,}")  # ==标题== -> 标题
_RE_WIKI_BOLD_ITALIC = re.compile(r"'{2,5}")         # '' ''' '''''
_RE_REF_NUM = re.compile(r"\[\d{1,4}\]")             # [1] [12] 数字角标
_RE_REF_TAG = re.compile(                            # 常见中/英文引用标记
    r"\[(?:来源请求|來源請求|需要?引用|引用请求|注\s*\d*|編輯|编辑|"
    r"citation needed|edit)\]",
    re.I,
)
_RE_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")  # 控制字符
_RE_WS = re.compile(r"\s+")                          # 任意空白（含 \n \t 　）


def clean_text(text, converter):
    """把一条原始正文清洗成「一行文档」。返回清洗后的字符串（可能为空）。"""
    if not text:
        return ""
    # 1) 去 wiki 标题标记（保留标题文字）
    text = _RE_WIKI_HEADING.sub(r"\1", text)
    # 2) 去 HTML 注释与模板（可能嵌套，循环直到不再变化）
    text = _RE_HTML_COMMENT.sub(" ", text)
    prev = None
    while prev != text:
        prev = text
        text = _RE_WIKI_TEMPLATE.sub(" ", text)
    # 3) 去 HTML 标签
    text = _RE_HTML_TAG.sub(" ", text)
    # 4) 去引用角标 / 粗体斜体标记
    text = _RE_REF_NUM.sub("", text)
    text = _RE_REF_TAG.sub("", text)
    text = _RE_WIKI_BOLD_ITALIC.sub("", text)
    # 5) 去控制字符
    text = _RE_CTRL.sub(" ", text)
    # 6) 换行/制表/全角空格等所有空白 -> 单个半角空格（合并为一行）
    text = _RE_WS.sub(" ", text).strip()
    # 7) 繁转简（兜底）
    if converter is not None and text:
        text = converter.convert(text)
    return text


def get_converter():
    """返回 opencc t2s 转换器；若未安装 opencc 则返回 None 并提示。"""
    try:
        import opencc
        return opencc.OpenCC("t2s")
    except Exception:
        print("[WARN] 未安装 opencc（opencc-python-reimplemented），"
              "跳过繁转简。建议：pip install opencc-python-reimplemented",
              file=sys.stderr)
        return None


# ----------------------------------------------------------------------------
# 下载
# ----------------------------------------------------------------------------
def do_download():
    """从 HuggingFace 下载原始语料到 RAW_DIR。返回本地文件路径。"""
    from huggingface_hub import hf_hub_download
    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"[INFO] 正在从 HuggingFace 下载 {REPO_ID}/{RAW_FILENAME} ...")
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=RAW_FILENAME,
        repo_type=REPO_TYPE,
        local_dir=RAW_DIR,
    )
    size = os.path.getsize(path)
    print(f"[INFO] 下载完成：{path}  ({size/1e6:.1f} MB)")
    return path


# ----------------------------------------------------------------------------
# 清洗（流式）
# ----------------------------------------------------------------------------
def iter_records(raw_path):
    """流式产出原始 JSON 数组里每条记录的正文文本，避免整文件读入内存。"""
    import ijson
    with open(raw_path, "rb") as f:
        # 该数据集是顶层 JSON 数组，'item' 匹配数组内每个对象
        for obj in ijson.items(f, "item"):
            if isinstance(obj, dict):
                yield obj.get(JSON_TEXT_FIELD, "") or ""
            elif isinstance(obj, str):
                yield obj


class ShardWriter:
    """按字节大小滚动分片写出 corpus_000.txt, corpus_001.txt ..."""

    def __init__(self, out_dir, shard_bytes, prefix="corpus_"):
        self.out_dir = out_dir
        self.shard_bytes = shard_bytes
        self.prefix = prefix
        self.idx = 0
        self.fh = None
        self.cur_bytes = 0
        self.total_bytes = 0
        os.makedirs(out_dir, exist_ok=True)

    def _open_new(self):
        if self.fh is not None:
            self.fh.close()
        name = os.path.join(self.out_dir, f"{self.prefix}{self.idx:03d}.txt")
        self.fh = open(name, "w", encoding="utf-8")
        self.cur_bytes = 0

    def write_line(self, line):
        if self.fh is None:
            self._open_new()
        data = line + "\n"
        nbytes = len(data.encode("utf-8"))
        # 当前分片已有内容且将写满，则滚动到下一片
        if self.cur_bytes > 0 and self.cur_bytes + nbytes > self.shard_bytes:
            self.idx += 1
            self._open_new()
        self.fh.write(data)
        self.cur_bytes += nbytes
        self.total_bytes += nbytes

    def close(self):
        if self.fh is not None:
            self.fh.close()
            self.fh = None


def do_clean(shard_mb, min_chars, rm_raw_gb, keep_raw):
    """清洗 RAW_DIR 下的原始语料，产出到 CORPUS_DIR。返回统计 dict。"""
    raw_path = os.path.join(RAW_DIR, RAW_FILENAME)
    if not os.path.exists(raw_path):
        print(f"[ERROR] 未找到原始文件 {raw_path}，请先执行 --download",
              file=sys.stderr)
        sys.exit(1)

    converter = get_converter()

    # 清空旧的 corpus 分片，避免残留
    if os.path.isdir(CORPUS_DIR):
        for fn in os.listdir(CORPUS_DIR):
            if re.match(r"corpus_\d+\.txt$", fn):
                os.remove(os.path.join(CORPUS_DIR, fn))

    writer = ShardWriter(CORPUS_DIR, shard_mb * 1024 * 1024)
    seen = set()  # 整行去重指纹（blake2b 16 字节）

    n_in = n_kept = n_dup = n_short = 0
    total_chars = 0

    print(f"[INFO] 开始清洗（min_chars={min_chars}, shard={shard_mb}MB）...")
    for raw_text in iter_records(raw_path):
        n_in += 1
        line = clean_text(raw_text, converter)
        if len(line) < min_chars:
            n_short += 1
            continue
        fp = hashlib.blake2b(line.encode("utf-8"), digest_size=16).digest()
        if fp in seen:
            n_dup += 1
            continue
        seen.add(fp)
        writer.write_line(line)
        n_kept += 1
        total_chars += len(line)
        if n_in % 20000 == 0:
            print(f"    ...已读取 {n_in} 条，保留 {n_kept} 条")
    writer.close()

    n_shards = writer.idx + 1 if writer.total_bytes > 0 else 0
    print(f"[INFO] 清洗完成：读取 {n_in} 条 -> 保留 {n_kept} 条 "
          f"(丢弃过短 {n_short}，重复 {n_dup})，共 {n_shards} 个分片")

    # 原始文件过大则删除以省磁盘
    raw_size = os.path.getsize(raw_path)
    if not keep_raw and raw_size > rm_raw_gb * 1e9:
        os.remove(raw_path)
        print(f"[INFO] 原始文件 {raw_size/1e9:.2f}GB > {rm_raw_gb}GB，已删除以省磁盘")
    else:
        print(f"[INFO] 保留原始文件（{raw_size/1e6:.1f}MB）")

    return {
        "n_in": n_in,
        "n_kept": n_kept,
        "n_dup": n_dup,
        "n_short": n_short,
        "total_chars": total_chars,
        "n_shards": n_shards,
        "total_bytes": writer.total_bytes,
    }


# ----------------------------------------------------------------------------
# 统计与抽样
# ----------------------------------------------------------------------------
def do_stats(sample_n=3, sample_chars=200):
    """扫描 CORPUS_DIR，输出总大小/文档数/总字符数，并抽样展示若干段文本。"""
    if not os.path.isdir(CORPUS_DIR):
        print(f"[ERROR] 未找到语料目录 {CORPUS_DIR}", file=sys.stderr)
        sys.exit(1)
    shards = sorted(fn for fn in os.listdir(CORPUS_DIR)
                    if re.match(r"corpus_\d+\.txt$", fn))
    if not shards:
        print(f"[ERROR] {CORPUS_DIR} 下没有 corpus_*.txt", file=sys.stderr)
        sys.exit(1)

    total_bytes = total_docs = total_chars = 0
    samples = []
    for shard in shards:
        path = os.path.join(CORPUS_DIR, shard)
        total_bytes += os.path.getsize(path)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                total_docs += 1
                total_chars += len(line)
                # 抽样：第 1、第 10 万、第 20 万篇的前 sample_chars 字
                if len(samples) < sample_n and total_docs in (1, 100000, 200000):
                    samples.append(line[:sample_chars])
    # 若文档数不足以命中上面的定点，退回取前 sample_n 行
    if len(samples) < sample_n:
        samples = []
        with open(os.path.join(CORPUS_DIR, shards[0]), encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= sample_n:
                    break
                samples.append(line.rstrip("\n")[:sample_chars])

    print("\n" + "=" * 60)
    print("语料统计报告 (data/pretrain/corpus/)")
    print("=" * 60)
    print(f"分片数量   : {len(shards)}  ({', '.join(shards)})")
    print(f"总大小     : {total_bytes/1e6:.1f} MB")
    print(f"文档数     : {total_docs:,}")
    print(f"总字符数   : {total_chars:,}")
    if total_docs:
        print(f"平均文档长度: {total_chars/total_docs:.0f} 字/篇")
    print("-" * 60)
    for i, s in enumerate(samples, 1):
        print(f"[样例 {i}] {s}")
        print("-" * 60)
    return {
        "n_shards": len(shards),
        "total_bytes": total_bytes,
        "total_docs": total_docs,
        "total_chars": total_chars,
    }


# ----------------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="中文预训练语料 下载/清洗/统计 一体化脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--download", action="store_true", help="下载原始语料")
    ap.add_argument("--clean", action="store_true", help="清洗原始语料")
    ap.add_argument("--stats", action="store_true", help="统计已生成的语料")
    ap.add_argument("--all", action="store_true",
                    help="等价于 --download --clean --stats")
    ap.add_argument("--shard-mb", type=int, default=DEFAULT_SHARD_MB,
                    help=f"每个分片大小(MB)，默认 {DEFAULT_SHARD_MB}")
    ap.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS,
                    help=f"丢弃短于该字符数的文档，默认 {DEFAULT_MIN_CHARS}")
    ap.add_argument("--rm-raw-threshold-gb", type=float, default=DEFAULT_RM_RAW_GB,
                    help=f"原始文件超过该大小(GB)则清洗后删除，默认 {DEFAULT_RM_RAW_GB}")
    ap.add_argument("--keep-raw", action="store_true", help="强制保留原始文件")
    args = ap.parse_args()

    if not (args.download or args.clean or args.stats or args.all):
        ap.print_help()
        sys.exit(0)

    if args.all or args.download:
        do_download()
    if args.all or args.clean:
        do_clean(args.shard_mb, args.min_chars,
                 args.rm_raw_threshold_gb, args.keep_raw)
    if args.all or args.stats:
        do_stats()


if __name__ == "__main__":
    main()
