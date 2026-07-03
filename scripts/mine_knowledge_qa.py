"""
从预训练维基语料中挖掘"实体→定义"知识问答对（v0.4.2）

维基文章的首句通常是"X，简称Y，是……。"式的定义句。本脚本抽取
（实体，定义句）对，供对话微调使用，让模型把预训练学到的知识
以问答形式暴露出来。

策略：全量扫描所有语料分片收集候选，再按实体名长度升序优选
（名字越短通常越是主条目/常见概念），截取前max_pairs组。

输出TSV：每行 "实体\t定义句"，微调脚本会为每个实体生成
"介绍一下X"/"X是什么"等提问变体。

用法：python3 mine_knowledge_qa.py [--max-pairs 3000]
"""

import argparse
import glob
import re

# 实体：2~10个汉字/字母/数字/间隔号，后面可跟（外文名）、至多两个短插入语，然后是"是"
ENTITY_RE = re.compile(
    r"^([0-9A-Za-z一-鿿·]{2,10})"
    r"(?:（[^（）]{0,60}）)?"
    r"(?:，[^，。]{1,20}){0,2}"
    r"，?是"
)
# 明显不是干净实体名的模式：纯数字、年份开头、"又称/或称"等被误并入结尾
BAD_ENTITY_RE = re.compile(
    r"^\d+$|^[0-9]+年|(?:又称|或称|通称|简称|全称|也称|旧称|前身)$"
    r"|(?:又|或|通|简|全|也|旧)称$")


def first_sentences(text, max_chars=110):
    """取首句；若首句很短则并入第二句，总长不超过max_chars"""
    parts = text.split("。")
    if not parts or not parts[0]:
        return None
    ans = parts[0] + "。"
    if len(ans) < 40 and len(parts) > 1 and parts[1]:
        cand = ans + parts[1] + "。"
        if len(cand) <= max_chars:
            ans = cand
    if not (12 <= len(ans) <= max_chars):
        return None
    return ans


def mine(files):
    pairs = {}
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if len(line) < 30:
                    continue
                m = ENTITY_RE.match(line)
                if not m:
                    continue
                entity = m.group(1)
                if entity in pairs or BAD_ENTITY_RE.search(entity):
                    continue
                ans = first_sentences(line)
                if ans is None or "*" in ans or ans.count("，") > 6:
                    continue
                pairs[entity] = ans
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-glob", default="data/pretrain/corpus/corpus_*.txt")
    ap.add_argument("--out", default="data/knowledge_qa.tsv")
    ap.add_argument("--max-pairs", type=int, default=3000)
    args = ap.parse_args()

    files = sorted(glob.glob(args.corpus_glob))
    pairs = mine(files)
    print(f"候选实体共{len(pairs)}组")

    # 名字短的优先（更可能是主条目/常见概念），同长度按定义句短的优先
    selected = sorted(pairs.items(),
                      key=lambda kv: (len(kv[0]), len(kv[1])))[:args.max_pairs]

    with open(args.out, "w", encoding="utf-8") as f:
        for entity, ans in selected:
            f.write(f"{entity}\t{ans}\n")
    print(f"挖掘完成：{len(selected)}组实体定义，已写入 {args.out}")
    for e, a in selected[:5]:
        print(f"  例：{e} -> {a[:50]}…")


if __name__ == "__main__":
    main()
