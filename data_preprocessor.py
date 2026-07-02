"""
数据预处理模块
负责文本数据的加载、清洗、分词和编码
支持普通文本和问答对（question, answer）两种训练样本
"""

import re
import pickle
from collections import Counter
import numpy as np
from config import Config


class DataPreprocessor:
    """数据预处理器"""

    def __init__(self, preserve_case=False):
        self.vocab = {}
        self.vocab_size = 0
        self.preserve_case = preserve_case  # 是否保留大小写（对话模式）
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
            '<SEP>': 4  # 问答分隔符：<BOS> 问题 <SEP> 回答 <EOS>
        }

    def clean_text(self, text, preserve_case=None):
        """清洗文本数据

        Args:
            text: 要清洗的文本
            preserve_case: 是否保留大小写，None时使用实例设置
        """
        if preserve_case is None:
            preserve_case = self.preserve_case

        if not preserve_case:
            text = text.lower()

        # 保留字母、数字、基本标点和撇号（I'm / don't 等常见缩写）
        text = re.sub(r"[^a-zA-Z0-9\s.,!?'\";:\-]", '', text)

        # 标点作为独立token，避免"Hello,"和"Hello"成为两个不同的词
        text = re.sub(r'([.,!?;:"])', r' \1 ', text)

        # 规范化空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def _flatten_items(items):
        """将混合样本（字符串或问答对）展平为纯文本列表"""
        texts = []
        for item in items:
            if isinstance(item, (tuple, list)):
                texts.extend(item)
            else:
                texts.append(item)
        return texts

    def build_vocab(self, items, max_vocab_size=Config.VOCAB_SIZE, preserve_case=None):
        """构建词汇表

        Args:
            items: 样本列表，元素为文本或(question, answer)元组
            max_vocab_size: 最大词汇表大小
            preserve_case: 是否保留大小写，None时使用实例设置
        """
        # 统计词频
        word_counts = Counter()
        for text in self._flatten_items(items):
            cleaned_text = self.clean_text(text, preserve_case=preserve_case)
            word_counts.update(cleaned_text.split())

        # 选择最常见的词
        most_common = word_counts.most_common(max_vocab_size - len(self.special_tokens))

        # 构建词汇表
        self.vocab = dict(self.special_tokens)
        current_idx = len(self.special_tokens)

        for word, _ in most_common:
            self.vocab[word] = current_idx
            current_idx += 1

        self.vocab_size = len(self.vocab)
        return self.vocab

    def _word_to_id(self, word):
        """查词，找不到时尝试大小写变体，最后回退到<UNK>"""
        for candidate in (word, word.lower(), word.capitalize()):
            if candidate in self.vocab:
                return self.vocab[candidate]
        return self.special_tokens['<UNK>']

    def _tokenize(self, text, preserve_case=None):
        """清洗并转换为ID列表（不含特殊标记）"""
        cleaned = self.clean_text(text, preserve_case=preserve_case)
        return [self._word_to_id(word) for word in cleaned.split()]

    def _pad_or_truncate(self, ids, max_length):
        """填充或截断到固定长度，截断时保证以<EOS>结尾"""
        if len(ids) > max_length:
            ids = ids[:max_length - 1] + [self.special_tokens['<EOS>']]
        else:
            ids = ids + [self.special_tokens['<PAD>']] * (max_length - len(ids))
        return ids

    def text_to_ids(self, text, max_length=Config.MAX_SEQ_LEN, preserve_case=None):
        """将文本转换为定长ID序列：<BOS> 文本 <EOS> [<PAD>...]"""
        ids = [self.special_tokens['<BOS>']]
        ids += self._tokenize(text, preserve_case=preserve_case)
        ids.append(self.special_tokens['<EOS>'])
        return self._pad_or_truncate(ids, max_length)

    def pair_to_ids(self, question, answer, max_length=Config.MAX_SEQ_LEN):
        """将问答对转换为定长ID序列：<BOS> 问题 <SEP> 回答 <EOS> [<PAD>...]"""
        ids = [self.special_tokens['<BOS>']]
        ids += self._tokenize(question)
        ids.append(self.special_tokens['<SEP>'])
        ids += self._tokenize(answer)
        ids.append(self.special_tokens['<EOS>'])
        return self._pad_or_truncate(ids, max_length)

    def encode_prompt(self, text):
        """编码推理提示：<BOS> 文本 [<SEP>]，不填充、不加<EOS>

        词表中存在<SEP>时追加它，提示模型接下来生成回答
        """
        ids = [self.special_tokens['<BOS>']]
        ids += self._tokenize(text)
        if '<SEP>' in self.vocab:
            ids.append(self.vocab['<SEP>'])
        return ids

    def ids_to_text(self, ids):
        """将ID序列转换回文本，遇到<EOS>停止，跳过其他特殊标记"""
        reverse_vocab = {v: k for k, v in self.vocab.items()}

        words = []
        for idx in ids:
            word = reverse_vocab.get(idx, '<UNK>')
            if word == '<EOS>':
                break
            if word in ('<BOS>', '<PAD>', '<SEP>'):
                continue
            words.append(word)

        text = ' '.join(words)
        # 恢复标点与前一个词的粘连（分词时标点被拆成了独立token）
        text = re.sub(r"\s+([.,!?;:])", r'\1', text)
        return text

    def save_vocab(self, filepath):
        """保存词汇表"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.vocab, f)

    def load_vocab(self, filepath):
        """加载词汇表"""
        with open(filepath, 'rb') as f:
            self.vocab = pickle.load(f)
        self.vocab_size = len(self.vocab)
        return self.vocab

    def prepare_dataset(self, items, build_vocab=True, max_length=None):
        """准备训练数据集

        Args:
            items: 样本列表，元素为文本或(question, answer)元组
            build_vocab: 是否重新构建词汇表
            max_length: 序列填充长度，默认Config.TRAIN_MAX_SEQ_LEN
        """
        if max_length is None:
            max_length = Config.TRAIN_MAX_SEQ_LEN

        # 构建词汇表（可选）
        if build_vocab:
            self.build_vocab(items)

        # 创建输入和目标序列（用于语言模型训练）
        # 输入是序列的前n-1个token，目标是后n-1个token
        pad_id = self.special_tokens['<PAD>']
        sep_id = self.special_tokens['<SEP>']

        inputs = []
        targets = []

        for item in items:
            if isinstance(item, (tuple, list)):
                seq = self.pair_to_ids(item[0], item[1], max_length)
                target = seq[1:]
                # 问答对只在回答部分计算损失（问题token置为<PAD>被忽略），
                # 让梯度集中于"问题→回答"的映射而不是复述问题
                if sep_id in seq:
                    sep_idx = seq.index(sep_id)
                    target = [pad_id] * sep_idx + target[sep_idx:]
                inputs.append(seq[:-1])
                targets.append(target)
            else:
                seq = self.text_to_ids(item, max_length)
                inputs.append(seq[:-1])
                targets.append(seq[1:])

        return np.array(inputs), np.array(targets)
