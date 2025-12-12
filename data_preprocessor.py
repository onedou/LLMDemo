"""
数据预处理模块
负责文本数据的加载、清洗、分词和编码
"""

import re
import pickle
from collections import Counter
import numpy as np
from config import Config


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self):
        self.vocab = {}
        self.vocab_size = 0
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3
        }
    
    def clean_text(self, text):
        """清洗文本数据"""
        # 转换为小写
        text = text.lower()
        # 移除特殊字符，保留字母、数字和基本标点
        text = re.sub(r'[^a-zA-Z0-9\s.,!?;:]', '', text)
        # 规范化空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def build_vocab(self, texts, max_vocab_size=Config.VOCAB_SIZE):
        """构建词汇表"""
        # 统计词频
        word_counts = Counter()
        for text in texts:
            words = text.split()
            word_counts.update(words)
        
        # 选择最常见的词
        most_common = word_counts.most_common(max_vocab_size - len(self.special_tokens))
        
        # 构建词汇表
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens.keys())}
        current_idx = len(self.special_tokens)
        
        for word, _ in most_common:
            self.vocab[word] = current_idx
            current_idx += 1
        
        self.vocab_size = len(self.vocab)
        return self.vocab
    
    def text_to_ids(self, text, max_length=Config.MAX_SEQ_LEN):
        """将文本转换为ID序列"""
        words = text.split()
        # 添加BOS和EOS标记
        ids = [self.special_tokens['<BOS>']]
        
        for word in words:
            if word in self.vocab:
                ids.append(self.vocab[word])
            else:
                ids.append(self.special_tokens['<UNK>'])
        
        ids.append(self.special_tokens['<EOS>'])
        
        # 填充或截断到固定长度
        if len(ids) > max_length:
            ids = ids[:max_length-1] + [self.special_tokens['<EOS>']]
        else:
            ids.extend([self.special_tokens['<PAD>']] * (max_length - len(ids)))
        
        return ids[:max_length]
    
    def ids_to_text(self, ids):
        """将ID序列转换回文本"""
        # 反转词汇表
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        words = []
        for idx in ids:
            if idx in self.special_tokens.values():
                if idx == self.special_tokens['<EOS>']:
                    break
                elif idx not in [self.special_tokens['<BOS>'], self.special_tokens['<PAD>']]:
                    words.append(f'<{list(self.special_tokens.keys())[list(self.special_tokens.values()).index(idx)]}>')
            else:
                words.append(reverse_vocab.get(idx, '<UNK>'))
        
        return ' '.join(words)
    
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
    
    def prepare_dataset(self, texts, build_vocab=True):
        """准备训练数据集"""
        # 清洗文本
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # 构建词汇表（可选）
        if build_vocab:
            self.build_vocab(cleaned_texts)
        
        # 转换为ID序列
        sequences = [self.text_to_ids(text) for text in cleaned_texts]
        
        # 创建输入和目标序列（用于语言模型训练）
        inputs = []
        targets = []
        
        for seq in sequences:
            # 输入是序列的前n-1个token
            inputs.append(seq[:-1])
            # 目标是序列的后n-1个token
            targets.append(seq[1:])
        
        return np.array(inputs), np.array(targets)