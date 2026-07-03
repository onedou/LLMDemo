#!/usr/bin/env python3
"""
使用对话数据重新训练LLM模型，增强对话能力

对话数据以问答对形式训练：<BOS> 问题 <SEP> 回答 <EOS>
这样模型才能学到"问题→回答"的映射，而不是简单的逐句续写
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from legacy.trainer import LLMTrainer
from legacy.data_preprocessor import DataPreprocessor
from legacy.model import SimpleLLM
from config import Config


def load_conversation_pairs(filepath='data/conversation_data.txt'):
    """解析对话数据为(问题, 回答)对

    文件格式：空行分隔的块，每块第一行是问题，其余行是回答
    """
    pairs = []
    if not os.path.exists(filepath):
        return pairs

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    for block in content.split('\n\n'):
        lines = [line.strip() for line in block.strip().split('\n') if line.strip()]
        if len(lines) >= 2:
            question = lines[0]
            answer = ' '.join(lines[1:])
            pairs.append((question, answer))

    return pairs


def load_plain_texts(filepath=Config.DATA_PATH):
    """加载通用文本数据（普通语言建模样本）"""
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def train_conversation_model():
    """训练具有对话能力的模型"""
    print("=== 训练对话增强型LLM模型 ===")

    # 加载数据
    pairs = load_conversation_pairs()
    plain_texts = load_plain_texts()

    # 数据增强：问题的无标点/全小写变体也训练一遍，容忍用户随意的输入习惯
    seen = {q for q, _ in pairs}
    augmented = []
    for q, a in pairs:
        stripped = q.rstrip(' ?!.？！。')
        for variant in (stripped, q.lower(), stripped.lower()):
            if variant and variant not in seen:
                seen.add(variant)
                augmented.append((variant, a))

    items = plain_texts + pairs + augmented

    print(f"加载了 {len(pairs)} 组问答对（增强后 {len(pairs) + len(augmented)} 组），{len(plain_texts)} 条通用文本")
    if not items:
        print("❌ 没有找到训练数据")
        return

    # 创建数据预处理器（对话模式：保留大小写）
    data_preprocessor = DataPreprocessor(preserve_case=True)

    # 数据量小，使用小模型（大模型在几百条数据上只会严重过拟合）
    model = SimpleLLM(Config.CONV_MODEL_CONFIG)

    # 创建训练器
    trainer = LLMTrainer(model=model, data_preprocessor=data_preprocessor)

    # 训练参数：演示场景数据极少，全部数据用于训练，让模型充分记住问答对
    trainer.num_epochs = 400
    trainer.batch_size = 8
    trainer.learning_rate = 3e-4
    trainer.warmup_steps = 100  # 默认1000步warmup对小数据集来说太长，大部分训练都在极低学习率
    trainer.train_ratio = 1.0
    trainer.val_ratio = 0.0

    # 模型保存路径
    model_save_path = 'models/conversation_llm_model.pth'
    trainer.vocab_save_path = 'models/conversation_vocab.pkl'

    # 开始训练
    print(f"开始训练，共{len(items)}条数据，{trainer.num_epochs}轮")
    trainer.train(items, model_save_path)

    print(f"\n最终训练损失: {trainer.train_losses[-1]:.4f}")
    print(f"✅ 模型训练完成，保存到: {model_save_path}")
    print(f"✅ 词汇表保存到: {trainer.vocab_save_path}")


if __name__ == "__main__":
    train_conversation_model()
