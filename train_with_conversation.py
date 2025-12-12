#!/usr/bin/env python3
"""
使用对话数据重新训练LLM模型，增强对话能力
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trainer import LLMTrainer
from data_preprocessor import DataPreprocessor
from config import Config

def load_combined_data():
    """加载结合了通用文本和对话数据的训练数据"""
    texts = []
    
    # 加载通用文本数据
    if os.path.exists(Config.DATA_PATH):
        with open(Config.DATA_PATH, 'r', encoding='utf-8') as f:
            texts.extend([line.strip() for line in f if line.strip()])
    
    # 加载对话数据
    conversation_path = 'data/conversation_data.txt'
    if os.path.exists(conversation_path):
        with open(conversation_path, 'r', encoding='utf-8') as f:
            texts.extend([line.strip() for line in f if line.strip()])
    
    print(f"加载了 {len(texts)} 条训练数据")
    print(f"其中包含 {len([t for t in texts if '?' in t or '!' in t or 'Hello' in t])} 条对话数据")
    
    return texts

def train_conversation_model():
    """训练具有对话能力的模型"""
    print("=== 训练对话增强型LLM模型 ===")
    
    # 加载数据
    texts = load_combined_data()
    if not texts:
        print("❌ 没有找到训练数据")
        return
    
    # 创建训练器
    trainer = LLMTrainer()
    
    # 优化训练参数（对话模型需要更多轮次）
    trainer.num_epochs = 100  # 增加训练轮次
    trainer.batch_size = 8    # 减小批次大小以获得更好的收敛
    
    # 模型保存路径
    model_save_path = 'models/conversation_llm_model.pth'
    vocab_save_path = 'models/conversation_vocab.pkl'
    
    # 开始训练
    print(f"开始训练，共{len(texts)}条数据，{trainer.num_epochs}轮")
    trainer.train(texts, model_save_path)
    
    # 保存词汇表
    trainer.data_preprocessor.save_vocab(vocab_save_path)
    
    # 显示训练历史
    trainer.plot_training_history()
    
    print(f"✅ 模型训练完成，保存到: {model_save_path}")
    print(f"✅ 词汇表保存到: {vocab_save_path}")

if __name__ == "__main__":
    train_conversation_model()