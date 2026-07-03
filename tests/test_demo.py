#!/usr/bin/env python3
"""
LLM演示程序测试脚本
用于快速验证程序功能
"""

import os
import sys
import torch
import os
import sys
# 使脚本可从任意目录直接运行（把项目根目录加入模块搜索路径）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from legacy.data_preprocessor import DataPreprocessor
from legacy.model import SimpleLLM


def test_data_preprocessing():
    """测试数据预处理功能"""
    print("=== 测试数据预处理 ===")
    
    # 创建预处理器
    preprocessor = DataPreprocessor()
    
    # 测试文本
    test_texts = [
        "Hello, this is a test sentence!",
        "Machine learning is amazing.",
        "The quick brown fox jumps over the lazy dog."
    ]
    
    # 测试清洗
    print("原始文本:")
    for text in test_texts:
        print(f"  {text}")
    
    cleaned_texts = [preprocessor.clean_text(text) for text in test_texts]
    print("\n清洗后文本:")
    for text in cleaned_texts:
        print(f"  {text}")
    
    # 测试词汇表构建
    vocab = preprocessor.build_vocab(cleaned_texts, max_vocab_size=100)
    print(f"\n词汇表大小: {len(vocab)}")
    print("特殊标记:")
    for token, idx in preprocessor.special_tokens.items():
        print(f"  {token}: {idx}")
    
    # 测试编码解码
    test_text = "machine learning is amazing"
    ids = preprocessor.text_to_ids(test_text)
    decoded_text = preprocessor.ids_to_text(ids)
    
    print(f"\n编码解码测试:")
    print(f"  原始: {test_text}")
    print(f"  编码: {ids[:10]}...")  # 只显示前10个ID
    print(f"  解码: {decoded_text}")
    
    print("✓ 数据预处理测试通过\n")


def test_model_creation():
    """测试模型创建功能"""
    print("=== 测试模型创建 ===")
    
    # 创建简化模型
    model = SimpleLLM({
        'vocab_size': 100,
        'd_model': 128,
        'n_head': 4,
        'num_layers': 2,
        'd_ff': 256
    })
    
    print(f"模型设备: {model.device}")
    print(f"模型配置: {model.config}")
    
    # 测试前向传播
    batch_size = 2
    seq_len = 10
    
    # 创建模拟输入
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    
    # 前向传播
    outputs = model.model(input_ids)
    
    print(f"输入形状: {input_ids.shape}")
    print(f"输出形状: {outputs.shape}")
    
    # 测试参数数量
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"模型参数总数: {total_params:,}")
    
    print("✓ 模型创建测试通过\n")


def test_training_preparation():
    """测试训练准备功能"""
    print("=== 测试训练准备 ===")
    
    from legacy.trainer import LLMTrainer
    
    # 创建训练器
    trainer = LLMTrainer()
    
    # 测试数据分割
    test_texts = [f"text {i}" for i in range(100)]
    train_texts, val_texts, test_texts = trainer.split_data(test_texts)
    
    print(f"训练集大小: {len(train_texts)}")
    print(f"验证集大小: {len(val_texts)}")
    print(f"测试集大小: {len(test_texts)}")
    
    # 测试数据加载器准备
    dataloader = trainer.prepare_data_loader(test_texts[:10])
    
    # 检查一个批次
    for inputs, targets in dataloader:
        print(f"输入形状: {inputs.shape}")
        print(f"目标形状: {targets.shape}")
        break  # 只检查第一个批次
    
    # 测试学习率调度
    learning_rates = []
    for step in range(0, 2000, 100):
        lr = trainer.get_learning_rate(step)
        learning_rates.append((step, lr))
    
    print(f"学习率调度测试: 步数={len(learning_rates)}")
    
    print("✓ 训练准备测试通过\n")


def test_inference():
    """测试推理功能"""
    print("=== 测试推理功能 ===")
    
    from legacy.inference import LLMInference
    
    # 创建推理器
    inference = LLMInference()
    
    # 测试模型信息获取
    model_info = inference.get_model_info()
    print("模型信息:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # 测试文本清理
    test_text = "hello <BOS> this is a test <EOS> <PAD>"
    cleaned = inference._clean_generated_text(test_text)
    print(f"\n文本清理测试:")
    print(f"  原始: {test_text}")
    print(f"  清理后: {cleaned}")
    
    print("✓ 推理功能测试通过\n")


def test_configuration():
    """测试配置功能"""
    print("=== 测试配置功能 ===")
    
    print("配置参数:")
    config_vars = [var for var in dir(Config) if not var.startswith('_')]
    
    for var in config_vars:
        value = getattr(Config, var)
        print(f"  {var}: {value}")
    
    print("✓ 配置功能测试通过\n")


def main():
    """主测试函数"""
    print("LLM演示程序功能测试\n")
    
    try:
        # 检查依赖
        import torch
        import numpy
        import tqdm
        print("✓ 所有依赖已安装")
    except ImportError as e:
        print(f"✗ 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        return
    
    # 运行测试
    tests = [
        test_configuration,
        test_data_preprocessing,
        test_model_creation,
        test_training_preparation,
        test_inference
    ]
    
    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"✗ {test_func.__name__} 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("=== 测试完成 ===")
    print("\n下一步:")
    print("1. 运行完整训练: python main.py train")
    print("2. 快速演示: python main.py demo")
    print("3. 交互式生成: python main.py interactive")


if __name__ == "__main__":
    main()