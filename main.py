#!/usr/bin/env python3
"""
LLM演示程序主入口
一个完整的语言模型训练和推理演示程序
"""

import os
import sys
import argparse
from interface import CommandLineInterface, SimpleWebInterface, main as interface_main
from config import Config


def check_dependencies():
    """检查依赖是否已安装"""
    try:
        import torch
        import numpy
        print("✓ 核心依赖已安装")
        return True
    except ImportError as e:
        print(f"✗ 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        return False


def create_directories():
    """创建必要的目录"""
    directories = ['models', 'data']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"创建目录: {directory}")


def demo_quick_start():
    """快速开始演示"""
    print("\n=== LLM演示程序快速开始 ===")
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 创建目录
    create_directories()
    
    # 检查示例数据
    if not os.path.exists(Config.DATA_PATH):
        print("示例数据文件已创建")
    
    print("\n快速开始选项:")
    print("1. 训练模型（使用示例数据）")
    print("2. 直接加载预训练模型进行推理")
    print("3. 交互式界面")
    
    try:
        choice = input("请选择 (1-3): ").strip()
        
        if choice == '1':
            # 训练模型
            from trainer import LLMTrainer
            from data_preprocessor import DataPreprocessor
            
            print("\n开始训练模型...")
            
            # 加载数据
            with open(Config.DATA_PATH, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            if not texts:
                print("没有可用的训练数据")
                return
            
            # 训练模型
            trainer = LLMTrainer()
            trainer.train(texts, Config.MODEL_SAVE_PATH)
            
            print(f"\n模型训练完成！保存到: {Config.MODEL_SAVE_PATH}")
            
        elif choice == '2':
            # 直接推理（需要先训练或加载预训练模型）
            from inference import LLMInference
            
            if not os.path.exists(Config.MODEL_SAVE_PATH):
                print("模型文件不存在，请先训练模型")
                return
            
            print("\n加载模型进行推理...")
            inference = LLMInference(Config.MODEL_SAVE_PATH, Config.VOCAB_SAVE_PATH)
            
            prompt = input("请输入提示文本: ")
            results = inference.generate_text(prompt)
            
            print(f"\n生成结果: {results[0]}")
            
        elif choice == '3':
            # 交互式界面
            interface_main()
            
        else:
            print("无效选择")
            
    except KeyboardInterrupt:
        print("\n退出演示")
    except Exception as e:
        print(f"演示过程中出错: {e}")


def show_help():
    """显示帮助信息"""
    print("""
LLM演示程序 - 使用说明

这是一个基于PyTorch的简单语言模型演示程序，包含完整的训练和推理功能。

快速开始:
  python main.py demo          # 快速开始演示
  python main.py train         # 训练模型
  python main.py infer         # 使用模型推理
  python main.py interactive   # 交互式生成

命令行使用:
  python main.py train --data data/sample_data.txt --epochs 10
  python main.py infer --prompt "人工智能是" --max_length 50

模块说明:
  config.py              - 配置文件
  data_preprocessor.py   - 数据预处理模块
  model.py               - 模型架构
  trainer.py             - 训练模块
  inference.py           - 推理模块
  interface.py           - 交互界面

文件结构:
  models/                - 模型保存目录
  data/                  - 数据文件目录
  requirements.txt       - 依赖包列表

示例数据:
  程序包含了一个包含技术相关文本的示例数据集，用于演示训练过程。
""")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LLM演示程序')
    parser.add_argument('command', nargs='?', help='命令: demo, train, infer, interactive, help')
    
    args = parser.parse_args()
    
    if not args.command:
        # 如果没有参数，显示帮助并进入交互模式
        show_help()
        demo_quick_start()
        return
    
    if args.command == 'demo':
        demo_quick_start()
    elif args.command == 'train':
        # 使用命令行界面进行训练
        sys.argv = ['interface.py', 'train']
        interface_main()
    elif args.command == 'infer':
        # 使用命令行界面进行推理
        sys.argv = ['interface.py', 'infer']
        interface_main()
    elif args.command == 'interactive':
        # 交互式界面
        interface_main()
    elif args.command == 'help':
        show_help()
    else:
        print(f"未知命令: {args.command}")
        show_help()


if __name__ == "__main__":
    # 添加当前目录到Python路径
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # 创建必要目录
    create_directories()
    
    # 运行主程序
    main()