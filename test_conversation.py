#!/usr/bin/env python3
"""
测试对话增强型LLM模型
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import LLMInference

def test_basic_conversation():
    """测试基本对话能力"""
    print("=== 测试对话能力 ===")
    
    # 模型路径
    model_path = 'models/conversation_llm_model.pth'
    vocab_path = 'models/conversation_vocab.pkl'
    
    # 如果对话模型不存在，使用默认模型
    if not os.path.exists(model_path):
        model_path = 'models/llm_demo_model.pth'
        vocab_path = 'models/vocab.pkl'
        print("⚠️  对话模型不存在，使用默认模型")
    
    # 加载模型
    inference = LLMInference(model_path, vocab_path)
    
    # 测试对话
    test_dialogues = [
        "Hello",
        "How are you?",
        "What's your name?",
        "Can you help me?",
        "Tell me a joke",
        "What time is it?",
        "How does machine learning work?",
        "What's the weather like?"
    ]
    
    print("\n🤖 开始对话测试...")
    print("-" * 50)
    
    for prompt in test_dialogues:
        print(f"👤 用户: {prompt}")
        
        try:
            results = inference.generate_text(
                prompt=prompt,
                max_length=30,
                temperature=0.9,
                top_k=50,
                is_conversation=True
            )
            
            if results:
                print(f"🤖 AI: {results[0]}")
            else:
                print("🤖 AI: 抱歉，我无法生成回复")
            
        except Exception as e:
            print(f"🤖 AI: 生成回复时出错: {e}")
        
        print("-" * 50)

def interactive_demo():
    """交互式演示"""
    print("\n=== 交互式对话演示 ===")
    
    # 模型路径
    model_path = 'models/conversation_llm_model.pth'
    vocab_path = 'models/conversation_vocab.pkl'
    
    # 如果对话模型不存在，使用默认模型
    if not os.path.exists(model_path):
        model_path = 'models/llm_demo_model.pth'
        vocab_path = 'models/vocab.pkl'
        print("⚠️  对话模型不存在，使用默认模型")
    
    # 加载模型
    inference = LLMInference(model_path, vocab_path)
    
    # 开始交互
    inference.interactive_generation()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_demo()
    else:
        test_basic_conversation()
        
    print("\n💡 提示: 要启动对话模型训练，请运行: python train_with_conversation.py")
    print("💡 提示: 要进入交互模式，请运行: python test_conversation.py interactive")