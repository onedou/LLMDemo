"""
模型推理模块
负责加载训练好的模型并进行文本生成
"""

import torch
import numpy as np
from model import SimpleLLM
from data_preprocessor import DataPreprocessor
from config import Config


class LLMInference:
    """LLM推理器"""
    
    def __init__(self, model_path=None, vocab_path=None):
        self.model = SimpleLLM()
        self.data_preprocessor = DataPreprocessor()
        
        if model_path and vocab_path:
            self.load_model(model_path, vocab_path)
    
    def load_model(self, model_path, vocab_path):
        """加载训练好的模型"""
        print(f"加载模型: {model_path}")
        
        # 加载词汇表
        self.data_preprocessor.load_vocab(vocab_path)
        
        # 加载模型
        self.model.load_model(model_path)
        self.model.data_preprocessor = self.data_preprocessor
        
        print(f"模型加载完成，词汇表大小: {self.data_preprocessor.vocab_size}")
        print(f"模型配置: {self.model.config}")
    
    def generate_text(self, prompt, max_length=50, temperature=0.8, 
                     top_k=50, num_return_sequences=1):
        """生成文本"""
        if not self.model.data_preprocessor:
            raise ValueError("请先加载模型和词汇表")
        
        print(f"输入提示: {prompt}")
        print(f"生成参数: max_length={max_length}, temperature={temperature}, top_k={top_k}")
        
        results = []
        
        for i in range(num_return_sequences):
            generated_text = self.model.model.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                data_preprocessor=self.data_preprocessor
            )
            
            # 清理生成的文本（移除特殊标记）
            cleaned_text = self._clean_generated_text(generated_text)
            results.append(cleaned_text)
            
            print(f"\n生成结果 {i+1}:")
            print(f"原始: {generated_text}")
            print(f"清理后: {cleaned_text}")
        
        return results
    
    def _clean_generated_text(self, text):
        """清理生成的文本，移除特殊标记"""
        # 移除特殊标记
        special_tokens = ['<BOS>', '<EOS>', '<PAD>', '<UNK>']
        for token in special_tokens:
            text = text.replace(f'<{token}>', '')
        
        # 规范化空格
        text = ' '.join(text.split())
        
        return text.strip()
    
    def calculate_perplexity(self, text):
        """计算文本的困惑度"""
        if not self.model.data_preprocessor:
            raise ValueError("请先加载模型和词汇表")
        
        self.model.model.eval()
        
        # 将文本转换为ID
        input_ids = self.data_preprocessor.text_to_ids(text)
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.model.device)
        
        # 创建目标序列（向前移动一个位置）
        target_ids = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1]
        
        with torch.no_grad():
            # 前向传播
            outputs = self.model.model(input_ids)
            
            # 计算损失
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.data_preprocessor.special_tokens['<PAD>'])
            loss = loss_fn(
                outputs.view(-1, outputs.size(-1)), 
                target_ids.view(-1)
            )
            
            # 计算困惑度
            perplexity = torch.exp(loss).item()
        
        return perplexity
    
    def interactive_generation(self):
        """交互式文本生成"""
        print("\n=== LLM交互式文本生成 ===")
        print("输入 'quit' 退出交互")
        print("输入 'settings' 调整生成参数")
        
        # 默认参数
        max_length = 50
        temperature = 0.8
        top_k = 50
        
        while True:
            try:
                prompt = input("\n请输入提示文本: ").strip()
                
                if prompt.lower() == 'quit':
                    break
                elif prompt.lower() == 'settings':
                    max_length, temperature, top_k = self._get_generation_settings()
                    continue
                elif not prompt:
                    print("提示文本不能为空")
                    continue
                
                # 生成文本
                results = self.generate_text(
                    prompt=prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k
                )
                
                # 计算困惑度
                perplexity = self.calculate_perplexity(prompt)
                print(f"输入文本困惑度: {perplexity:.2f}")
                
            except KeyboardInterrupt:
                print("\n退出交互")
                break
            except Exception as e:
                print(f"生成过程中出现错误: {e}")
    
    def _get_generation_settings(self):
        """获取生成参数设置"""
        print("\n=== 生成参数设置 ===")
        
        try:
            max_length = int(input("最大生成长度 (默认50): ") or 50)
            temperature = float(input("温度参数 (默认0.8): ") or 0.8)
            top_k = int(input("Top-k采样 (默认50): ") or 50)
            
            print(f"参数已更新: max_length={max_length}, temperature={temperature}, top_k={top_k}")
            return max_length, temperature, top_k
            
        except ValueError:
            print("输入参数无效，使用默认值")
            return 50, 0.8, 50
    
    def batch_generate(self, prompts, **kwargs):
        """批量生成文本"""
        results = []
        
        for i, prompt in enumerate(prompts):
            print(f"处理第 {i+1}/{len(prompts)} 个提示...")
            
            try:
                generated = self.generate_text(prompt, **kwargs)
                results.append({
                    'prompt': prompt,
                    'generated_text': generated[0] if generated else ''
                })
            except Exception as e:
                print(f"处理提示 '{prompt}' 时出错: {e}")
                results.append({
                    'prompt': prompt,
                    'generated_text': '',
                    'error': str(e)
                })
        
        return results
    
    def get_model_info(self):
        """获取模型信息"""
        info = {
            'vocab_size': self.data_preprocessor.vocab_size if self.data_preprocessor else '未加载',
            'model_config': self.model.config if self.model else '未加载',
            'device': str(self.model.device) if self.model else '未加载',
            'special_tokens': self.data_preprocessor.special_tokens if self.data_preprocessor else '未加载'
        }
        
        return info