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
                     top_k=50, num_return_sequences=1, is_conversation=True):
        """生成文本"""
        if not self.model.data_preprocessor:
            raise ValueError("请先加载模型和词汇表")
        
        print(f"用户输入: {prompt}")
        print(f"生成参数: max_length={max_length}, temperature={temperature}, top_k={top_k}")
        
        results = []
        
        for i in range(num_return_sequences):
            # 如果是对话模式，优化生成参数
            if is_conversation:
                # 对话回复通常更短、更有温度
                conversation_max_length = min(max_length, 30)
                conversation_temperature = min(temperature + 0.1, 1.0)
                
                generated_text = self.model.model.generate(
                    prompt=prompt,
                    max_length=conversation_max_length,
                    temperature=conversation_temperature,
                    top_k=top_k,
                    data_preprocessor=self.data_preprocessor
                )
            else:
                generated_text = self.model.model.generate(
                    prompt=prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    data_preprocessor=self.data_preprocessor
                )
            
            # 清理生成的文本（移除特殊标记）
            cleaned_text = self._clean_generated_text(generated_text)
            
            # 对话模式下的后处理
            if is_conversation:
                cleaned_text = self._conversation_postprocess(cleaned_text, prompt)
            
            results.append(cleaned_text)
            
            print(f"\nAI回复 {i+1}:")
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
    
    def _conversation_postprocess(self, generated_text, original_prompt):
        """对话模式下的后处理"""
        # 移除重复的问候语
        greetings = ['hello', 'hi', 'hey', 'greetings']
        for greeting in greetings:
            if generated_text.lower().startswith(greeting) and original_prompt.lower().startswith(greeting):
                # 找到第一个句号或换行
                sentences = generated_text.split('.', 1)
                if len(sentences) > 1:
                    generated_text = sentences[1].strip()
                break
        
        # 确保回复不以句号结尾（如果长度很短）
        if len(generated_text.split()) <= 5 and generated_text.endswith('.'):
            generated_text = generated_text[:-1]
        
        # 如果回复太短，尝试重新生成
        if len(generated_text) < 3:
            generated_text = "I'm here to help! What can I assist you with today?"
        
        # 确保回复是完整的句子
        if not generated_text.endswith(('.', '!', '?')):
            generated_text = generated_text + '.'
        
        # 首字母大写
        if generated_text and generated_text[0].islower():
            generated_text = generated_text[0].upper() + generated_text[1:]
        
        return generated_text
    
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
        """交互式对话生成"""
        print("\n=== AI对话助手 ===")
        print("现在您可以与AI进行自然对话了！")
        print("输入 'quit' 退出对话")
        print("输入 'settings' 调整生成参数")
        print("输入 'clear' 清空对话历史")
        print("-" * 50)
        
        # 默认参数（优化对话）
        max_length = 30  # 对话回复更短
        temperature = 0.9  # 对话更有创意
        top_k = 50
        
        # 对话历史
        conversation_history = []
        
        while True:
            try:
                user_input = input("\n👤 您: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'settings':
                    max_length, temperature, top_k = self._get_generation_settings()
                    continue
                elif user_input.lower() == 'clear':
                    conversation_history = []
                    print("\n🗑️  对话历史已清空")
                    continue
                elif not user_input:
                    print("请输入内容")
                    continue
                
                # 构建上下文提示（包含历史对话）
                if conversation_history:
                    # 使用最近3轮对话作为上下文
                    recent_history = conversation_history[-3:]
                    context_prompt = "\n".join(recent_history) + "\n" + user_input
                else:
                    context_prompt = user_input
                
                # 生成回复
                print("🤖 AI思考中...")
                results = self.generate_text(
                    prompt=context_prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    is_conversation=True
                )
                
                ai_response = results[0] if results else "I'm here to help!"
                
                # 显示AI回复
                print(f"🤖 AI: {ai_response}")
                
                # 保存到对话历史
                conversation_history.append(f"用户: {user_input}")
                conversation_history.append(f"AI: {ai_response}")
                
                # 限制历史长度
                if len(conversation_history) > 10:
                    conversation_history = conversation_history[-10:]
                
            except KeyboardInterrupt:
                print("\n👋 退出对话")
                break
            except Exception as e:
                print(f"❌ 对话过程中出现错误: {e}")
    
    def _get_generation_settings(self):
        """获取生成参数设置"""
        print("\n=== 对话参数设置 ===")
        print("推荐对话设置: 最大长度20-40, 温度0.7-1.0")
        
        try:
            max_length = int(input("最大回复长度 (默认30): ") or 30)
            temperature = float(input("创意程度 (默认0.9): ") or 0.9)
            top_k = int(input("多样性控制 (默认50): ") or 50)
            
            print(f"✅ 参数已更新: 回复长度={max_length}, 创意程度={temperature}, 多样性={top_k}")
            return max_length, temperature, top_k
            
        except ValueError:
            print("❌ 输入参数无效，使用默认对话设置")
            return 30, 0.9, 50
    
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