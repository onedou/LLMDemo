"""
LLM模型架构模块
基于Transformer的简单语言模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import Config


class PositionalEncoding(nn.Module):
    """位置编码层"""
    
    def __init__(self, d_model, max_len=Config.MAX_SEQ_LEN):
        super(PositionalEncoding, self).__init__()
        
        # 计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """前向传播"""
        return x + self.pe[:x.size(0), :]


class TransformerLM(nn.Module):
    """基于Transformer的语言模型"""
    
    def __init__(self, vocab_size=Config.VOCAB_SIZE, d_model=Config.D_MODEL, 
                 n_head=Config.N_HEAD, num_layers=Config.NUM_LAYERS, 
                 d_ff=Config.D_FF, dropout=Config.DROPOUT, max_len=Config.MAX_SEQ_LEN):
        super(TransformerLM, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # 词嵌入层
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # 输出层
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None):
        """前向传播"""
        batch_size, seq_len = input_ids.size()
        
        # 词嵌入
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # Dropout
        x = self.dropout(x)
        
        # 创建注意力掩码（因果掩码）
        if attention_mask is None:
            # 创建因果掩码，防止模型看到未来的token
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len) * float('-inf'), 
                diagonal=1
            ).to(input_ids.device)
        else:
            causal_mask = attention_mask
        
        # Transformer编码器
        x = self.transformer_encoder(x, mask=causal_mask)
        
        # 输出层
        logits = self.output_layer(x)
        
        return logits
    
    def generate(self, prompt, max_length=50, temperature=1.0, top_k=50, 
                data_preprocessor=None):
        """文本生成方法"""
        self.eval()
        
        if data_preprocessor is None:
            raise ValueError("需要提供数据预处理器")
        
        # 将提示转换为ID
        input_ids = data_preprocessor.text_to_ids(prompt)
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(next(self.parameters()).device)
        
        generated_ids = input_ids.tolist()[0]
        
        with torch.no_grad():
            for _ in range(max_length):
                # 前向传播
                outputs = self(input_ids)
                
                # 获取最后一个token的logits
                next_token_logits = outputs[0, -1, :] / temperature
                
                # Top-k过滤
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][-1]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # 应用softmax获取概率
                probs = F.softmax(next_token_logits, dim=-1)
                
                # 从分布中采样
                next_token_id = torch.multinomial(probs, num_samples=1).item()
                
                # 如果生成了EOS标记，停止生成
                if next_token_id == data_preprocessor.special_tokens['<EOS>']:
                    break
                
                # 添加新token到序列
                generated_ids.append(next_token_id)
                
                # 更新输入
                input_ids = torch.tensor([generated_ids[-Config.MAX_SEQ_LEN:]]).to(next(self.parameters()).device)
        
        # 将ID转换回文本
        generated_text = data_preprocessor.ids_to_text(generated_ids)
        
        return generated_text


class SimpleLLM:
    """简化的LLM包装类"""
    
    def __init__(self, model_config=None):
        if model_config is None:
            model_config = {}
        
        self.config = {
            'vocab_size': model_config.get('vocab_size', Config.VOCAB_SIZE),
            'd_model': model_config.get('d_model', Config.D_MODEL),
            'n_head': model_config.get('n_head', Config.N_HEAD),
            'num_layers': model_config.get('num_layers', Config.NUM_LAYERS),
            'd_ff': model_config.get('d_ff', Config.D_FF),
            'dropout': model_config.get('dropout', Config.DROPOUT)
        }
        
        self.model = TransformerLM(**self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.data_preprocessor = None
    
    def to(self, device):
        """移动模型到指定设备"""
        self.device = device
        self.model.to(device)
        return self
    
    def save_model(self, filepath):
        """保存模型"""
        # 更新配置中的词汇表大小为实际值
        if self.data_preprocessor:
            self.config['vocab_size'] = self.data_preprocessor.vocab_size
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'vocab': self.data_preprocessor.vocab if self.data_preprocessor else None
        }, filepath)
    
    def load_model(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 更新配置，但使用实际词汇表大小
        self.config.update(checkpoint['config'])
        
        # 如果检查点中有词汇表信息，使用实际的词汇表大小
        if checkpoint.get('vocab'):
            actual_vocab_size = len(checkpoint['vocab'])
            self.config['vocab_size'] = actual_vocab_size
        
        # 重新创建模型
        self.model = TransformerLM(**self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        # 加载词汇表
        if checkpoint.get('vocab') and self.data_preprocessor:
            self.data_preprocessor.vocab = checkpoint['vocab']
            self.data_preprocessor.vocab_size = len(checkpoint['vocab'])
        
        return self