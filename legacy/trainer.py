"""
模型训练模块
负责模型的训练、验证和评估
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import time
import os
from config import Config
from legacy.model import SimpleLLM, TransformerLM
from legacy.data_preprocessor import DataPreprocessor


class LLMTrainer:
    """LLM训练器"""
    
    def __init__(self, model=None, data_preprocessor=None):
        self.model = model or SimpleLLM()
        self.data_preprocessor = data_preprocessor or DataPreprocessor()
        
        # 训练参数
        self.batch_size = Config.BATCH_SIZE
        self.learning_rate = Config.LEARNING_RATE
        self.num_epochs = Config.NUM_EPOCHS
        self.warmup_steps = Config.WARMUP_STEPS
        self.train_ratio = Config.TRAIN_RATIO
        self.val_ratio = Config.VAL_RATIO
        self.vocab_save_path = Config.VOCAB_SAVE_PATH
        
        # 优化器和损失函数
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.data_preprocessor.special_tokens['<PAD>'])
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_perplexities = []
        self.val_perplexities = []
    
    def prepare_data_loader(self, texts, build_vocab=False):
        """准备数据加载器"""
        # 准备数据集
        inputs, targets = self.data_preprocessor.prepare_dataset(texts, build_vocab=build_vocab)
        
        # 转换为PyTorch张量
        inputs_tensor = torch.tensor(inputs, dtype=torch.long)
        targets_tensor = torch.tensor(targets, dtype=torch.long)
        
        # 创建数据集
        dataset = TensorDataset(inputs_tensor, targets_tensor)
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0
        )
        
        return dataloader
    
    def split_data(self, texts):
        """分割数据集（先打乱，避免验证/测试集全是同一类数据）"""
        texts = list(texts)
        random.Random(42).shuffle(texts)

        total_size = len(texts)
        train_size = int(total_size * self.train_ratio)
        val_size = int(total_size * self.val_ratio)

        train_texts = texts[:train_size]
        val_texts = texts[train_size:train_size + val_size]
        test_texts = texts[train_size + val_size:]

        return train_texts, val_texts, test_texts
    
    def get_learning_rate(self, step):
        """学习率调度（带warmup）"""
        if step < self.warmup_steps:
            # Warmup阶段：线性增加学习率
            return self.learning_rate * (step / self.warmup_steps)
        else:
            # 正常训练阶段：使用平方根衰减
            return self.learning_rate * (self.warmup_steps ** 0.5) / (step ** 0.5)
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.model.train()
        total_loss = 0
        total_tokens = 0
        
        progress_bar = tqdm(train_loader, desc="训练")
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            # 移动到设备
            inputs = inputs.to(self.model.device)
            targets = targets.to(self.model.device)
            
            # 前向传播
            outputs = self.model.model(inputs)
            
            # 计算损失
            loss = self.criterion(
                outputs.view(-1, outputs.size(-1)), 
                targets.view(-1)
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            
            # 更新学习率
            current_step = batch_idx + len(self.train_losses) * len(train_loader)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.get_learning_rate(current_step)
            
            # 统计
            total_loss += loss.item() * inputs.size(0)
            total_tokens += (targets != self.data_preprocessor.special_tokens['<PAD>']).sum().item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        avg_loss = total_loss / len(train_loader.dataset)
        perplexity = np.exp(avg_loss)
        
        return avg_loss, perplexity
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.model.device)
                targets = targets.to(self.model.device)
                
                outputs = self.model.model(inputs)
                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)), 
                    targets.view(-1)
                )
                
                total_loss += loss.item() * inputs.size(0)
        
        avg_loss = total_loss / len(val_loader.dataset)
        perplexity = np.exp(avg_loss)
        
        return avg_loss, perplexity
    
    def train(self, texts, save_path=None):
        """完整训练流程"""
        print("开始训练LLM模型...")
        
        # 分割数据
        train_texts, val_texts, test_texts = self.split_data(texts)
        
        print(f"训练集大小: {len(train_texts)}")
        print(f"验证集大小: {len(val_texts)}")
        print(f"测试集大小: {len(test_texts)}")
        
        # 先构建词汇表（但不创建数据加载器）
        self.data_preprocessor.build_vocab(train_texts)
        actual_vocab_size = self.data_preprocessor.vocab_size
        print(f"实际词汇表大小: {actual_vocab_size}")
        
        # 使用实际的词汇表大小重新创建模型
        self.model.config['vocab_size'] = actual_vocab_size
        self.model.model = TransformerLM(**self.model.config)
        self.model.model.to(self.model.device)
        
        # 确保模型关联了数据预处理器
        self.model.data_preprocessor = self.data_preprocessor
        
        # 准备数据加载器（在模型创建之后）
        train_loader = self.prepare_data_loader(train_texts)
        val_loader = self.prepare_data_loader(val_texts) if val_texts else None
        
        # 初始化优化器
        self.optimizer = torch.optim.AdamW(
            self.model.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        # 训练循环
        best_loss = float('inf')

        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            # 训练
            start_time = time.time()
            train_loss, train_ppl = self.train_epoch(train_loader)
            train_time = time.time() - start_time

            # 记录历史
            self.train_losses.append(train_loss)
            self.train_perplexities.append(train_ppl)
            print(f"训练损失: {train_loss:.4f}, 训练困惑度: {train_ppl:.2f}")

            # 验证（验证集为空时按训练损失保存）
            if val_loader is not None:
                val_loss, val_ppl = self.validate(val_loader)
                self.val_losses.append(val_loss)
                self.val_perplexities.append(val_ppl)
                print(f"验证损失: {val_loss:.4f}, 验证困惑度: {val_ppl:.2f}")
                current_loss = val_loss
            else:
                current_loss = train_loss

            print(f"训练时间: {train_time:.2f}秒")

            # 保存最佳模型
            if current_loss < best_loss:
                best_loss = current_loss
                if save_path:
                    # 确保目录存在
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    self.model.save_model(save_path)
                    self.data_preprocessor.save_vocab(self.vocab_save_path)
                    print(f"保存最佳模型到: {save_path}")
        
        print("\n训练完成!")
        
        # 测试模型
        if len(test_texts) > 0:
            test_loader = self.prepare_data_loader(test_texts)
            test_loss, test_ppl = self.validate(test_loader)
            print(f"测试损失: {test_loss:.4f}, 测试困惑度: {test_ppl:.2f}")
        
        return self.model
    
    def plot_training_history(self):
        """绘制训练历史（需要matplotlib）"""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 4))
            
            # 损失图
            plt.subplot(1, 2, 1)
            plt.plot(self.train_losses, label='训练损失')
            plt.plot(self.val_losses, label='验证损失')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('训练和验证损失')
            
            # 困惑度图
            plt.subplot(1, 2, 2)
            plt.plot(self.train_perplexities, label='训练困惑度')
            plt.plot(self.val_perplexities, label='验证困惑度')
            plt.xlabel('Epoch')
            plt.ylabel('Perplexity')
            plt.legend()
            plt.title('训练和验证困惑度')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("未安装matplotlib，无法绘制训练历史图")
    
    def evaluate_model(self, texts):
        """评估模型性能"""
        self.model.model.eval()
        
        # 准备数据
        dataloader = self.prepare_data_loader(texts)
        
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.model.device)
                targets = targets.to(self.model.device)
                
                outputs = self.model.model(inputs)
                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)), 
                    targets.view(-1)
                )
                
                total_loss += loss.item() * inputs.size(0)
                total_tokens += (targets != self.data_preprocessor.special_tokens['<PAD>']).sum().item()
        
        avg_loss = total_loss / len(dataloader.dataset)
        perplexity = np.exp(avg_loss)
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'total_samples': len(dataloader.dataset),
            'total_tokens': total_tokens
        }