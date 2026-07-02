# LLM演示程序

一个基于PyTorch的简单语言模型演示程序，展示了完整的LLM训练和推理流程。

## 功能特性

- ✅ **自定义模型训练**: 支持从头开始训练Transformer语言模型
- ✅ **模型加载和推理**: 可以加载训练好的模型进行文本生成
- ✅ **交互式界面**: 提供命令行和Web两种交互方式
- ✅ **完整流程**: 包含数据预处理、模型训练、推理评估的全流程
- ✅ **配置灵活**: 支持自定义模型参数和训练超参数

## 项目结构

```
LLMDemo/
├── main.py                 # 主程序入口
├── config.py              # 配置文件
├── requirements.txt       # 依赖包列表
├── data_preprocessor.py   # 数据预处理模块
├── model.py               # 模型架构定义
├── trainer.py             # 训练模块
├── inference.py           # 推理模块
├── interface.py           # 交互界面模块
├── web_server.py          # Web聊天服务器
├── static/                # Web前端页面
│   └── index.html         # 聊天页面
├── data/                  # 数据目录
│   └── sample_data.txt    # 示例数据集
├── models/                # 模型保存目录
└── README.md              # 项目说明
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 快速演示

```bash
python main.py demo
```

### 3. 训练模型

```bash
# 使用默认参数训练
python main.py train

# 自定义训练参数
python interface.py train --data data/sample_data.txt --epochs 10 --batch_size 32
```

### 4. 使用模型生成文本

```bash
# 单次生成
python interface.py infer --prompt "人工智能是" --max_length 50

# 交互式生成
python interface.py interactive
```

### 5. Web界面

启动Web聊天服务器，通过浏览器与Nova对话（支持中文和英文输入）：

```bash
python web_server.py
```

启动后在浏览器中访问 [http://127.0.0.1:8000](http://127.0.0.1:8000) 即可开始聊天。

也可以直接调用聊天接口：

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "你好"}'
```

## 详细使用说明

### 数据预处理

程序包含一个数据预处理器，支持：
- 文本清洗和规范化
- 词汇表构建
- 序列编码和解码
- 训练/验证/测试集分割

### 模型架构

基于Transformer的简化语言模型：
- 词嵌入层 + 位置编码
- 多层Transformer编码器
- 因果注意力掩码
- 可配置的模型参数

### 训练流程

训练器提供完整的训练功能：
- 学习率调度（带warmup）
- 梯度裁剪
- 训练历史记录
- 自动保存最佳模型
- 训练过程可视化

### 推理功能

推理器支持多种生成方式：
- 单次文本生成
- 批量生成
- 交互式生成
- 困惑度计算
- Top-k采样和温度控制

## 配置参数

在`config.py`中可以修改以下参数：

### 模型参数
- `VOCAB_SIZE`: 词汇表大小
- `D_MODEL`: 模型维度
- `N_HEAD`: 注意力头数
- `NUM_LAYERS`: Transformer层数
- `MAX_SEQ_LEN`: 最大序列长度

### 训练参数
- `BATCH_SIZE`: 批次大小
- `LEARNING_RATE`: 学习率
- `NUM_EPOCHS`: 训练轮数
- `WARMUP_STEPS`: warmup步数

### 文件路径
- `MODEL_SAVE_PATH`: 模型保存路径
- `VOCAB_SAVE_PATH`: 词汇表保存路径
- `DATA_PATH`: 训练数据路径

## 示例数据集

程序包含一个包含技术相关文本的示例数据集，涵盖：
- 人工智能和机器学习
- 计算机科学基础概念
- 新兴技术领域
- 学术和行业术语

## 扩展建议

1. **更大数据集**: 使用更大规模的中文或英文文本数据
2. **模型优化**: 实现更复杂的注意力机制和位置编码
3. **多任务学习**: 支持文本分类、情感分析等任务
4. **部署优化**: 添加模型量化、推理优化等功能
5. **可视化增强**: 添加训练过程实时监控界面

## 技术栈

- **框架**: PyTorch 2.0+
- **数据处理**: NumPy
- **进度显示**: tqdm
- **Web界面**: Flask（可选）
- **可视化**: Matplotlib（可选）

## 注意事项

1. 首次运行时会自动创建必要的目录结构
2. 训练过程需要一定的计算资源，建议使用GPU加速
3. 示例数据集较小，实际应用时需要更大规模的数据
4. 模型参数可以根据硬件条件进行调整

## 许可证

本项目仅用于学习和演示目的。