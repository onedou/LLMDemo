"""
LLM演示程序配置
"""

class Config:
    """模型配置参数"""
    
    # 模型参数
    VOCAB_SIZE = 10000
    D_MODEL = 512
    N_HEAD = 8
    NUM_LAYERS = 6
    D_FF = 2048
    DROPOUT = 0.1
    MAX_SEQ_LEN = 512
    # 训练样本的填充长度（演示数据都是短句，512会浪费大量计算在<PAD>上）
    TRAIN_MAX_SEQ_LEN = 64

    # 对话模型参数（数据量小，用小模型避免过拟合且加快训练）
    # dropout=0：演示场景目标是精确记住问答对，正则化反而妨碍记忆
    CONV_MODEL_CONFIG = {
        'd_model': 128,
        'n_head': 4,
        'num_layers': 2,
        'd_ff': 512,
        'dropout': 0.0
    }
    
    # 训练参数
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 10
    WARMUP_STEPS = 1000
    
    # 数据参数
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    # 文件路径
    MODEL_SAVE_PATH = "models/llm_demo_model.pth"
    VOCAB_SAVE_PATH = "models/vocab.pkl"
    DATA_PATH = "data/sample_data.txt"