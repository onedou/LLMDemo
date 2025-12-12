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