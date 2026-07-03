# GPT-2 预训练迁移包使用说明

本包包含在任何机器上继续预训练所需的全部文件，无需原项目的其他代码。

## 包内容

| 文件 | 说明 |
|------|------|
| `pretrain.py` | 训练脚本（自动选择设备：mps > cuda > cpu） |
| `gpt2_model.py` | GPT-2 模型定义（gpt2-demo 预设 = 33.87M 参数） |
| `bpe_tokenizer.py` | BPE 分词器封装 |
| `generate_gpt2.py` | 训练后的文本生成/测试脚本 |
| `models/bpe_tokenizer/tokenizer.json` | 已训练的 BPE 分词器（16384 词表） |
| `data/pretrain/train.bin` | 训练集（127.2M tokens，uint16） |
| `data/pretrain/val.bin` | 验证集（1.285M tokens） |
| `colab_pretrain.ipynb` | Colab/Kaggle 一键笔记本 |

## 环境要求（唯一需要装的东西）

```bash
pip install torch numpy tokenizers
```

- **另一台 Mac（Apple Silicon）**：上面一条命令即可，训练自动走 MPS。
- **Windows/Linux + NVIDIA 显卡**：装 CUDA 版 torch：
  `pip install torch --index-url https://download.pytorch.org/whl/cu121`
  再 `pip install numpy tokenizers`。
- **Google Colab / Kaggle**：torch/numpy 已预装，只需 `pip install tokenizers`（通常也已预装）。

## 启动训练（从头开始）

```bash
python3 pretrain.py --preset gpt2-demo --amp \
  --batch-size 16 --grad-accum 8 \
  --max-steps 3900 --lr 4e-4 --min-lr 4e-5 --warmup-steps 300 \
  --eval-interval 250 --ckpt-interval 250 --log-interval 10 \
  --out-dir models/gpt2_pretrain
```

NVIDIA GPU（显存 ≥ 12GB）建议改为 `--batch-size 32 --grad-accum 4`（每步 token 数不变，速度更快）。

## 断点续训

checkpoint 每 250 步自动存到 `models/gpt2_pretrain/ckpt.pt`。中断后加 `--resume` 继续：

```bash
python3 pretrain.py --preset gpt2-demo --amp ...（参数同上）... --resume
```

换机器时把 `models/gpt2_pretrain/ckpt.pt` 一起拷过去即可接着训练。

## 训练完成后验证

```bash
python3 generate_gpt2.py --ckpt models/gpt2_pretrain/best.pt \
  --tokenizer-dir models/bpe_tokenizer --prompt "人工智能" --max-new-tokens 100
```

## 预计耗时（3900 步 ≈ 2 个 epoch）

| 硬件 | 速度 | 总耗时 |
|------|------|--------|
| M1 (MPS, fp16) | ~4,000 tok/s | ~18 小时 |
| Colab T4 | ~25,000 tok/s | ~3 小时 |
| RTX 3060 以上 | 30,000+ tok/s | ~2 小时 |

## 训练完成后带回原项目

只需拷回 `models/gpt2_pretrain/` 目录（ckpt.pt / best.pt / train.log）到
`/Volumes/CODE/LLMDemo/models/gpt2_pretrain/`。

语料许可：中文维基百科（CC BY-SA 3.0）。
