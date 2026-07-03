# LLMDemo 项目指南

教学演示项目：从零构建中文对话 LLM。自研模型统称 **Nova**（GPT-2 风格结构的教学复现，
**不叫 GPT-2**——只是"向 GPT-2 方向迈进"；备注/文案一律用"Nova"或"GPT-2 风格结构"）。
机器人对外名字是 `Config.BOT_NAME`（"Nova"）。

## 目录结构

`nova/`（新一代模型核心）、`legacy/`（旧记忆式模型，v0.2 回退用）、`scripts/`
（训练与数据流水线，从项目根目录运行）、`tests/`、`docs/`（产品计划、迁移指南）；
根目录只保留入口 `web_server.py` 和全局 `config.py`。

## 架构总览

两代模型并存，Web 端自动选择：

- **新一代（v0.3+，主力）**：Nova 基座 = GPT-2 风格 Transformer（`nova/gpt2_model.py`，
  预设 gpt2-demo：8层/8头/512维/33.87M 参数）+ Byte-Level BPE 16384 词表
  （`bpe_tokenizer.py`）。中文维基预训练（127M tokens，val loss 3.985）→
  对话微调（问答格式 + 损失掩码 + 语料回放）。
- **旧一代（v0.2，回退用）**：词表级记忆式小模型（`legacy/`（model/trainer/inference），只能复现训练过的 ~100 组问答。

**对话回复流水线**（`nova/chat_inference.py` 的 ChatBot.reply）：
1. 内置技能（`legacy/inference.py` 的 `_builtin_response`：时间/日期/天气/算术，规则实现，新旧模型共用）
2. 知识库精确检索（`data/knowledge_qa.tsv`，3008 条"实体→维基定义句"，零幻觉）
3. Nova 模型生成（重复惩罚采样 rp=1.2/temp=0.3/top_k=20，防"复读机"退化）
4. 后处理（截断自问自答、折叠循环重复、空回复兜底）

## 关键文件

| 文件 | 作用 |
|------|------|
| `nova/gpt2_model.py` | Nova 模型结构（GPTConfig/GPT/PRESETS），文件名保留勿改 |
| `scripts/pretrain.py` | 预训练（nanoGPT 式：memmap + AdamW + warmup/cosine + AMP，支持 --resume） |
| `scripts/finetune_chat.py` | 对话微调（对话+知识+回放三源混合，损失掩码只算回答部分） |
| `nova/chat_inference.py` | ChatBot 推理封装（上面的四层流水线） |
| `scripts/mine_knowledge_qa.py` | 从维基语料挖"实体→定义句"知识对 |
| `scripts/generate_gpt2.py` | 基座模型续写测试（交互/单次） |
| `web_server.py` | Flask 服务（127.0.0.1:8000），优先 Nova、回退旧模型 |
| `nova/bpe_tokenizer.py` / `scripts/encode_corpus.py` / `scripts/prepare_pretrain_data.py` | BPE 训练 / 语料编码 / 语料下载清洗 |
| `docs/PRODUCT_PLAN.md` | 产品定位、验收清单、路线图、各版本里程碑（改动后同步更新） |
| `tests/test_chinese_conversation.py` | 回归测试（12 用例，覆盖内置技能与旧模型链路） |

模型产物：`models/gpt2_pretrain/best.pt`（基座）、`models/gpt2_chat/chat.pt`（对话）、
`models/bpe_tokenizer/tokenizer.json`。数据：`data/conversation_data.txt`（问答对，
空行分块，首行问题）、`data/knowledge_qa.tsv`、`data/pretrain/`（gitignored，
504MB 语料 + train.bin/val.bin）。

## 常用命令

```bash
python3 web_server.py                 # 启动 Web UI（端口 8000）
python3 nova/chat_inference.py        # 命令行对话测试
python3 scripts/finetune_chat.py      # 重新微调对话模型（~7 分钟，改对话数据后必跑）
python3 scripts/mine_knowledge_qa.py  # 重挖知识库（--max-pairs 调覆盖面，候选 7.5 万）
python3 -m unittest tests.test_chinese_conversation  # 回归测试（改内置技能后必跑）
python3 scripts/generate_gpt2.py --prompt "人工智能"  # 基座续写测试
```

预训练完整流程（一般不需重跑）：`prepare_pretrain_data.py --all` →
`bpe_tokenizer.py train` → `encode_corpus.py` → `scripts/pretrain.py`（M1 约 18h，
建议用迁移包上 Colab T4 约 3h，见 `docs/README_MIGRATE.md`）。

## 约定与坑

- **命名**：对外/注释不称 GPT-2，用 Nova；结构描述可写"GPT-2 风格"。
- **注释风格**：全项目中文注释；注释解释"为什么"而非"是什么"。
- **不要主动 git commit**——用户明确要求先问。`.gitignore` 已排除 data/pretrain/。
- **用户常自跑 `python web_server.py`**：改代码前先 `pgrep -f web_server` 确认，
  不要 kill 用户的进程；改完提醒用户重启生效。
- **本机环境**：Apple M1 16GB，torch 2.9.0，设备自动选择 mps>cuda>cpu；
  MPS 上 AMP 用 fp16 autocast（无 GradScaler）。
- **代理坑**：终端有 `ALL_PROXY=socks5://127.0.0.1:7890`，curl 本机服务必须加
  `--noproxy '*'`；HuggingFace Xet 下载会挂死，用分段 curl + HTTP/1.1（见
  scripts/prepare_pretrain_data.py）。
- **训练数据格式**：conversation_data.txt 改动后跑 scripts/finetune_chat.py 重训；
  知识事实一律走 knowledge_qa.tsv 检索，**不要试图让 34M 模型背事实**
  （几百步微调每条样本只被看到 <1 次，只会学出流利的胡编）。
- **微调超参已调优**：默认值即可（1e-4 cosine、回放 25%、知识 200 实体只教格式）；
  推理采样参数 rp=1.2/temp=0.3/top_k=20 是对比实验的结果，改前先跑对比。
- **语料许可**：中文维基（pleisto/wikipedia-cn-20230720-filtered），CC BY-SA 3.0。
