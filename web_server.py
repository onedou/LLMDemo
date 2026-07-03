# -*- coding: utf-8 -*-
"""
Web服务器模块
提供基于浏览器的聊天界面，让用户可以通过网页与Nova对话
启动方式: python web_server.py

模型选择：优先加载微调GPT-2对话模型（models/gpt2_chat/chat.pt，v0.4），
文件不存在时回退到旧的记忆式小模型（v0.2）。
"""

import os
import threading

from flask import Flask, jsonify, request

from config import Config

# 服务配置
HOST = '127.0.0.1'
PORT = 8000
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

GPT2_CHAT_CKPT = 'models/gpt2_chat/chat.pt'

app = Flask(__name__, static_folder=STATIC_DIR)
app.json.ensure_ascii = False  # 保证返回的JSON中文不被转义

# 模型单例（启动时加载一次，所有请求共享）
_generate_reply = None  # 统一的回复函数：str -> str
_inference_lock = threading.Lock()  # 推理过程加锁，避免并发请求互相干扰


def load_model():
    """加载对话模型（仅在启动时执行一次）"""
    global _generate_reply
    print("正在加载对话模型，请稍候...")

    if os.path.exists(GPT2_CHAT_CKPT):
        # v0.4：微调Nova模型（GPT-2风格结构、BPE分词，具备基本语言泛化能力）
        from nova.chat_inference import ChatBot
        bot = ChatBot(ckpt_path=GPT2_CHAT_CKPT)
        _generate_reply = bot.reply
    else:
        # v0.2回退：记忆式小模型（仅能复现训练过的问答）
        from legacy.inference import LLMInference
        inference = LLMInference(
            model_path='models/conversation_llm_model.pth',
            vocab_path='models/conversation_vocab.pkl',
            preserve_case=True
        )

        def _legacy_reply(message):
            replies = inference.generate_text(
                message, max_length=40, temperature=0.3,
                top_k=1, is_conversation=True)
            return replies[0].strip() if replies else ''

        _generate_reply = _legacy_reply
        print("提示：未找到Nova对话模型，已回退到旧记忆式模型。"
              "运行 python3 finetune_chat.py 可训练新模型。")

    print("模型加载完成！")


@app.route('/')
def index():
    """返回聊天页面"""
    html_path = os.path.join(STATIC_DIR, 'index.html')
    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()
    # 注入机器人名字，与config.py保持一致
    return html.replace('{{BOT_NAME}}', Config.BOT_NAME)


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    聊天接口
    请求: {"message": "用户输入"}
    响应: {"reply": "Nova的回复"} 或 {"error": "错误信息"}
    """
    data = request.get_json(silent=True) or {}
    message = (data.get('message') or '').strip()

    # 空输入检查
    if not message:
        return jsonify({'error': '消息不能为空，请输入内容后再发送~'}), 400

    try:
        with _inference_lock:
            reply = _generate_reply(message)
        if not reply:
            reply = '抱歉，我暂时不知道怎么回答这个问题~'
        return jsonify({'reply': reply})
    except Exception as e:
        print(f"生成回复时出错: {e}")
        return jsonify({'error': '哎呀，生成回复时出错了，请稍后再试~'}), 500


if __name__ == '__main__':
    load_model()
    print(f"\n{Config.BOT_NAME} Web聊天界面已启动！")
    print(f"请在浏览器中访问: http://{HOST}:{PORT}\n")
    app.run(host=HOST, port=PORT, debug=False)
