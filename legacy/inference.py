"""
模型推理模块
负责加载训练好的模型并进行文本生成
"""

import re
import torch
import numpy as np
from datetime import datetime
from legacy.model import SimpleLLM
from legacy.data_preprocessor import DataPreprocessor
from config import Config


class LLMInference:
    """LLM推理器"""
    
    def __init__(self, model_path=None, vocab_path=None, preserve_case=False):
        self.model = SimpleLLM()
        self.preserve_case = preserve_case  # 对话模型需要设为True
        self.data_preprocessor = DataPreprocessor(preserve_case=preserve_case)
        
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

        # 对话模式下先检查内置技能（如实时时间），命中则无需模型生成
        if is_conversation:
            builtin = self._builtin_response(prompt)
            if builtin:
                print(f"命中内置技能: {builtin}")
                return [builtin] * num_return_sequences

        results = []

        for i in range(num_return_sequences):
            # generate返回生成的回复部分（不含prompt）
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
    
    def _builtin_response(self, prompt):
        """内置规则技能：语言模型无法感知实时信息，时间/日期类问题直接读系统时间

        命中返回回答字符串，未命中返回None（交给模型生成）
        """
        p = prompt.lower()
        now = datetime.now()
        # 提问包含汉字时用中文回答
        is_chinese = bool(re.search(r'[一-鿿]', prompt))

        # 时间类：What time is it? / 现在几点 / 现在的时间是多少
        # "现在.{0,4}时间"容忍"现在的时间""现在北京时间"等插入语；
        # 不用裸"时间"关键词，避免"训练要多少时间"这类时长问题误命中
        if re.search(r'几点|现在.{0,4}时间|当前时间|时间是多少|什么时间', prompt) or (
                re.search(r'\btime\b', p) and re.search(r'what|now|current|tell|know', p)):
            if is_chinese:
                return now.strftime("现在是 %H:%M。")
            return now.strftime("It's %H:%M right now.")

        # 日期类：What's the date? / What day is it? / 今天几号
        if re.search(r'几号|日期|星期几', prompt) or (
                re.search(r'\b(date|day)\b', p) and re.search(r'what|now|current|today|tell', p)):
            if is_chinese:
                weekday = '一二三四五六日'[now.weekday()]
                return f"今天是{now.year}年{now.month}月{now.day}日，星期{weekday}。"
            return now.strftime("Today is %A, %B %d, %Y.")

        # 天气类：北京天气怎么样 / 上海的天气（实时数据，查Open-Meteo免费API）
        weather = self._weather_response(prompt, is_chinese)
        if weather:
            return weather

        # 算术类：1+1等于几 / What is 3*4 / 12除以4
        arithmetic = self._arithmetic_response(prompt, is_chinese)
        if arithmetic:
            return arithmetic

        return None

    # WMO天气代码 -> 中文描述（Open-Meteo使用的标准代码）
    _WEATHER_CODES = {
        0: "晴", 1: "基本晴朗", 2: "少云", 3: "阴",
        45: "有雾", 48: "冻雾",
        51: "毛毛雨", 53: "毛毛雨", 55: "毛毛雨",
        56: "冻毛毛雨", 57: "冻毛毛雨",
        61: "小雨", 63: "中雨", 65: "大雨", 66: "冻雨", 67: "冻雨",
        71: "小雪", 73: "中雪", 75: "大雪", 77: "米雪",
        80: "阵雨", 81: "阵雨", 82: "强阵雨",
        85: "阵雪", 86: "强阵雪",
        95: "雷阵雨", 96: "雷阵雨伴冰雹", 99: "强雷阵雨伴冰雹",
    }

    # 常见外国城市中文名 -> 英文名（地理编码API的中文索引不含外国城市译名）
    _CITY_ALIASES = {
        '纽约': 'New York', '伦敦': 'London', '东京': 'Tokyo', '巴黎': 'Paris',
        '首尔': 'Seoul', '洛杉矶': 'Los Angeles', '旧金山': 'San Francisco',
        '悉尼': 'Sydney', '莫斯科': 'Moscow', '新加坡': 'Singapore',
        '曼谷': 'Bangkok', '迪拜': 'Dubai', '柏林': 'Berlin', '罗马': 'Rome',
    }

    def _weather_response(self, prompt, is_chinese):
        """实时天气技能：提取城市名，调用Open-Meteo（免费、无需密钥）

        命中"天气/weather"关键词才处理；查询失败时给出友好提示而不是报错
        """
        if not re.search(r'天气|weather', prompt, re.IGNORECASE):
            return None

        # 提取城市：取"天气"前面的连续汉字/字母，剥掉常见修饰词
        m = re.search(r'([一-鿿A-Za-z]+?)(?:市|省|县|区)?的?(?:今天|明天|现在|当前)?的?天气', prompt)
        city = m.group(1) if m else ''
        for word in ('请问', '查询', '查一下', '查', '今天', '明天', '现在',
                     '当前', '想知道', '告诉我', '一下', '的'):
            city = city.replace(word, '')

        if not city:
            if is_chinese:
                return "你想查询哪个城市的天气呢？比如：北京天气怎么样"
            return "Which city's weather would you like to know? e.g. 'weather in Beijing'"

        try:
            import json as _json
            from urllib.request import urlopen
            from urllib.parse import quote

            # 同名地点很多（如"佛山"首个结果是云南一个海拔4800米的村镇），
            # 同时搜"X"和"X市"，取人口最多的结果——用户问天气时几乎总是指大城市
            candidates = []
            alias = self._CITY_ALIASES.get(city)
            if alias:
                names = [alias]
            elif city.endswith(('市', '省', '县', '区')):
                names = [city]
            else:
                names = [city, city + '市']
            for name in names:
                # 英文名必须用language=en查（zh语言下英文名匹配不准）
                lang = 'en' if name.isascii() else 'zh'
                geo_url = ("https://geocoding-api.open-meteo.com/v1/search"
                           f"?name={quote(name)}&count=10&language={lang}")
                with urlopen(geo_url, timeout=6) as resp:
                    geo = _json.loads(resp.read().decode('utf-8'))
                candidates.extend(geo.get('results') or [])
            if not candidates:
                if is_chinese:
                    return f"抱歉，我没找到“{city}”这个城市，换个名字试试？"
                return f"Sorry, I couldn't find the city '{city}'."

            loc = max(candidates, key=lambda r: r.get('population') or 0)
            wx_url = ("https://api.open-meteo.com/v1/forecast"
                      f"?latitude={loc['latitude']}&longitude={loc['longitude']}"
                      "&current=temperature_2m,weather_code,wind_speed_10m"
                      "&timezone=auto")
            with urlopen(wx_url, timeout=6) as resp:
                wx = _json.loads(resp.read().decode('utf-8'))
            cur = wx['current']
            desc = self._WEATHER_CODES.get(cur['weather_code'], '未知天气')
            name = loc.get('name', city)
            # 带上省份，便于用户核对是不是自己想问的那个同名城市
            admin = loc.get('admin1', '')
            if admin and not name.isascii() and not name.startswith(admin[:2]):
                name = f"{admin}{name}"
            if is_chinese:
                return (f"{name}当前天气：{desc}，气温{cur['temperature_2m']}°C，"
                        f"风速{cur['wind_speed_10m']}km/h。")
            return (f"Current weather in {name}: {desc}, "
                    f"{cur['temperature_2m']}°C, wind {cur['wind_speed_10m']} km/h.")
        except Exception as e:
            print(f"天气查询失败: {e}")
            if is_chinese:
                return "天气服务暂时连不上，请稍后再试~"
            return "The weather service is unavailable right now, please try again later."

    def _arithmetic_response(self, prompt, is_chinese):
        """简单算术技能：提取"数字 运算符 数字"并计算，支持中英文表达

        命中返回回答字符串，未命中返回None
        """
        match = re.search(
            r'(\d+(?:\.\d+)?)\s*(加|减|乘以|乘|除以|除|[+\-*xX×/÷])\s*(\d+(?:\.\d+)?)',
            prompt)
        if not match:
            return None

        a, op, b = float(match.group(1)), match.group(2), float(match.group(3))

        # 减号易与连字符/日期（如2026-07-02）混淆，仅在有明确算术意图时才计算
        if op == '-' and not re.search(
                r"等于|是多少|算|计算|减|what is|what's|calculate|how much|=", prompt.lower()):
            return None

        if op in ('加', '+'):
            result = a + b
        elif op in ('减', '-'):
            result = a - b
        elif op in ('乘', '乘以', '*', 'x', 'X', '×'):
            result = a * b
        else:  # 除 / 除以 / '/' / '÷'
            if b == 0:
                return "0不能作为除数哦。" if is_chinese else "Sorry, I can't divide by zero."
            result = a / b

        # 整数结果不显示小数点，小数结果保留4位
        result_str = str(int(result)) if result == int(result) else str(round(result, 4))
        expr = f"{match.group(1)} {op} {match.group(3)}"
        return f"{expr} = {result_str}。" if is_chinese else f"{expr} = {result_str}."

    def _clean_generated_text(self, text):
        """清理生成的文本，移除特殊标记"""
        # 移除特殊标记
        for token in ['<BOS>', '<EOS>', '<PAD>', '<UNK>', '<SEP>']:
            text = text.replace(token, '')

        # 规范化空格
        text = ' '.join(text.split())

        return text.strip()

    def _conversation_postprocess(self, generated_text, prompt=''):
        """对话模式下的后处理，按提问/回复语言分别处理中英文"""
        is_chinese_prompt = bool(re.search(r'[一-鿿]', prompt))

        # 如果回复为空，使用兜底回复（中文提问用中文兜底）
        if len(generated_text) < 2:
            if is_chinese_prompt:
                return "我在呢！有什么可以帮你的吗？"
            return "I'm here to help! What can I assist you with today?"

        if re.search(r'[一-鿿]', generated_text):
            # 中文回复：把训练时归一化的英文标点还原为中文标点，不做首字母大写
            for en_punct, zh_punct in ((',', '，'), ('!', '！'), ('?', '？'),
                                       (';', '；'), ('.', '。')):
                generated_text = generated_text.replace(en_punct, zh_punct)
            if not generated_text.endswith(('。', '！', '？')):
                generated_text = generated_text + '。'
            return generated_text

        # 英文回复：确保以句号结尾且首字母大写
        if not generated_text.endswith(('.', '!', '?')):
            generated_text = generated_text + '.'
        if generated_text[0].islower():
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
        print(f"\n=== {Config.BOT_NAME} 对话助手 ===")
        print(f"现在您可以与{Config.BOT_NAME}进行自然对话了！")
        print("输入 'quit' 退出对话")
        print("输入 'settings' 调整生成参数")
        print("输入 'clear' 清空对话历史")
        print("-" * 50)
        
        # 默认参数（优化对话）
        max_length = 40   # 对话回复更短
        temperature = 0.3  # 小模型是记忆式回答，低温回答最稳定
        top_k = 1          # 贪心解码：全部训练问答可精确复现，采样反而引入错误

        # 对话历史（仅用于展示，模型按单轮问答训练，输入只用当前问题）
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
                
                # 生成回复（模型按单轮问答训练，直接使用当前输入作为问题）
                print(f"🤖 {Config.BOT_NAME}思考中...")
                results = self.generate_text(
                    prompt=user_input,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    is_conversation=True
                )
                
                ai_response = results[0] if results else "I'm here to help!"

                # 显示AI回复
                print(f"🤖 {Config.BOT_NAME}: {ai_response}")

                # 保存到对话历史
                conversation_history.append(f"用户: {user_input}")
                conversation_history.append(f"{Config.BOT_NAME}: {ai_response}")
                
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
        print("推荐对话设置: 最大长度30-50, 温度0.5-0.8")
        
        try:
            max_length = int(input("最大回复长度 (默认40): ") or 40)
            temperature = float(input("创意程度 (默认0.7): ") or 0.7)
            top_k = int(input("多样性控制 (默认20): ") or 20)

            print(f"✅ 参数已更新: 回复长度={max_length}, 创意程度={temperature}, 多样性={top_k}")
            return max_length, temperature, top_k

        except ValueError:
            print("❌ 输入参数无效，使用默认对话设置")
            return 40, 0.7, 20
    
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