#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文对话能力验收 / 回归测试（对应产品验收清单 1~9、14）

覆盖项：
  1. 中文身份问答（含Nova）          -> TestIdentity.test_chinese_identity
  2. 英文身份问答（含Nova）          -> TestIdentity.test_english_identity
  3. 中文知识问答抽查（>=5条训练内）  -> TestKnowledgeQA
  4. 时间技能（中/英）               -> TestBuiltinSkills.test_time_*
  5. 日期技能                        -> TestBuiltinSkills.test_date_chinese
  6. 算术技能（含除零）              -> TestBuiltinSkills.test_arithmetic*
  7. 训练外问题兜底                  -> TestRobustness.test_out_of_training_fallback
  8. 输出无特殊标记                  -> TestRobustness.test_no_special_tokens_leak
  9. 中文标点规范                    -> TestChinesePunctuation
 14. 训练数据编码长度<=64            -> TestDataSpec.test_pair_length_within_limit

运行方式（模型只加载一次，全部用例复用）：
    python test_chinese_conversation.py -v
    或 python -m pytest test_chinese_conversation.py -v
"""

import contextlib
import io
import os
import re
import sys
import unittest
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from inference import LLMInference
from train_with_conversation import load_conversation_pairs

MODEL_PATH = 'models/conversation_llm_model.pth'
VOCAB_PATH = 'models/conversation_vocab.pkl'
DATA_PATH = 'data/conversation_data.txt'

SPECIAL_TOKENS = ('<BOS>', '<EOS>', '<SEP>', '<PAD>', '<UNK>')

# 模块级单例：模型只加载一次，所有测试类复用
_inference = None
# 回复缓存：同一提问只生成一次，加快整体测试速度
_reply_cache = {}


def get_inference():
    """加载（或复用）对话模型"""
    global _inference
    if _inference is None:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _inference = LLMInference(
                model_path=MODEL_PATH,
                vocab_path=VOCAB_PATH,
                preserve_case=True
            )
    return _inference


def generate(prompt):
    """向Nova提问并返回回复（不走缓存；generate_text的调试输出被静默）"""
    inference = get_inference()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        results = inference.generate_text(
            prompt,
            max_length=40,
            temperature=0.3,
            top_k=1,
            is_conversation=True
        )
    return results[0] if results else ''


def ask(prompt):
    """向Nova提问并返回回复（同一提问只生成一次）"""
    if prompt not in _reply_cache:
        _reply_cache[prompt] = generate(prompt)
    return _reply_cache[prompt]


def has_chinese(text):
    return bool(re.search(r'[一-鿿]', text))


def squeeze(text):
    """去掉所有空白，用于与训练数据做内容比对
    （分词还原时ASCII词与汉字之间会留空格，属显示格式差异，不影响内容）"""
    return re.sub(r'\s+', '', text)


class TestIdentity(unittest.TestCase):
    """验收项1、2：中英文身份问答"""

    def test_chinese_identity(self):
        """1. "你叫什么名字" 返回含Nova的中文回答"""
        reply = ask('你叫什么名字')
        self.assertIn('Nova', reply, f'回复未包含Nova: {reply!r}')
        self.assertTrue(has_chinese(reply), f'回复不是中文: {reply!r}')

    def test_english_identity(self):
        """2. "What's your name?" 返回含Nova的英文回答"""
        reply = ask("What's your name?")
        self.assertIn('Nova', reply, f'回复未包含Nova: {reply!r}')
        self.assertFalse(has_chinese(reply), f'英文提问不应返回中文: {reply!r}')


class TestKnowledgeQA(unittest.TestCase):
    """验收项3：中文知识问答抽查（均为data/conversation_data.txt训练内问题）"""

    # (问题, 期望完整回答[与训练数据一致], 关键内容)
    CASES = [
        ('中国的首都是哪里', '中国的首都是北京。', '北京'),
        ('什么是人工智能', '人工智能是让机器模拟人类智能的技术。', '人工智能'),
        ('什么是机器学习', '机器学习是让计算机从数据中学习规律的方法。', '机器学习'),
        ('什么是深度学习', '深度学习是用多层神经网络学习数据特征的方法。', '深度学习'),
        ('天空为什么是蓝色的', '因为大气散射阳光，蓝光散射最强，所以天空是蓝色的。', '蓝'),
        ('太阳从哪边升起', '太阳从东边升起，从西边落下。', '东'),
        ('一周有几天', '一周有7天。', '7'),
    ]

    def test_knowledge_answers_match_training_data(self):
        """3. 至少5条训练内问题，回答与训练数据一致（忽略空白差异）"""
        for question, expected, keyword in self.CASES:
            with self.subTest(question=question):
                reply = ask(question)
                self.assertIn(keyword, reply,
                              f'{question!r} 回复缺少关键内容 {keyword!r}: {reply!r}')
                self.assertEqual(
                    squeeze(expected), squeeze(reply),
                    f'{question!r} 回复与训练数据不一致:\n'
                    f'  期望: {expected!r}\n  实际: {reply!r}')


class TestBuiltinSkills(unittest.TestCase):
    """验收项4、5、6：时间/日期/算术内置技能"""

    def _assert_time_close(self, reply, before, after):
        """回复中的HH:MM与系统时间误差<=1分钟"""
        match = re.search(r'(\d{1,2}):(\d{2})', reply)
        self.assertIsNotNone(match, f'回复未包含HH:MM时间: {reply!r}')
        replied = before.replace(hour=int(match.group(1)),
                                 minute=int(match.group(2)),
                                 second=0, microsecond=0)
        # 跨午夜的极端情况下允许日期差一天
        diffs = [abs((replied + timedelta(days=d) - t).total_seconds())
                 for t in (before, after) for d in (-1, 0, 1)]
        self.assertLessEqual(min(diffs), 60,
                             f'回复时间与系统时间误差超过1分钟: {reply!r}')

    def test_time_chinese(self):
        """4a. "现在几点" 返回当前系统时间"""
        before = datetime.now()
        reply = generate('现在几点')  # 时间类不走缓存，保证与当前时间可比
        after = datetime.now()
        self.assertTrue(has_chinese(reply), f'中文提问应返回中文: {reply!r}')
        self._assert_time_close(reply, before, after)

    def test_time_english(self):
        """4b. "What time is it?" 同样返回当前系统时间"""
        before = datetime.now()
        reply = generate('What time is it?')
        after = datetime.now()
        self.assertFalse(has_chinese(reply), f'英文提问应返回英文: {reply!r}')
        self._assert_time_close(reply, before, after)

    def test_date_chinese(self):
        """5. "今天几号" 返回的年/月/日/星期与系统日期一致"""
        now = datetime.now()
        reply = generate('今天几号')  # 日期类不走缓存
        expected_date = f'{now.year}年{now.month}月{now.day}日'
        expected_weekday = '星期' + '一二三四五六日'[now.weekday()]
        self.assertIn(expected_date, reply,
                      f'回复日期与系统不符（期望{expected_date}）: {reply!r}')
        self.assertIn(expected_weekday, reply,
                      f'回复星期与系统不符（期望{expected_weekday}）: {reply!r}')

    def test_arithmetic(self):
        """6a. 加/除/乘算术技能"""
        cases = [
            ('1加1等于几', '2'),
            ('12除以4', '3'),
            ('3乘5', '15'),
        ]
        for question, expected in cases:
            with self.subTest(question=question):
                reply = ask(question)
                match = re.search(r'=\s*(-?\d+(?:\.\d+)?)', reply)
                self.assertIsNotNone(match, f'{question!r} 回复无计算结果: {reply!r}')
                self.assertEqual(expected, match.group(1),
                                 f'{question!r} 计算结果错误: {reply!r}')

    def test_divide_by_zero(self):
        """6b. "5除以0" 给出除零提示且不崩溃"""
        reply = ask('5除以0')
        self.assertTrue(reply, '除零提问返回了空回复')
        self.assertIn('除数', reply, f'除零回复未包含除零提示: {reply!r}')


class TestRobustness(unittest.TestCase):
    """验收项7、8：训练外问题兜底、输出干净"""

    OUT_OF_TRAINING = [
        '量子纠缠是什么',
        '相对论讲的是什么',
        '如何做一道红烧肉',
    ]

    def test_out_of_training_fallback(self):
        """7. 训练外问题返回非空、不崩溃"""
        for question in self.OUT_OF_TRAINING:
            with self.subTest(question=question):
                try:
                    reply = ask(question)
                except Exception as e:  # noqa: BLE001 - 验收要求"不崩溃"
                    self.fail(f'{question!r} 触发异常: {e!r}')
                self.assertTrue(reply and reply.strip(),
                                f'{question!r} 返回了空回复')

    def test_no_special_tokens_leak(self):
        """8. 任意回复中不出现 <BOS> <EOS> <SEP> <PAD> <UNK>"""
        prompts = (
            ['你叫什么名字', "What's your name?", '你好', 'Hello',
             '现在几点', '今天几号', '1加1等于几', '5除以0']
            + [q for q, _, _ in TestKnowledgeQA.CASES]
            + self.OUT_OF_TRAINING
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                reply = ask(prompt)
                for token in SPECIAL_TOKENS:
                    self.assertNotIn(token, reply,
                                     f'{prompt!r} 回复泄漏特殊标记{token}: {reply!r}')


class TestChinesePunctuation(unittest.TestCase):
    """验收项9：中文回复使用中文标点且以句末标点结尾"""

    CHINESE_PROMPTS = [
        '你叫什么名字', '你好', '你是谁', '中国的首都是哪里',
        '什么是人工智能', '天空为什么是蓝色的', '今天几号', '现在几点',
    ]

    def test_chinese_replies_use_chinese_punctuation(self):
        for prompt in self.CHINESE_PROMPTS:
            with self.subTest(prompt=prompt):
                reply = ask(prompt)
                self.assertTrue(has_chinese(reply),
                                f'{prompt!r} 未返回中文回复: {reply!r}')
                # 不应残留英文标点（撇号除外，可能出现在英文人名/缩写中）
                leaked = re.findall(r'[,.!?;]', reply)
                self.assertFalse(leaked,
                                 f'{prompt!r} 回复残留英文标点{leaked}: {reply!r}')
                self.assertTrue(reply.endswith(('。', '！', '？')),
                                f'{prompt!r} 回复未以句末标点结尾: {reply!r}')


class TestDataSpec(unittest.TestCase):
    """验收项14：训练数据每条问答对编码后有效长度<=TRAIN_MAX_SEQ_LEN(64)"""

    def test_pair_length_within_limit(self):
        pairs = load_conversation_pairs(DATA_PATH)
        self.assertGreater(len(pairs), 0, '未解析到任何问答对')

        preprocessor = get_inference().data_preprocessor
        pad_id = preprocessor.special_tokens['<PAD>']
        limit = Config.TRAIN_MAX_SEQ_LEN

        oversized = []
        for question, answer in pairs:
            # 用远大于限制的填充长度编码，统计<PAD>之前的有效token数
            seq = preprocessor.pair_to_ids(question, answer,
                                           max_length=Config.MAX_SEQ_LEN)
            effective = seq.index(pad_id) if pad_id in seq else len(seq)
            if effective > limit:
                oversized.append((question, effective))

        self.assertFalse(
            oversized,
            f'以下问答对编码后超过{limit}个token（训练时会被截断）: {oversized}')


if __name__ == '__main__':
    unittest.main(verbosity=2)
