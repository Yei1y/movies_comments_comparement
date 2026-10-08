# -*- coding: utf-8 -*-
"""豆瓣影评文本预处理：文本清洗 -> jieba 分词 -> 停用词过滤。

词云/LDA、SnowNLP 情感分析与各分类模型共用本模块，以保证三个任务看到的
是完全一致的预处理文本（避免"同一批评论、不同预处理"导致结果不可比）。

约定：所有路径均以项目根目录为基准解析，因此脚本可在任意工作目录下运行。
"""
from pathlib import Path
import re

import jieba
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'

# 自定义词典未收录的影视专有名词（jieba 会把它们切碎，故作补充）
CUSTOM_WORDS = [
    '钢铁侠', '超级英雄', '超英电影', '寡姐', '美队', '小蜘蛛', '治愈人',
    '雷霆特工队', '模仿大师', '叶莲娜', '瓦伦蒂娜', '美国队长', '美国密探', '反英雄',
]


def load_comments(filename):
    """从 data/ 读取评论数据，返回 DataFrame(Comment, Type)。"""
    return pd.read_csv(DATA_DIR / filename)


class DoubanTextCleaner:
    """豆瓣短评预处理器。"""

    def __init__(self, user_dict=None, stopwords_file=None):
        user_dict = Path(user_dict) if user_dict else DATA_DIR / 'user.dict.utf8'
        stopwords_file = Path(stopwords_file) if stopwords_file else DATA_DIR / 'stopwords_hit.txt'
        self._setup_jieba(user_dict, stopwords_file)

    def _setup_jieba(self, user_dict, stopwords_file):
        """加载自定义词典、补充影视专有名词并读取停用词表。"""
        if user_dict.exists():
            jieba.load_userdict(str(user_dict))
        for word in CUSTOM_WORDS:
            jieba.add_word(word)

        if stopwords_file.exists():
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                self.stopwords = {line.strip() for line in f if line.strip()}
        else:
            self.stopwords = set()

    def clean_text(self, text):
        """去除标点、数字、英文、表情符号与零宽字符，保留纯中文。"""
        if not isinstance(text, str):
            return ''

        patterns = [
            r'[^\w\s]',                    # 标点符号（Python 中 \w 可匹配汉字，故只清标点）
            r'[\U0001F600-\U0001F64F]',    # 表情符号
            r'[\U0001F300-\U0001F5FF]',    # 其他符号
            r'[\U0001F680-\U0001F6FF]',    # 交通和地图符号
            r'[\U0001F1E0-\U0001F1FF]',    # 国旗
            r'[\U00002695-\U0000269F]',    # 杂项符号
            r'[\U00002600-\U00002B55]',    # 更多符号
            r'[\U0000231A-\U0000231B]',    # 时钟
            r'[\U000025FB-\U000025FE]',    # 几何图形
            r'[\d]',                       # 数字
            r'[a-zA-Z]',                   # 英文字母
            r'[\u200b\u200c\u200d]',       # 零宽字符
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text)

        # 压缩连续空白
        return re.sub(r'\s+', ' ', text).strip()

    def segment_text(self, text):
        """清洗 -> 分词 -> 去停用词与单字，返回词语列表。"""
        words = jieba.lcut(self.clean_text(text))
        return [w for w in words if w not in self.stopwords and len(w) > 1]

    def preprocess_for_wordcloud(self, texts, type_filter=None):
        """词云/LDA 用预处理：按评论类型筛选、去重去空后分词。

        :param texts: 评论记录列表（含 Comment / Type 字段）或 DataFrame
        :param type_filter: 可选的评论类型过滤，如 '好评'
        :return: 词语列表（含重复，供词频统计使用）
        """
        df = texts if isinstance(texts, pd.DataFrame) else pd.DataFrame(texts)
        if type_filter is not None and 'Type' in df.columns:
            df = df[df['Type'] == type_filter]
        df = df.dropna(subset=['Comment']).drop_duplicates(subset=['Comment'])

        all_words = []
        for comment in df['Comment']:
            all_words.extend(self.segment_text(str(comment)))
        return all_words

    def preprocess_for_analysis(self, text):
        """情感分析/分类模型用预处理：返回空格分隔的词序列。"""
        return ' '.join(self.segment_text(text))


if __name__ == '__main__':
    cleaner = DoubanTextCleaner()

    test_cases = [
        "钢铁侠3️⃣是2020年最棒👍的超级英雄电影！IMDb评分8.5⭐",
        {"Comment": "寡姐🙄的表现惊艳！剧情⭐⭐⭐", "Type": "好评"},
        12345,
        "这部电影很一般......",
    ]

    print("=== 清洗测试 ===")
    for case in test_cases:
        content = case['Comment'] if isinstance(case, dict) else str(case)
        print(f"原始: {content}\n清洁后: {cleaner.clean_text(content)}\n")

    print("=== 分词测试 ===")
    sample = "美国队长和钢铁侠在《复仇者联盟》中的对决令人难忘！"
    print(f"原始: {sample}")
    print(f"分词: {cleaner.segment_text(sample)}")
    print(f"分类模型输入: {cleaner.preprocess_for_analysis(sample)}")