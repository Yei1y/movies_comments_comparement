import jieba
import re
from collections import Counter
import pandas as pd

class DoubanTextCleaner:
    def __init__(self):
        # 初始化jieba分词器，加载自定义词典和停用词
        self._setup_jieba()
        
    def _setup_jieba(self):
        """配置jieba分词器，加载自定义词典和停用词"""
        # 加载自定义词典（假设文件路径相同）
        jieba.load_userdict('./user.dict.utf8')
        
        # 添加自定义词语
        custom_words = ["钢铁侠", "超级英雄", "超英电影", "寡姐", "美队",
                       "小蜘蛛", "治愈人", "雷霆特工队", "模仿大师", "叶莲娜",
                       "瓦伦蒂娜", "美国队长", "美国密探", "反英雄"]
        for word in custom_words:
            jieba.add_word(word)
        
        # 加载停用词（从文件读取）
        try:
            with open('stopwords_hit.txt', 'r', encoding='utf-8') as f:
                self.stopwords = set([line.strip() for line in f])
        except FileNotFoundError:
            self.stopwords = set()
    
    def clean_text(self, text):
        """
        文本清洗
        :param text: 输入文本
        :return: 清洗后的文本
        """
        if not isinstance(text, str):
            return ''
            
        # 定义所有需要清除的模式
        patterns = [
            r'[[:punct:]]|[\n]',           # 标点符号和换行符
            r'[\U0001F600-\U0001F64F]',    # 表情符号
            r'[\U0001F300-\U0001F5FF]',    # 其他符号
            r'[\U0001F680-\U0001F6FF]',    # 交通和地图符号
            r'[\U0001F1E0-\U0001F1FF]',    # 国旗
            r'[\U00002695-\U0000269F]',    # 杂项符号
            r'[\U00002600-\U00002B55]',    # 更多符号
            r'[\U0000231A-\U0000231B]',    # 时钟
            r'[\U000025FB-\U000025FE]',    # 几何图形
            r'[\U00002000-\U0000206F]',    # 通用标点
            r'[\d]',                       # 数字
            r'[a-zA-Z]',                   # 英文字母
            r'\s{2,}',                     # 多个空白字符
            r'[\u200b\u200c\u200d]'        # 零宽字符
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text)
            
        # 去除所有多余空格（包括连续空格）
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def segment_text(self, text):
        """
        完整的分词处理流程:
        1. 文本清洗
        2. 中文分词
        3. 去除停用词
        """
        cleaned_text = self.clean_text(text)
        words = jieba.lcut(cleaned_text)
        # 去除停用词和单字
        words = [word for word in words if word not in self.stopwords and len(word) > 1]
        return words
    
    def preprocess_for_wordcloud(self, texts, type_filter=None):
        """
        为词云图准备的预处理
        :param texts: 文本列表(可以是字符串列表或字典列表)
        :param type_filter: 可选的评论类型过滤
        :return: 分词后的词语列表
        """
        if type_filter and isinstance(texts[0], dict):
            texts = [t for t in texts if t.get('Type') == type_filter]
        texts = pd.DataFrame(texts).drop_duplicates(subset=['Comment'])
        texts = texts.dropna(subset=['Comment'])
        texts = texts['Comment'].tolist()
        all_words = []
        for text in texts:
            if isinstance(text, dict):
                content = text.get('Comment', '')
            else:
                content = str(text)
            words = self.segment_text(content)
            all_words.extend(words)
        return all_words
    
    def preprocess_for_analysis(self, text):
        """
        为情感分析和LSTM准备的预处理:
        返回空格分隔的清洗后词序列
        """
        words = self.segment_text(text)
        return ' '.join(words)


# 使用示例
if __name__ == "__main__":
    cleaner = DoubanTextCleaner()
    
    # 测试数据
    test_cases = [
        "钢铁侠3️⃣是2020年最棒👍的超级英雄电影！IMDb评分8.5⭐",
        {"Comment": "寡姐🙄的表现惊艳！剧情⭐⭐⭐", "Type": "好评"},
        12345,
        "这部电影很一般......"
    ]
    
    print("=== 清洗测试 ===")
    for case in test_cases:
        if isinstance(case, dict):
            content = case['Comment']
        else:
            content = str(case)
        cleaned = cleaner.clean_text(content)
        print(f"原始: {content}\n清洁后: {cleaned}\n")
    
    print("\n=== 分词测试 ===")
    sample = "美国队长和钢铁侠在《复仇者联盟》中的对决令人难忘！"
    print(f"原始: {sample}")
    print(f"分词: {cleaner.segment_text(sample)}")
    
    print("\n=== 完整预处理测试 ===")
    processed = cleaner.preprocess_for_analysis(sample)
    print(f"LSTM输入: {processed}")
