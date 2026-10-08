# -*- coding: utf-8 -*-
"""任务一：短评词云、高频词与 LDA 主题建模。

产出（均落在 output/ 下）
- figures/wordcloud_{movie}_{type}.png   三类评论的分词词云
- figures/topwords_{movie}_{type}.png    三类评论的高频词 TOP10 条形图
- figures/lda_topics_{movie}.png         两个主题的高频词条形图
- figures/lda_visualization_{movie}.html pyLDAvis 交互式主题可视化
- tables/topwords_{movie}.csv            三类评论的高频词与词频（便于核验与引用）
"""
from pathlib import Path
from collections import Counter
import os
import webbrowser

import matplotlib
matplotlib.use('Agg')            # 无界面环境也能出图
import matplotlib.pyplot as plt
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models
import seaborn as sns
from gensim import corpora, models
from matplotlib import font_manager
from wordcloud import WordCloud

from text_cleaner import DoubanTextCleaner, load_comments

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / 'output' / 'figures'
TABLE_DIR = ROOT / 'output' / 'tables'

MOVIES = {
    'av': {'title': '复仇者联盟4', 'file': 'Avengers-Endgame_comments.csv'},
    'th': {'title': '雷霆特工队', 'file': 'Thunderbolts_comments.csv'},
}
COMMENT_TYPES = [('好评', 'h'), ('差评', 'l'), ('一般', 'm')]

NUM_TOPICS = 2        # 两部影片的短评都呈现"认可 / 批评"二分结构，故设为 2
TOP_N = 10            # 报告的高频词个数
LDA_PASSES = 15
RANDOM_SEED = 42
DPI = 100             # 与原项目导出尺寸（词云 1000×800、条形图 1200×800）保持一致

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def _chinese_font():
    """定位可用的中文字体文件（词云需要字体文件路径，而非字体族名）。"""
    for name in ('SimHei', 'Microsoft YaHei', 'SimSun'):
        try:
            return font_manager.findfont(name, fallback_to_default=False)
        except Exception:
            continue
    raise RuntimeError("未找到中文字体，请安装 SimHei 或在 _chinese_font() 中指定字体路径")


class EnhancedVisualizer:
    """词云、词频与主题图的绘制工具。"""

    FONT_PATH = None

    @classmethod
    def generate_wordcloud(cls, words, title, output_path, font_path=None):
        """生成中文词云并保存为 PNG，同时返回词频字典。"""
        font_path = font_path or cls.FONT_PATH or _chinese_font()
        word_freq = dict(Counter(words).most_common(200))

        wc = WordCloud(
            font_path=font_path,
            width=800, height=600,
            background_color='white',
            max_words=200, max_font_size=100,
        ).generate_from_frequencies(word_freq)

        plt.figure(figsize=(10, 8))
        plt.imshow(wc, interpolation='bilinear')
        plt.title(f"{title} - 词云图", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=DPI)
        plt.close()
        return word_freq

    @staticmethod
    def plot_word_frequency(word_freq, title, output_path, top_n=TOP_N):
        """绘制高频词条形图并保存。"""
        top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n])

        plt.figure(figsize=(12, 8))
        sns.barplot(x=list(top_words.values()), y=list(top_words.keys()), palette="viridis")
        plt.title(f"{title} - 高频词TOP{top_n}", fontsize=14)
        plt.xlabel("出现频次")
        plt.ylabel("词语")
        plt.tight_layout()
        plt.savefig(output_path, dpi=DPI)
        plt.close()

    @staticmethod
    def visualize_lda(lda_model, corpus, dictionary, html_path, open_browser=False):
        """导出 pyLDAvis 交互式可视化到 HTML。"""
        vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary, sort_topics=False)
        pyLDAvis.save_html(vis, str(html_path))
        if open_browser:
            webbrowser.open(html_path.resolve().as_uri())
        return vis

    @staticmethod
    def plot_topic_words(lda_model, output_path, num_topics=NUM_TOPICS, num_words=TOP_N):
        """绘制每个主题的前 N 个高频词条形图并保存。"""
        topics = lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)

        plt.figure(figsize=(12, 4 * num_topics))
        for topic_id, topic_words in topics:
            words = [word for word, _ in topic_words]
            weights = [weight for _, weight in topic_words]
            plt.subplot(num_topics, 1, topic_id + 1)
            sns.barplot(x=weights, y=words, palette="rocket")
            plt.title(f"主题 #{topic_id + 1} 高频词TOP{num_words}", fontsize=12)
            plt.xlabel("权重")
            plt.ylabel("词语")
        plt.tight_layout()
        plt.savefig(output_path, dpi=DPI)
        plt.close()


class TopicModeler:
    """LDA 主题建模。"""

    @staticmethod
    def lda_model(texts, num_topics=NUM_TOPICS, passes=LDA_PASSES, random_state=RANDOM_SEED):
        """在"空格分隔词序列"构成的语料上训练 LDA。"""
        tokenized = [text.split() for text in texts]
        dictionary = corpora.Dictionary(tokenized)
        corpus = [dictionary.doc2bow(tokens) for tokens in tokenized]

        lda = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=passes,
            random_state=random_state,
        )
        return lda, corpus, dictionary

    @staticmethod
    def show_topics(lda_model, num_words=TOP_N):
        """返回每个主题的关键词分布。"""
        return lda_model.print_topics(num_words=num_words)


def run_movie(key, open_browser=False):
    """对一部影片执行词云 + 高频词 + 主题建模全流程。"""
    movie = MOVIES[key]
    title, filename = movie['title'], movie['file']
    cleaner = DoubanTextCleaner()
    data = load_comments(filename)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 三类评论的词云与高频词
    freq_rows = []
    for comment_type, slug in COMMENT_TYPES:
        print(f"\n正在处理《{title}》- {comment_type}...")
        words = cleaner.preprocess_for_wordcloud(data, type_filter=comment_type)
        word_freq = EnhancedVisualizer.generate_wordcloud(
            words, f"{title} - {comment_type}",
            FIG_DIR / f"wordcloud_{key}_{slug}.png",
        )
        if word_freq:
            EnhancedVisualizer.plot_word_frequency(
                word_freq, f"{title} - {comment_type}",
                FIG_DIR / f"topwords_{key}_{slug}.png",
            )
            for rank, (word, count) in enumerate(
                    sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:50], start=1):
                freq_rows.append({'movie': title, 'type': comment_type,
                                  'rank': rank, 'word': word, 'count': count})

    pd.DataFrame(freq_rows).to_csv(
        TABLE_DIR / f"topwords_{key}.csv", index=False, encoding='utf-8-sig')

    # 2) 主题建模
    print(f"\n正在对《{title}》进行主题建模...")
    segmented = [cleaner.preprocess_for_analysis(str(c)) for c in data['Comment'].fillna('')]
    lda, corpus, dictionary = TopicModeler.lda_model(segmented)

    print(f"\n《{title}》主题关键词分布:")
    for idx, topic in TopicModeler.show_topics(lda):
        print(f"  主题 #{idx + 1}: {topic}")

    EnhancedVisualizer.visualize_lda(
        lda, corpus, dictionary,
        FIG_DIR / f"lda_visualization_{key}.html", open_browser=open_browser)
    EnhancedVisualizer.plot_topic_words(lda, FIG_DIR / f"lda_topics_{key}.png")

    return lda, corpus, dictionary


def task1(open_browser=False):
    """任务一入口：处理两部影片的词云与主题建模。"""
    results = {}
    for key in MOVIES:
        results[key] = run_movie(key, open_browser=open_browser)
    return results


if __name__ == "__main__":
    task1(open_browser=os.environ.get('OPEN_LDA_BROWSER') == '1')
    print("\n词云与主题建模任务已完成。")