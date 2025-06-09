from data_handling import DoubanTextCleaner
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from gensim import corpora, models
import pandas as pd
import pyLDAvis.gensim_models
import seaborn as sns
from collections import Counter
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# =========== 可视化模块增强版 ===========
class EnhancedVisualizer:
    @staticmethod
    def generate_wordcloud(words, title, font_path='simhei.ttf'):
        """生成中文词云图"""
        word_freq = dict(Counter(words).most_common(200))
        
        wc = WordCloud(
            font_path=font_path,
            width=800,
            height=600,
            background_color='white',
            max_words=200,
            max_font_size=100
        ).generate_from_frequencies(word_freq)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(wc, interpolation='bilinear')
        plt.title(f"{title} - 词云图", fontsize=16)
        plt.axis('off')
        plt.show()
        return word_freq  # 返回词频用于进一步分析

    @staticmethod
    def plot_word_frequency(word_freq, title, top_n=10):
        """绘制词频柱状图"""
        top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n])
        
        plt.figure(figsize=(12, 8))
        sns.barplot(
            x=list(top_words.values()),
            y=list(top_words.keys()),
            palette="viridis"
        )
        plt.title(f"{title} - 高频词TOP{top_n}", fontsize=14)
        plt.xlabel("出现频次")
        plt.ylabel("词语")
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def visualize_lda(lda_model, corpus, dictionary, html_path="lda_visualization.html"):
        """主题模型可视化(pyLDAvis)"""
        vis = pyLDAvis.gensim_models.prepare(
            lda_model, 
            corpus, 
            dictionary,
            sort_topics=False
        )
        # 保存为HTML文件并自动打开
        pyLDAvis.save_html(vis, html_path)
        os.system(f"start {html_path}")  # Windows系统自动打开
        return vis

    @staticmethod
    def plot_topic_words(lda_model, num_topics=3, num_words=10):
        """绘制每个主题的前N个高频词柱形图"""
        topics = lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)
        
        plt.figure(figsize=(12, 8))
        for topic_id, topic_words in topics:
            # 提取词和权重
            words = [word for word, _ in topic_words]
            weights = [weight for _, weight in topic_words]
            
            # 绘制柱形图
            plt.subplot(num_topics, 1, topic_id+1)
            sns.barplot(x=weights, y=words, palette="rocket")
            plt.title(f"主题 #{topic_id+1} 高频词TOP{num_words}", fontsize=12)
            plt.xlabel("权重")
            plt.ylabel("词语")
        
        plt.tight_layout()
        plt.show()

# =========== 主题建模模块 ===========
class TopicModeler:
    @staticmethod
    def lda_model(texts, num_topics=3, passes=15):
        """LDA主题建模"""
        dictionary = corpora.Dictionary([text.split() for text in texts])
        corpus = [dictionary.doc2bow(text.split()) for text in texts]
        
        lda = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=passes,
            random_state=1
        )
        
        return lda, corpus, dictionary
    
    @staticmethod
    def show_topics(lda_model, num_words=10):
        """显示每个主题的关键词"""
        return lda_model.print_topics(num_words=num_words)

# =========== 主业务流程 ===========
def task1():
    """任务1：生成复仇者联盟4和雷霆特工队的评论词云和主题建模
    读取两部电影的评论数据，生成词云、词频分析，并进行主题建模。
    """
    # 初始化文本处理器
    cleaner = DoubanTextCleaner()
    
    # 读取数据
    data1 = pd.read_csv('Avengers-Endgame_comments.csv').to_dict('records')
    data2 = pd.read_csv('Thunderbolts_comments.csv').to_dict('records')
    
    # 生成词云和词频分析
    def process_comments(data, title):
        # 词云和词频统计
        for comment_type in ['好评', '差评', '一般']:
            print(f"\n正在处理《{title}》- {comment_type}...")
            words = cleaner.preprocess_for_wordcloud(
                texts=data,
                type_filter=comment_type
            )
            
            # 词云可视化
            word_freq = EnhancedVisualizer.generate_wordcloud(
                words, 
                f"{title} - {comment_type}"
            )
            
            # 词频柱状图
            if word_freq:  # 确保有数据才绘制
                EnhancedVisualizer.plot_word_frequency(
                    word_freq,
                    f"{title} - {comment_type}"
                )
    
    # 处理两部电影数据
    process_comments(data1, "复仇者联盟4")
    process_comments(data2, "雷霆特工队")
    
    # 主题建模
    print("\n正在进行主题建模...")
    segmented_texts1 = [
        cleaner.preprocess_for_analysis(str(item['Comment']))
        for item in data1
    ]
    segmented_texts2 = [
        cleaner.preprocess_for_analysis(str(item['Comment']))
        for item in data2
    ]
    
    # 运行LDA模型
    lda, corpus, dictionary = TopicModeler.lda_model(segmented_texts1, num_topics=2)
    lda2, corpus2, dictionary2 = TopicModeler.lda_model(segmented_texts2, num_topics=2)

    # 显示主题关键词
    topics = TopicModeler.show_topics(lda)
    topics2 = TopicModeler.show_topics(lda2)
    print("\n主题关键词分布:")
    for idx, topic in topics:
        print(f"\n主题 #{idx+1}:")
        print(topic)

    for idx, topic in topics2:
        print(f"\n主题 #{idx+1}:")
        print(topic)

    # 主题模型可视化
    print("\n生成主题模型可视化(浏览器自动打开)...")
    EnhancedVisualizer.visualize_lda(lda, corpus, dictionary, "lda_visualization_avengers.html")
    EnhancedVisualizer.visualize_lda(lda2, corpus2, dictionary2, "lda_visualization_thunderbolts.html")

    # 主题词柱形图可视化
    print("\n生成主题高频词柱形图...")
    EnhancedVisualizer.plot_topic_words(lda)
    EnhancedVisualizer.plot_topic_words(lda2)

if __name__ == "__main__":
    task1()
