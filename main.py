import os
from data_handling import DoubanTextCleaner
from wordcloudTopsis import task1 as wordcloud_task
from sentiment_analysis import task2 as sentiment_task
from lstm_comment_av import task3 as lstm_av_task
from lstm_comment_th import task4 as lstm_th_task
from rnn_comment import task5 as rnn_task

def main():
    """主程序：整合所有分析模块"""
    print("=== 影评分析系统 ===")
    
    # 1. 初始化文本处理器
    print("\n初始化文本处理器...")
    cleaner = DoubanTextCleaner()
    
    # 2. 词云和主题分析
    print("\n=== 开始词云和主题分析 ===")
    wordcloud_task()
    
    # 3. 情感分析
    print("\n=== 开始情感分析 ===")
    sentiment_task()
    
    # 4. LSTM分类
    print("\n=== 开始LSTM分类 ===")
    print("\n分析复仇者联盟评论...")
    lstm_av_task()
    print("\n分析雷霆特工队评论...")
    lstm_th_task()
    
    # 5. RNN分类
    print("\n=== 开始RNN分类 ===")
    rnn_task()
    
    print("\n=== 所有分析任务已完成 ===")
    print("结果文件已保存在当前目录:")
    print("- av_classified_comments.csv: 复仇者联盟LSTM分类结果")
    print("- th_classified_comments.csv: 雷霆特工队LSTM分类结果")
    print("- av_classified_comments_rnn.csv: 复仇者联盟RNN分类结果")
    print("- th_classified_comments_rnn.csv: 雷霆特工队RNN分类结果")
    print("- lda_visualization.html: 主题模型可视化")
    print("- *.png: 各种分析图表")

if __name__ == "__main__":
    main()
