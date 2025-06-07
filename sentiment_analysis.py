import pandas as pd
from snownlp import SnowNLP
from data_handling import DoubanTextCleaner

def analyze_sentiment(df):
    # 初始化文本清洗器
    cleaner = DoubanTextCleaner()
    
    # 情感分析结果存储
    results = {
        'positive': 0,
        'negative': 0,
        'neutral': 0,
        'total': 0
    }
    
    # 分析每条评论
    for comment in df['Comment'].dropna():
        try:
            # 预处理文本
            processed = cleaner.preprocess_for_analysis(comment)
            
            # 跳过空文本
            if not processed.strip():
                continue
                
            # 情感分析
            s = SnowNLP(processed)
            sentiment = s.sentiments
            
            # 分类情感
            if sentiment > 0.6:
                results['positive'] += 1
            elif sentiment < 0.4:
                results['negative'] += 1
            else:
                results['neutral'] += 1
            results['total'] += 1
            
        except Exception as e:
            print(f"分析评论时出错: {comment[:50]}... 错误: {str(e)}")
    
    # 计算百分比
    if results['total'] > 0:
        results['positive_pct'] = results['positive'] / results['total'] * 100
        results['negative_pct'] = results['negative'] / results['total'] * 100
        results['neutral_pct'] = results['neutral'] / results['total'] * 100
    else:
        results['positive_pct'] = 0
        results['negative_pct'] = 0
        results['neutral_pct'] = 0
    
    return results

def task2():
    """
    任务2：情感分析
    读取复仇者联盟4和雷霆特工队的评论数据，进行情感分析。
    """
    # 读取CSV文件
    df_avengers = pd.read_csv("Avengers-Endgame_comments.csv")
    df_thunderbolts = pd.read_csv("Thunderbolts_comments.csv")
    
    # 分析复仇者联盟数据
    print("=== 复仇者联盟4:终局之战 ===")
    avengers_analysis = analyze_sentiment(df_avengers)
    print(f"总评论数: {avengers_analysis['total']}")
    print(f"正面评论: {avengers_analysis['positive']} ({avengers_analysis['positive_pct']:.1f}%)")
    print(f"负面评论: {avengers_analysis['negative']} ({avengers_analysis['negative_pct']:.1f}%)")
    print(f"中性评论: {avengers_analysis['neutral']} ({avengers_analysis['neutral_pct']:.1f}%)")
    
    # 分析雷霆特工队数据
    print("\n=== 雷霆特工队 ===")
    thunderbolts_analysis = analyze_sentiment(df_thunderbolts)
    print(f"总评论数: {thunderbolts_analysis['total']}")
    print(f"正面评论: {thunderbolts_analysis['positive']} ({thunderbolts_analysis['positive_pct']:.1f}%)")
    print(f"负面评论: {thunderbolts_analysis['negative']} ({thunderbolts_analysis['negative_pct']:.1f}%)")
    print(f"中性评论: {thunderbolts_analysis['neutral']} ({thunderbolts_analysis['neutral_pct']:.1f}%)")

if __name__ == "__main__":
    task2()