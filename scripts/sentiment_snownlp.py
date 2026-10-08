# -*- coding: utf-8 -*-
"""任务二：基于 SnowNLP 的短评情感极性分析。

对每条短评计算 [0, 1] 区间的情感得分（越接近 1 越正面），并按
0.4 / 0.6 两个阈值划为负面 / 中性 / 正面三类。

产出
- output/tables/{movie}_sentiment_results.csv  逐条评论的情感得分与标签
- output/tables/sentiment_summary.csv          两部影片的汇总比例（便于核验）
"""
from pathlib import Path

import pandas as pd
from snownlp import SnowNLP

from text_cleaner import DoubanTextCleaner, load_comments

ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / 'output' / 'tables'

MOVIES = {
    'av': {'title': '复仇者联盟4', 'file': 'Avengers-Endgame_comments.csv',
           'out': 'avengers_sentiment_results.csv'},
    'th': {'title': '雷霆特工队', 'file': 'Thunderbolts_comments.csv',
           'out': 'thunderbolts_sentiment_results.csv'},
}

POSITIVE_THRESHOLD = 0.6
NEGATIVE_THRESHOLD = 0.4


def analyze_sentiment(df, cleaner=None):
    """逐条计算情感得分并归类，返回 (汇总统计, 明细 DataFrame)。"""
    cleaner = cleaner or DoubanTextCleaner()
    counts = {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0}
    detailed_results = []

    for idx, comment in df['Comment'].dropna().items():
        try:
            processed = cleaner.preprocess_for_analysis(comment)
            if not processed.strip():      # 清洗后为空的评论不参与统计
                continue

            sentiment = SnowNLP(processed).sentiments
            if sentiment > POSITIVE_THRESHOLD:
                label = 'positive'
            elif sentiment < NEGATIVE_THRESHOLD:
                label = 'negative'
            else:
                label = 'neutral'

            counts[label] += 1
            counts['total'] += 1
            detailed_results.append({
                'comment_id': idx,
                'comment': comment,
                'sentiment_score': sentiment,
                'sentiment_label': label,
            })
        except Exception as exc:
            print(f"分析评论时出错: {str(comment)[:50]}... 错误: {exc}")

    total = counts['total']
    for label in ('positive', 'negative', 'neutral'):
        counts[f'{label}_pct'] = counts[label] / total * 100 if total else 0.0

    return counts, pd.DataFrame(detailed_results)


def task2():
    """任务二入口：对两部影片做情感分析并落盘。"""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    cleaner = DoubanTextCleaner()
    summary_rows = []

    for key, movie in MOVIES.items():
        print(f"\n=== {movie['title']} ===")
        df = load_comments(movie['file'])
        analysis, details = analyze_sentiment(df, cleaner)

        print(f"总评论数: {analysis['total']}")
        print(f"正面评论: {analysis['positive']} ({analysis['positive_pct']:.1f}%)")
        print(f"负面评论: {analysis['negative']} ({analysis['negative_pct']:.1f}%)")
        print(f"中性评论: {analysis['neutral']} ({analysis['neutral_pct']:.1f}%)")

        details.to_csv(TABLE_DIR / movie['out'], index=False, encoding='utf-8-sig')
        summary_rows.append({
            'movie': movie['title'],
            'total_valid': analysis['total'],
            'skipped': len(df) - analysis['total'],
            'positive': analysis['positive'],
            'negative': analysis['negative'],
            'neutral': analysis['neutral'],
            'positive_pct': round(analysis['positive_pct'], 2),
            'negative_pct': round(analysis['negative_pct'], 2),
            'neutral_pct': round(analysis['neutral_pct'], 2),
            'mean_score': round(details['sentiment_score'].mean(), 4),
            'median_score': round(details['sentiment_score'].median(), 4),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLE_DIR / 'sentiment_summary.csv', index=False, encoding='utf-8-sig')
    print("\n情感分析汇总:")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    task2()
    print("\n情感分析任务已完成。")