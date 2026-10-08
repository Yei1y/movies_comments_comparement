# -*- coding: utf-8 -*-
"""一键运行影评比较分析的完整流程。

依次执行三个任务：
    任务一  词云、高频词与 LDA 主题建模
    任务二  SnowNLP 情感极性分析
    任务三  四个分类模型的对比（LSTM / SVM / 朴素贝叶斯 / 逻辑回归）

全部产物写入 output/（图表在 output/figures/，数值表在 output/tables/）。
注意：任务三的 LSTM 需要 5 折 × 50 epoch 训练，单部影片在普通 CPU 上约需
30–45 分钟；若只想快速复现其余结果，可直接运行
``python scripts/model_comparison.py av`` 或单独执行前两个脚本。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))   # 让脚本目录内的模块可被导入

from model_comparison import task3 as model_task          # noqa: E402
from sentiment_snownlp import task2 as sentiment_task     # noqa: E402
from wordcloud_lda import task1 as wordcloud_task         # noqa: E402


def main():
    """整合所有分析模块。"""
    print("=== 影评分析系统 ===")

    print("\n=== 开始词云和主题分析 ===")
    wordcloud_task()

    print("\n=== 开始情感分析 ===")
    sentiment_task()

    print("\n=== 开始四模型对比 ===")
    model_task()

    print("\n=== 所有分析任务已完成 ===")
    print("结果文件已保存在 output/ 目录:")
    print("- output/figures/: 词云、高频词、主题图、训练曲线、模型对比图、pyLDAvis 交互页")
    print("- output/tables/ : 情感分析明细与汇总、各模型逐条预测结果与指标汇总")


if __name__ == "__main__":
    main()