# 影评比较分析系统

本项目对《复仇者联盟4》和《雷霆特工队》的豆瓣影评进行多维度分析比较，包括词云可视化、情感分析和深度学习分类。

## 功能模块

### 1. 数据预处理 (data_handling.py)
- 文本清洗：去除标点、表情符号、数字等
- 中文分词：使用jieba分词，加载自定义词典
- 停用词过滤
- 提供两种预处理模式：
  - 词云分析专用预处理
  - 情感分析/LSTM专用预处理

### 2. 词云与主题分析 (wordcloudTopsis.py)
- 生成词云图
- 绘制高频词条形图
- LDA主题建模
- 主题可视化(HTML)

### 3. 情感分析 (sentiment_analysis.py)
- 使用SnowNLP进行情感分析
- 统计正面/负面/中性评论比例
- 支持CSV数据输入

### 4. 深度学习分类
- LSTM模型 (lstm_comment_av.py, lstm_comment_th.py)
  - 对"好评"和"差评"进行训练
  - 预测"一般"评论的情感倾向
  - 可视化训练过程
- RNN模型 (rnn_comment.py)
  - 训练类同LSTM模型
  - 出现较明显的过拟合问题
- 主要比较了两个模型的差异

## 使用方法

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行完整分析流程：
```bash
python main.py
```

3. 查看结果文件：
- `*_classified_comments.csv`: 分类结果
- `*_training_metrics.png`: 训练指标图表
- `lda_visualization.html`: 主题模型可视化

## 项目结构

```
影评比较/
├── data_handling.py        # 文本预处理
├── wordcloudTopsis.py      # 词云和主题分析
├── sentiment_analysis.py   # 情感分析
├── lstm_comment_av.py      # 复仇者联盟LSTM分析
├── lstm_comment_th.py      # 雷霆特工队LSTM分析  
├── rnn_comment.py          # RNN对比分析
├── main.py                 # 主程序
├── README.md               # 项目说明
├── Avengers-Endgame_comments.csv    # 复仇者联盟评论数据
├── Thunderbolts_comments.csv        # 雷霆特工队评论数据
└── pic/                    # 生成的图表目录
```

## 注意事项

1. 确保已安装中文字体(如SimHei)以正确显示图表
2. 自定义词典(user.dict.utf8)和停用词表(stopwords_hit.txt)需放在项目目录
3. 首次运行会自动下载SnowNLP和jieba的词典
