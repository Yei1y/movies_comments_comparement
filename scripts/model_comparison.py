# -*- coding: utf-8 -*-
"""任务三：四种分类模型在"用好评/差评训练、预测一般评论"任务上的对比。

作者标注的三类短评中，"好评 / 差评"作为有监督训练集，"一般"作为待判别集合：
模型学到的判别边界被用来回答一个实质问题——**那些难以判断的中间派评论，
在语义上更接近好评还是差评？**

对比的四个模型
- LSTM        ：Embedding + LSTM + 全连接，Keras 实现（深度学习基线）
- SVM         ：线性核支持向量机
- 朴素贝叶斯   ：MultinomialNB
- 逻辑回归     ：L2 正则 LogisticRegression

后三者共用 TF-IDF（5000 维）特征。全部模型均以 5 折交叉验证评估，
折内独立拟合分词器/TF-IDF，避免验证折信息泄漏到训练过程。

产出
- output/tables/{prefix}_{model}.csv        逐条预测结果（含 Predicted_Type）
- output/tables/{prefix}_model_metrics.csv  各模型 5 折准确率与"一般"判好评比例
- output/figures/{prefix}_comparison.png    模型验证准确率与预测好评率对比
- output/figures/{prefix}_lstm_training.png LSTM 首折训练曲线

用法::

    python scripts/model_comparison.py av     # 只跑复仇者联盟4
    python scripts/model_comparison.py both   # 两部影片都跑（默认）
"""
from pathlib import Path
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from text_cleaner import DoubanTextCleaner, load_comments

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / 'output' / 'figures'
TABLE_DIR = ROOT / 'output' / 'tables'

MOVIES = {
    'av': {'title': '复仇者联盟4', 'file': 'Avengers-Endgame_comments.csv',
           'prefix': 'av_classified_comments'},
    'th': {'title': '雷霆特工队', 'file': 'Thunderbolts_comments.csv',
           'prefix': 'th_classified_comments'},
}

RANDOM_SEED = 42
N_SPLITS = 5
MAX_LEN = 200
VOCAB_SIZE = 5000
EPOCHS = 50
BATCH_SIZE = 64

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 延迟导入 Keras：环境未装 TensorFlow 时其余三个模型仍可运行
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')


def _keras():
    """按需导入 Keras 组件，避免无谓的启动开销。

    统一走 ``tensorflow.keras`` 命名空间：Keras 3 已把 ``Tokenizer`` 等文本
    预处理工具移出顶层 ``keras.preprocessing``，而 ``tf.keras`` 仍保留兼容入口。
    """
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.utils import to_categorical
    return dict(keras=tf.keras, Dense=Dense, Dropout=Dropout, Embedding=Embedding,
                LSTM=LSTM, Sequential=Sequential, pad_sequences=pad_sequences,
                Tokenizer=Tokenizer, to_categorical=to_categorical)


def set_seed(seed=RANDOM_SEED):
    """固定全部随机源，保证交叉验证划分与网络初始化可复现。"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        _keras()['keras'].utils.set_random_seed(seed)
    except ImportError:
        pass


def load_and_preprocess_data(movie):
    """读取评论并做统一预处理，返回 (原始 DataFrame, 词序列数组, 标签数组)。"""
    df = load_comments(movie['file'])
    cleaner = DoubanTextCleaner()
    comments = [cleaner.preprocess_for_analysis(str(c)) for c in df['Comment'].fillna('')]
    return df, np.array(comments), df['Type'].values


def prepare_datasets(comments, types):
    """切分训练集（好评/差评）与待判别集（一般）。"""
    train_mask = np.isin(types, ['好评', '差评'])
    predict_mask = types == '一般'

    X_train = comments[train_mask]
    y_train = np.array([{'差评': 0, '好评': 1}[t] for t in types[train_mask]])
    X_predict = comments[predict_mask]
    return X_train, y_train, X_predict, predict_mask


# --------------------------------------------------------------------------- #
# LSTM
# --------------------------------------------------------------------------- #
def build_lstm_model(vocab_size=VOCAB_SIZE, max_len=MAX_LEN):
    """构建 LSTM 分类器（结构与原项目保持一致）。

    说明：Embedding -> LSTM(8) -> Dense(2, relu) -> Dropout(0.6) -> softmax，
    其中 Dense(2, relu) 把 LSTM 输出压到 2 维再送入 softmax，相当于一个
    低维投影头，用于抑制过拟合。该结构在本项目的样本规模下表现稳定，
    但并非通用设计，详见 README 的"已知问题"。
    """
    k = _keras()
    model = k['Sequential']([
        k['Embedding'](input_dim=vocab_size, output_dim=256, input_length=max_len),
        k['LSTM'](8, dropout=0.2, recurrent_dropout=0.2),
        k['Dense'](2, activation='relu'),
        k['Dropout'](0.6),
        k['Dense'](2, activation='softmax'),
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adamw', metrics=['accuracy'])
    return model


def train_lstm_cv(X_train, y_train, verbose=0):
    """5 折交叉验证训练 LSTM：每折重建模型与分词器，避免跨折信息泄漏。"""
    k = _keras()
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    train_accs, val_accs, histories = [], [], []
    model = tokenizer = None

    for fold_no, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train), start=1):
        print(f'  [LSTM] 第 {fold_no}/{N_SPLITS} 折训练中...')

        tokenizer = k['Tokenizer'](num_words=VOCAB_SIZE)
        tokenizer.fit_on_texts(X_train[train_idx])
        X_tr = k['pad_sequences'](tokenizer.texts_to_sequences(X_train[train_idx]), maxlen=MAX_LEN)
        X_va = k['pad_sequences'](tokenizer.texts_to_sequences(X_train[val_idx]), maxlen=MAX_LEN)

        y_tr = k['to_categorical'](y_train[train_idx])
        y_va = k['to_categorical'](y_train[val_idx])

        model = build_lstm_model()
        history = model.fit(X_tr, y_tr, batch_size=BATCH_SIZE, epochs=EPOCHS,
                            validation_data=(X_va, y_va), verbose=verbose)

        _, train_acc = model.evaluate(X_tr, y_tr, verbose=0)
        _, val_acc = model.evaluate(X_va, y_va, verbose=0)
        train_accs.append(float(train_acc))
        val_accs.append(float(val_acc))
        histories.append(history)

    metrics = {
        'train_accuracy': float(np.mean(train_accs)),
        'train_accuracy_std': float(np.std(train_accs)),
        'val_accuracy': float(np.mean(val_accs)),
        'val_accuracy_std': float(np.std(val_accs)),
        'fold_val_accuracies': [round(a, 4) for a in val_accs],
    }
    print(f"  [LSTM] 训练准确率 {metrics['train_accuracy']:.4f} ± {metrics['train_accuracy_std']:.4f}"
          f" | 验证准确率 {metrics['val_accuracy']:.4f} ± {metrics['val_accuracy_std']:.4f}")
    return model, tokenizer, metrics, histories[0]


# --------------------------------------------------------------------------- #
# 传统机器学习模型
# --------------------------------------------------------------------------- #
ML_MODELS = {
    'SVM': lambda: SVC(kernel='linear', probability=True, random_state=RANDOM_SEED),
    'NB': lambda: MultinomialNB(),
    'LR': lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
}
ML_LABELS = {'SVM': 'SVM', 'NB': '朴素贝叶斯', 'LR': '逻辑回归'}


def train_ml_cv(build_model, X_train, y_train, model_name, verbose=True):
    """5 折交叉验证训练 TF-IDF + 传统分类器，折内独立拟合向量器。"""
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    train_accs, val_accs = [], []
    model = vectorizer = None

    for fold_no, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train), start=1):
        if verbose:
            print(f'  [{model_name}] 第 {fold_no}/{N_SPLITS} 折训练中...')

        vectorizer = TfidfVectorizer(max_features=VOCAB_SIZE)
        X_tr = vectorizer.fit_transform(X_train[train_idx])
        X_va = vectorizer.transform(X_train[val_idx])

        model = build_model()
        model.fit(X_tr, y_train[train_idx])
        train_accs.append(float(accuracy_score(y_train[train_idx], model.predict(X_tr))))
        val_accs.append(float(accuracy_score(y_train[val_idx], model.predict(X_va))))

    metrics = {
        'train_accuracy': float(np.mean(train_accs)),
        'train_accuracy_std': float(np.std(train_accs)),
        'val_accuracy': float(np.mean(val_accs)),
        'val_accuracy_std': float(np.std(val_accs)),
        'fold_val_accuracies': [round(a, 4) for a in val_accs],
    }
    if verbose:
        print(f"  [{ML_LABELS.get(model_name, model_name)}] 训练准确率 "
              f"{metrics['train_accuracy']:.4f} ± {metrics['train_accuracy_std']:.4f}"
              f" | 验证准确率 {metrics['val_accuracy']:.4f} ± {metrics['val_accuracy_std']:.4f}")
    return model, vectorizer, metrics


# --------------------------------------------------------------------------- #
# 预测与可视化
# --------------------------------------------------------------------------- #
def predict_general(model, processor, X_predict, model_type):
    """对"一般"评论给出好评/差评判别结果。"""
    if model_type == 'lstm':
        k = _keras()
        X = k['pad_sequences'](processor.texts_to_sequences(X_predict), maxlen=MAX_LEN)
        proba = model.predict(X, verbose=0)
        return ['好评' if p[1] > 0.5 else '差评' for p in proba]

    X = processor.transform(X_predict)
    return ['好评' if p == 1 else '差评' for p in model.predict(X)]


def plot_comparison(metrics_dict, output_path):
    """绘制各模型的验证准确率与预测好评率对比图。"""
    models = list(metrics_dict)
    val_acc = [metrics_dict[m]['val_accuracy'] for m in models]
    good_ratio = [metrics_dict[m]['good_ratio'] for m in models]
    colors = ['#4DBBD5', '#F39B7F', '#00A087', '#91D1C2'][:len(models)]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(models, val_acc, color=colors)
    for i, v in enumerate(val_acc):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
    plt.title('模型验证准确率比较')
    plt.xlabel('模型')
    plt.ylabel('验证准确率')
    plt.ylim(0, 1)

    plt.subplot(1, 2, 2)
    plt.bar(models, good_ratio, color=colors)
    for i, v in enumerate(good_ratio):
        plt.text(i, v + 0.02, f'{v:.1%}', ha='center')
    plt.title('模型预测好评率比较')
    plt.xlabel('模型')
    plt.ylabel('好评率')
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_training_curve(history, output_path):
    """绘制 LSTM 首折的训练/验证准确率与损失曲线。"""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title('模型准确率（第 1 折）')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型损失（第 1 折）')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run_analysis(movie_key):
    """对一部影片运行四模型对比全流程，返回指标字典。"""
    movie = MOVIES[movie_key]
    prefix = movie['prefix']
    print(f"\n{'=' * 60}\n《{movie['title']}》模型对比\n{'=' * 60}")

    set_seed()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    df, comments, types = load_and_preprocess_data(movie)
    X_train, y_train, X_predict, predict_mask = prepare_datasets(comments, types)
    print(f"训练集（好评/差评）{len(X_train)} 条，待判别集（一般）{len(X_predict)} 条")

    metrics_dict = {}

    # 1) LSTM
    print("\n--- LSTM ---")
    lstm_model, lstm_tokenizer, lstm_metrics, history = train_lstm_cv(X_train, y_train)
    metrics_dict['LSTM'] = lstm_metrics
    _save_predictions(df, predict_mask,
                      predict_general(lstm_model, lstm_tokenizer, X_predict, 'lstm'),
                      TABLE_DIR / f'{prefix}_lstm.csv', metrics_dict['LSTM'])
    plot_training_curve(history, FIG_DIR / f'{prefix}_lstm_training.png')

    # 2) 传统机器学习模型
    for key, builder in ML_MODELS.items():
        print(f"\n--- {ML_LABELS[key]} ---")
        model, vectorizer, ml_metrics = train_ml_cv(builder, X_train, y_train, key)
        metrics_dict[key] = ml_metrics
        _save_predictions(df, predict_mask,
                          predict_general(model, vectorizer, X_predict, 'ml'),
                          TABLE_DIR / f'{prefix}_{key.lower()}.csv', metrics_dict[key])

    # 3) 汇总与可视化
    summary = pd.DataFrame([
        {
            'movie': movie['title'],
            'model': name,
            'train_accuracy': round(m['train_accuracy'], 4),
            'train_accuracy_std': round(m['train_accuracy_std'], 4),
            'val_accuracy': round(m['val_accuracy'], 4),
            'val_accuracy_std': round(m['val_accuracy_std'], 4),
            'good_ratio': round(m['good_ratio'], 4),
            'n_general': m['n_general'],
            'n_general_pred_good': m['n_general_pred_good'],
            'fold_val_accuracies': '|'.join(f'{a:.4f}' for a in m['fold_val_accuracies']),
        }
        for name, m in metrics_dict.items()
    ])
    summary.to_csv(TABLE_DIR / f'{prefix}_model_metrics.csv',
                   index=False, encoding='utf-8-sig')
    plot_comparison(metrics_dict, FIG_DIR / f'{prefix}_comparison.png')

    print(f"\n《{movie['title']}》最终模型性能比较:")
    print(summary.to_string(index=False))
    return metrics_dict, summary


def _save_predictions(df, predict_mask, labels, output_csv, metrics):
    """把"一般"评论的判别结果写回 DataFrame 并落盘，同时记录好评比例。"""
    out = df.copy()
    out.loc[predict_mask, 'Predicted_Type'] = labels
    out.to_csv(output_csv, index=False, encoding='utf-8-sig')

    general = out[out['Type'] == '一般']
    n_good = int((general['Predicted_Type'] == '好评').sum())
    metrics['n_general'] = int(len(general))
    metrics['n_general_pred_good'] = n_good
    metrics['good_ratio'] = float(n_good / len(general)) if len(general) else float('nan')
    print(f"  一般评论中判为好评: {n_good}/{len(general)} ({metrics['good_ratio']:.2%})"
          f" -> {output_csv.name}")


def task3(movie_keys=('av', 'th')):
    """任务三入口：对指定影片运行四模型对比。"""
    set_seed()
    summaries = []
    for key in movie_keys:
        _, summary = run_analysis(key)
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='四模型文本分类对比')
    parser.add_argument('movie', nargs='?', default='both', choices=['av', 'th', 'both'],
                        help='av=复仇者联盟4, th=雷霆特工队, both=两部都跑（默认）')
    args = parser.parse_args()

    keys = tuple(MOVIES) if args.movie == 'both' else (args.movie,)
    task3(keys)
    print("\n模型对比任务已完成。")