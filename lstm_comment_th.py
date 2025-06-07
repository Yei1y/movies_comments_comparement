import re
import pandas as pd
import numpy as np
import jieba
import matplotlib.pyplot as plt
from data_handling import DoubanTextCleaner
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data(input_csv, comment_col='Comment', type_col='Type'):
    """加载数据并进行预处理"""
    df = pd.read_csv(input_csv)
    comments = df[comment_col].values
    types = df[type_col].values
    
    text_cleaner = DoubanTextCleaner()
    comments = [comment if pd.notna(comment) else '' for comment in comments]
    comments = [text_cleaner.preprocess_for_analysis(str(comment)) for comment in comments]
    return df, np.array(comments), types

def prepare_datasets(comments, types):
    """准备训练和预测数据集"""
    train_mask = np.isin(types, ['好评', '差评'])
    predict_mask = types == '一般'
    
    X_train = comments[train_mask]
    y_train = types[train_mask]
    X_predict = comments[predict_mask]
    
    label_map = {'差评': 0, '好评': 1}
    y_train = np.array([label_map[label] for label in y_train])
    y_train = to_categorical(y_train)
    
    return X_train, y_train, X_predict, predict_mask

def build_lstm_model(vocab_size=5000, max_len=200):
    """构建LSTM模型"""
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len))
    model.add(LSTM(128, dropout=0.4, recurrent_dropout=0.4))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                 optimizer='adamw',
                 metrics=['accuracy'])
    return model

def train_and_evaluate(model, X_train_pad, y_train, epochs=50, batch_size=64):
    """训练模型并评估"""
    history = model.fit(X_train_pad, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_split=0.2)
    return history

def visualize_training(history, output_img):
    """可视化训练过程"""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title('模型准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_img)
    plt.show()

def predict_and_save(model, tokenizer, df, X_predict, predict_mask, output_csv):
    """预测并保存结果"""
    X_predict_seq = tokenizer.texts_to_sequences(X_predict)
    X_predict_pad = pad_sequences(X_predict_seq, maxlen=200)
    predictions = model.predict(X_predict_pad)
    predicted_labels = ['好评' if pred[1] > 0.5 else '差评' for pred in predictions]
    
    df.loc[predict_mask, 'Predicted_Type'] = predicted_labels
    df.to_csv(output_csv, index=False)
    
    general_comments = df[df['Type'] == '一般']
    good_ratio = len(general_comments[general_comments['Predicted_Type'] == '好评']) / len(general_comments)
    print(f"\n一般评论中好评占比: {good_ratio:.2%}")
    print(f"\n分类完成，结果已保存到{output_csv}")

def run_lstm_analysis(input_csv, output_csv, metrics_img, comment_col='Comment', type_col='Type'):
    """主函数：运行完整的LSTM分析流程"""
    # 1. 数据加载和预处理
    df, comments, types = load_and_preprocess_data(input_csv, comment_col, type_col)
    
    # 2. 准备数据集
    X_train, y_train, X_predict, predict_mask = prepare_datasets(comments, types)
    
    # 3. 文本向量化
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_train_pad = pad_sequences(X_train_seq, maxlen=200)
    
    # 4. 构建和训练模型
    model = build_lstm_model()
    history = train_and_evaluate(model, X_train_pad, y_train)
    
    # 5. 可视化训练过程
    visualize_training(history, metrics_img)
    
    # 6. 预测并保存结果
    predict_and_save(model, tokenizer, df, X_predict, predict_mask, output_csv)

def task4():
    """
    任务4：LSTM文本分类
    读取雷霆特工队的评论数据，进行LSTM文本分类。
    """

    # 分析评论
    results = run_lstm_analysis(
        input_csv='Thunderbolts_comments.csv',
        output_csv='th_classified_comments.csv',
        metrics_img='th_training_metrics_test.png'
    )
    
    print(f"情感分析结果: {results}")

if __name__ == "__main__":
    task4()
    print("LSTM文本分类任务已完成。")

