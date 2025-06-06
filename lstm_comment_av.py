import re
import pandas as pd
import numpy as np
import jieba
import matplotlib.pyplot as plt
from data_handling import DoubanTextCleaner
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# 1. 数据加载和预处理
df = pd.read_csv('Avengers-Endgame_comments.csv')

# 假设CSV列名为'Comment'和'Type'
comments = df['Comment'].values
types = df['Type'].values

# 初始化文本清洗器
text_cleaner = DoubanTextCleaner()

# 文本清洗和分词
def preprocess_text(text):
    # 使用DoubanTextCleaner进行预处理
    return text_cleaner.preprocess_for_analysis(text)

# 处理可能的NaN值
comments = [comment if pd.notna(comment) else '' for comment in comments]
comments = [preprocess_text(str(comment)) for comment in comments]
comments = np.array(comments)  # 转换为numpy数组

# 2. 划分数据集
# 训练集：好评和差评
train_mask = np.isin(types, ['好评', '差评'])
train_mask = np.array(train_mask)  # 确保是numpy数组
X_train = comments[train_mask]
y_train = types[train_mask]

# 待分类数据：一般
predict_mask = types == '一般'
X_predict = comments[predict_mask]

# 标签编码
label_map = {'差评': 0, '好评': 1}
y_train = np.array([label_map[label] for label in y_train])
y_train = to_categorical(y_train)

# 3. 文本向量化
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_train_pad = pad_sequences(X_train_seq, maxlen=200)

# 4. 构建LSTM模型
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=200))
model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 5. 训练模型并可视化
history = model.fit(X_train_pad, y_train, 
          batch_size=32,
          epochs=50,
          validation_split=0.2)

# 绘制训练和验证的准确率曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.title('模型准确率')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 绘制训练和验证的损失曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('模型损失')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('av_training_metrics.png')
plt.show()

# 6. 预测一般评论
X_predict_seq = tokenizer.texts_to_sequences(X_predict)
X_predict_pad = pad_sequences(X_predict_seq, maxlen=200)
predictions = model.predict(X_predict_pad)

# 将预测结果映射回标签
predicted_labels = ['好评' if pred[1] > 0.5 else '差评' for pred in predictions]

# 7. 保存结果
df.loc[predict_mask, 'Predicted_Type'] = predicted_labels
df.to_csv('av_classified_comments.csv', index=False)

# 8. 输出一般评论及其分类结果
general_comments = df[df['Type'] == '一般']
print("\n一般评论及其分类结果:")
# for idx, row in general_comments.iterrows():
#     print(f"评论: {row['Comment']} -> 分类: {row['Predicted_Type']}")

# 计算好评占比
good_ratio = len(general_comments[general_comments['Predicted_Type'] == '好评']) / len(general_comments)
print(f"\n一般评论中好评占比: {good_ratio:.2%}")

print("\n分类完成，结果已保存到av_classified_comments.csv")
