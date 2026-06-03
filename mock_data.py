"""生成模拟 A 模块输出的数据，无需联网。

用法: python mock_data.py
效果: 在 A_output/ 下生成所有 A 应产出的文件
"""

import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

from config import A_DIR, X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH, Y_TEST_PATH, VECTORIZER_PATH

os.makedirs(A_DIR, exist_ok=True)

# 构造带有情感信号的假文本（不依赖任何外部数据）
POS_WORDS = "excellent amazing wonderful great fantastic brilliant loved perfect beautiful " \
            "superb outstanding masterpiece engaging touching hilarious refreshing clever " \
            "intelligent moving powerful unforgettable magnificent captivating entertaining"
NEG_WORDS = "terrible awful horrible boring worst waste disappointing dull poor bad stupid " \
            "ridiculous dreadful lame cheesy unwatchable crap garbage trash pathetic mediocre"

POS_LIST = POS_WORDS.split()
NEG_LIST = NEG_WORDS.split()
FILLER = "the movie was a film that i watched and it felt very much like a story about life " \
         "the acting the plot the direction the script the characters the dialogue the ending"


def _gen_review(label, min_words=30, max_words=200):
    """生成一条带情感信号的假评论。"""
    sentiment_words = POS_LIST if label == 1 else NEG_LIST
    n_words = np.random.randint(min_words, max_words)
    words = []
    for _ in range(n_words):
        if np.random.random() < 0.3:  # 30% 概率插入情感词
            words.append(np.random.choice(sentiment_words))
        else:
            words.append(np.random.choice(FILLER.split()))
    return " ".join(words)


def make_mock_data(n_train=2000, n_test=500, n_features=5000):
    """生成二分类模拟数据集。"""
    print(f"生成模拟数据: train={n_train}, test={n_test}, features={n_features}")

    np.random.seed(42)

    # 生成平衡的标签和文本
    train_texts, train_labels = [], []
    for _ in range(n_train):
        label = np.random.randint(0, 2)
        train_texts.append(_gen_review(label))
        train_labels.append(label)

    test_texts, test_labels = [], []
    for _ in range(n_test):
        label = np.random.randint(0, 2)
        test_texts.append(_gen_review(label))
        test_labels.append(label)

    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    # TF-IDF 向量化
    vectorizer = TfidfVectorizer(max_features=n_features, stop_words="english")
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)

    # 保存
    sparse.save_npz(X_TRAIN_PATH, x_train)
    sparse.save_npz(X_TEST_PATH, x_test)
    np.save(Y_TRAIN_PATH, y_train)
    np.save(Y_TEST_PATH, y_test)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"A_output/ 生成完毕:")
    print(f"  x_train.npz  shape={x_train.shape}")
    print(f"  y_train.npy  shape={y_train.shape}, "
          f"pos={int(y_train.sum())}, neg={int(len(y_train)-y_train.sum())}")
    print(f"  x_test.npz   shape={x_test.shape}")
    print(f"  y_test.npy   shape={y_test.shape}, "
          f"pos={int(y_test.sum())}, neg={int(len(y_test)-y_test.sum())}")
    print(f"  vectorizer.pkl")


if __name__ == "__main__":
    make_mock_data()
