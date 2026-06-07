"""生成模拟 A 模块输出的数据，无需联网。

用法: python mock_data.py
效果: 在 A_output/ 下生成所有 A 应产出的文件（含 train/dev/test 三分）
"""

import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy import sparse

from config import (A_DIR, X_TRAIN_PATH, Y_TRAIN_PATH, X_DEV_PATH, Y_DEV_PATH,
                    X_TEST_PATH, Y_TEST_PATH, VECTORIZER_PATH,
                    RAW_TRAIN_PATH, RAW_DEV_PATH, RAW_TEST_PATH,
                    RAW_LABELS_TRAIN_PATH, RAW_LABELS_DEV_PATH, RAW_LABELS_TEST_PATH)

os.makedirs(A_DIR, exist_ok=True)

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
    sentiment_words = POS_LIST if label == 1 else NEG_LIST
    n_words = np.random.randint(min_words, max_words)
    words = []
    for _ in range(n_words):
        if np.random.random() < 0.3:
            words.append(np.random.choice(sentiment_words))
        else:
            words.append(np.random.choice(FILLER.split()))
    return " ".join(words)


def make_mock_data(n_total=2500, n_features=5000):
    print(f"生成模拟数据: total={n_total}, features={n_features}")

    np.random.seed(42)

    texts, labels_list = [], []
    for _ in range(n_total):
        label = np.random.randint(0, 2)
        texts.append(_gen_review(label))
        labels_list.append(label)

    texts = np.array(texts, dtype=object)
    labels = np.array(labels_list)

    # 70/15/15 三分
    x_temp, x_test_raw, y_temp, y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels,
    )
    x_train_raw, x_dev_raw, y_train, y_dev = train_test_split(
        x_temp, y_temp, test_size=0.15 / 0.85, random_state=42, stratify=y_temp,
    )

    vectorizer = TfidfVectorizer(max_features=n_features, stop_words="english")
    x_train = vectorizer.fit_transform(x_train_raw)
    x_dev = vectorizer.transform(x_dev_raw)
    x_test = vectorizer.transform(x_test_raw)

    # 保存
    sparse.save_npz(X_TRAIN_PATH, x_train)
    sparse.save_npz(X_DEV_PATH, x_dev)
    sparse.save_npz(X_TEST_PATH, x_test)
    np.save(Y_TRAIN_PATH, y_train)
    np.save(Y_DEV_PATH, y_dev)
    np.save(Y_TEST_PATH, y_test)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    np.save(RAW_TRAIN_PATH, x_train_raw)
    np.save(RAW_DEV_PATH, x_dev_raw)
    np.save(RAW_TEST_PATH, x_test_raw)
    np.save(RAW_LABELS_TRAIN_PATH, y_train)
    np.save(RAW_LABELS_DEV_PATH, y_dev)
    np.save(RAW_LABELS_TEST_PATH, y_test)

    print(f"A_output/ 生成完毕:")
    for name, arr in [("x_train.npz", x_train), ("x_dev.npz", x_dev), ("x_test.npz", x_test)]:
        print(f"  {name}  shape={arr.shape}")
    for name, arr in [("y_train.npy", y_train), ("y_dev.npy", y_dev), ("y_test.npy", y_test)]:
        print(f"  {name}  shape={arr.shape}, "
              f"pos={int(arr.sum())}, neg={int(len(arr)-arr.sum())}")
    print(f"  vectorizer.pkl + raw_*.npy")


if __name__ == "__main__":
    make_mock_data()
