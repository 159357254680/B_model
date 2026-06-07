"""A 模块：IMDB 真实数据预处理 —— 下载、清洗、三分划分、TF-IDF 向量化，输出到 A_output/"""
import os
import re
import pickle
import tarfile
import urllib.request
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy import sparse

from config import (A_DIR, X_TRAIN_PATH, Y_TRAIN_PATH, X_DEV_PATH, Y_DEV_PATH,
                    X_TEST_PATH, Y_TEST_PATH, VECTORIZER_PATH,
                    RAW_TRAIN_PATH, RAW_DEV_PATH, RAW_TEST_PATH,
                    RAW_LABELS_TRAIN_PATH, RAW_LABELS_DEV_PATH, RAW_LABELS_TEST_PATH)

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TAR_PATH = os.path.join(DATA_DIR, "aclImdb_v1.tar.gz")
EXTRACT_DIR = os.path.join(DATA_DIR, "aclImdb")

os.makedirs(A_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def _download():
    if os.path.exists(TAR_PATH):
        print(f"已缓存: {TAR_PATH}")
        return
    print("下载 IMDB 数据集 (约 80MB)...")
    urllib.request.urlretrieve(IMDB_URL, TAR_PATH)
    print("下载完成")


def _extract():
    if os.path.exists(EXTRACT_DIR):
        print(f"已解压: {EXTRACT_DIR}")
        return
    print("解压中...")
    with tarfile.open(TAR_PATH, "r:gz") as tar:
        tar.extractall(DATA_DIR, filter="data")
    print("解压完成")


def _clean(text):
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _load_reviews():
    data = {"text": [], "label": []}
    for split in ["train", "test"]:
        for label, folder in [(1, "pos"), (0, "neg")]:
            folder_path = os.path.join(EXTRACT_DIR, split, folder)
            if not os.path.exists(folder_path):
                continue
            for fname in os.listdir(folder_path):
                if fname.endswith(".txt"):
                    with open(os.path.join(folder_path, fname), encoding="utf-8") as f:
                        text = _clean(f.read())
                    data["text"].append(text)
                    data["label"].append(label)
    print(f"读取 {len(data['text'])} 条评论 (正={sum(data['label'])}, 负={len(data['label'])-sum(data['label'])})")
    return data


def main(max_features=5000):
    print("=" * 50)
    print("A 模块: IMDB 真实数据预处理")
    print("=" * 50)

    _download()
    _extract()
    data = _load_reviews()

    texts = np.array(data["text"], dtype=object)
    labels = np.array(data["label"])

    # 70/15/15 分层三分
    x_temp, x_test_raw, y_temp, y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels,
    )
    x_train_raw, x_dev_raw, y_train, y_dev = train_test_split(
        x_temp, y_temp, test_size=0.15 / 0.85, random_state=42, stratify=y_temp,
    )
    print(f"train: {len(x_train_raw)}, dev: {len(x_dev_raw)}, test: {len(x_test_raw)}")

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                                  stop_words="english", sublinear_tf=True)
    x_train = vectorizer.fit_transform(x_train_raw)
    x_dev = vectorizer.transform(x_dev_raw)
    x_test = vectorizer.transform(x_test_raw)
    print(f"词汇量: {len(vectorizer.vocabulary_)}, train: {x_train.shape}, dev: {x_dev.shape}, test: {x_test.shape}")

    # 保存稀疏特征
    sparse.save_npz(X_TRAIN_PATH, x_train)
    sparse.save_npz(X_DEV_PATH, x_dev)
    sparse.save_npz(X_TEST_PATH, x_test)
    np.save(Y_TRAIN_PATH, y_train)
    np.save(Y_DEV_PATH, y_dev)
    np.save(Y_TEST_PATH, y_test)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    # 保存原始文本供数据分析
    np.save(RAW_TRAIN_PATH, x_train_raw)
    np.save(RAW_DEV_PATH, x_dev_raw)
    np.save(RAW_TEST_PATH, x_test_raw)
    np.save(RAW_LABELS_TRAIN_PATH, y_train)
    np.save(RAW_LABELS_DEV_PATH, y_dev)
    np.save(RAW_LABELS_TEST_PATH, y_test)

    print(f"A_output/ 已就绪:")
    for name, arr in [("x_train.npz", x_train), ("x_dev.npz", x_dev), ("x_test.npz", x_test)]:
        print(f"  {name}  shape={arr.shape}")
    for name, arr in [("y_train.npy", y_train), ("y_dev.npy", y_dev), ("y_test.npy", y_test)]:
        print(f"  {name}  shape={arr.shape}, 分布={dict(zip(*np.unique(arr, return_counts=True)))}")
    print(f"  vectorizer.pkl  + raw_*.npy (原始文本)")


if __name__ == "__main__":
    main()
