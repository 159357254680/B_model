"""A 模块：IMDB 真实数据预处理 —— 下载、清洗、划分、TF-IDF 向量化，输出到 A_output/"""
import os
import re
import pickle
import tarfile
import urllib.request
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy import sparse

from config import A_DIR, X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH, Y_TEST_PATH, VECTORIZER_PATH

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
    print(f"下载 IMDB 数据集 (约 80MB)...")
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

    texts = np.array(data["text"])
    labels = np.array(data["label"])

    # 8:2 分层划分
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels,
    )
    print(f"train: {len(x_train_raw)}, test: {len(x_test_raw)}")

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    x_train = vectorizer.fit_transform(x_train_raw)
    x_test = vectorizer.transform(x_test_raw)
    print(f"词汇量: {len(vectorizer.vocabulary_)}, x_train: {x_train.shape}, x_test: {x_test.shape}")

    sparse.save_npz(X_TRAIN_PATH, x_train)
    sparse.save_npz(X_TEST_PATH, x_test)
    np.save(Y_TRAIN_PATH, y_train)
    np.save(Y_TEST_PATH, y_test)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"A_output/ 已就绪:")
    print(f"  x_train.npz  shape={x_train.shape}")
    print(f"  y_train.npy  shape={y_train.shape}, 分布={dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  x_test.npz   shape={x_test.shape}")
    print(f"  y_test.npy   shape={y_test.shape}, 分布={dict(zip(*np.unique(y_test, return_counts=True)))}")
    print(f"  vectorizer.pkl")


if __name__ == "__main__":
    main()
