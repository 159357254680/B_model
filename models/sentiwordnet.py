"""SentiWordNet 情感分析 —— 基于 WordNet 情感词典打分"""
import warnings
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore", message=".*pathsec.*")


def _ensure_swn():
    import nltk
    import os
    nltk_data = os.path.expanduser("~/nltk_data")
    if os.path.exists(nltk_data):
        nltk.data.path.insert(0, nltk_data)
    try:
        from nltk.corpus import sentiwordnet as swn
        list(swn.all_senti_synsets())
    except LookupError:
        nltk.download("sentiwordnet", download_dir=nltk_data, quiet=True)
        nltk.download("wordnet", download_dir=nltk_data, quiet=True)


def _word_sentiment(word):
    from nltk.corpus import sentiwordnet as swn
    synsets = list(swn.senti_synsets(word))
    if not synsets:
        return 0.0, 0.0, 0.0
    pos = sum(s.pos_score() for s in synsets) / len(synsets)
    neg = sum(s.neg_score() for s in synsets) / len(synsets)
    obj = sum(s.obj_score() for s in synsets) / len(synsets)
    return pos, neg, obj


def _text_sentiwordnet_score(text):
    words = text.lower().split()
    pos_total, neg_total = 0.0, 0.0
    count = 0
    for w in words:
        p, n, o = _word_sentiment(w)
        if p + n + o > 0:
            pos_total += p
            neg_total += n
            count += 1
    if count == 0:
        return {"pos_score": 0.0, "neg_score": 0.0, "sentiment": 0.0, "coverage": 0.0}
    return {
        "pos_score": round(pos_total / count, 4),
        "neg_score": round(neg_total / count, 4),
        "sentiment": round((pos_total - neg_total) / count, 4),
        "coverage": round(count / len(words), 4),
    }


def train_sentiwordnet(x_train, y_train, x_test, y_test, vectorizer=None, **_kw):
    print("  SentiWordNet: 基于 WordNet 情感词典打分 ...")
    _ensure_swn()

    if vectorizer is None:
        raise ValueError("SentiWordNet 需要 vectorizer 还原文本")

    texts = vectorizer.inverse_transform(x_test)
    texts = [" ".join(words) for words in texts]

    scores = [_text_sentiwordnet_score(t) for t in texts]
    preds = np.array([1 if s["sentiment"] > 0 else 0 for s in scores])
    probas = np.column_stack([1 - preds, preds]).astype(float)

    metrics = {
        "model": "SentiWordNet",
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "avg_pos_score": round(float(np.mean([s["pos_score"] for s in scores])), 4),
        "avg_neg_score": round(float(np.mean([s["neg_score"] for s in scores])), 4),
        "avg_coverage": round(float(np.mean([s["coverage"] for s in scores])), 4),
    }
    return None, metrics, preds, probas
