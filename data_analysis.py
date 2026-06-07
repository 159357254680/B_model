"""数据分析 —— 训练集分布、Top10词频、PMI、词性分析"""
import os
import json
import numpy as np
from collections import Counter
import nltk
from config import (RAW_TRAIN_PATH, RAW_DEV_PATH, RAW_TEST_PATH,
                    RAW_LABELS_TRAIN_PATH, RAW_LABELS_DEV_PATH, RAW_LABELS_TEST_PATH,
                    B_DIR)

OUT_DIR = os.path.join(B_DIR, "analysis")
os.makedirs(OUT_DIR, exist_ok=True)


def _ensure_nltk():
    nltk_data = os.path.expanduser("~/nltk_data")
    if os.path.exists(nltk_data):
        nltk.data.path.insert(0, nltk_data)
    for res in ["averaged_perceptron_tagger_eng", "punkt_tab", "punkt"]:
        try:
            nltk.data.find(f"tokenizers/{res}") if "tokenizers" in str(res) else nltk.data.find(f"taggers/{res}")
        except LookupError:
            try:
                nltk.download(res, download_dir=nltk_data, quiet=True)
            except Exception:
                pass


def _tokenize(text):
    return nltk.word_tokenize(text)


def _pos_analysis(tokens):
    tagged = nltk.pos_tag(tokens)
    pos_counts = Counter(tag for _, tag in tagged)
    total = sum(pos_counts.values())
    return {tag: round(cnt / total, 4) for tag, cnt in pos_counts.most_common()}


def _pmi(words_pos, words_neg, topk=20):
    all_pos = Counter()
    all_neg = Counter()
    for w, c in words_pos.items():
        all_pos[w] += c
    for w, c in words_neg.items():
        all_neg[w] += c
    all_words = all_pos + all_neg
    total = sum(all_words.values())
    pmi_scores = {}
    for word, count in all_words.items():
        p_w = count / total
        if p_w < 0.00001:
            continue
        p_w_given_pos = all_pos.get(word, 0) / sum(all_pos.values()) if sum(all_pos.values()) > 0 else 0
        p_w_given_neg = all_neg.get(word, 0) / sum(all_neg.values()) if sum(all_neg.values()) > 0 else 0
        pmi_pos = np.log2(p_w_given_pos / p_w) if p_w_given_pos > 0 else -10
        pmi_neg = np.log2(p_w_given_neg / p_w) if p_w_given_neg > 0 else -10
        pmi_scores[word] = max(pmi_pos, pmi_neg)  # 取与任一类的最大关联
    return sorted(pmi_scores.items(), key=lambda x: -x[1])[:topk]


def main():
    print("=" * 50)
    print("数据分析: IMDB 训练集")
    print("=" * 50)
    _ensure_nltk()

    raw_train = np.load(RAW_TRAIN_PATH, allow_pickle=True)
    raw_dev = np.load(RAW_DEV_PATH, allow_pickle=True)
    raw_test = np.load(RAW_TEST_PATH, allow_pickle=True)
    y_train = np.load(RAW_LABELS_TRAIN_PATH, allow_pickle=True)
    y_dev = np.load(RAW_LABELS_DEV_PATH, allow_pickle=True)
    y_test = np.load(RAW_LABELS_TEST_PATH, allow_pickle=True)

    results = {}

    # 1. 数据集分布
    dist = {
        "train": {"total": int(len(y_train)), "positive": int(y_train.sum()), "negative": int(len(y_train) - y_train.sum())},
        "dev":   {"total": int(len(y_dev)),   "positive": int(y_dev.sum()),   "negative": int(len(y_dev) - y_dev.sum())},
        "test":  {"total": int(len(y_test)),  "positive": int(y_test.sum()),  "negative": int(len(y_test) - y_test.sum())},
    }
    results["distribution"] = dist
    print(f"\n数据集分布:")
    for name, d in dist.items():
        print(f"  {name}: total={d['total']}, pos={d['positive']}, neg={d['negative']}")

    # 2. 每种情感的 Top10 词
    pos_texts = [raw_train[i] for i in range(len(raw_train)) if y_train[i] == 1]
    neg_texts = [raw_train[i] for i in range(len(raw_train)) if y_train[i] == 0]

    pos_words = Counter()
    for t in pos_texts:
        pos_words.update(_tokenize(t))
    neg_words = Counter()
    for t in neg_texts:
        neg_words.update(_tokenize(t))

    # 过滤停用词
    from nltk.corpus import stopwords
    try:
        sw = set(stopwords.words("english"))
    except LookupError:
        sw = {"the", "a", "an", "is", "of", "and", "to", "in", "it", "that", "was", "for", "on", "are", "with", "as", "be", "this", "but", "by", "at", "from", "or", "an", "we", "our", "its", "if", "my", "me", "no", "not", "so", "all", "just", "about", "been", "has", "had", "were", "they", "he", "she", "his", "her", "you", "i", "have", "do", "does", "did", "will", "would", "can", "could", "what", "when", "where", "who", "which", "how", "then", "than", "some", "more", "very", "much", "also", "into", "other", "only"}

    pos_filtered = [(w, c) for w, c in pos_words.most_common(50) if w not in sw and len(w) >= 2]
    neg_filtered = [(w, c) for w, c in neg_words.most_common(50) if w not in sw and len(w) >= 2]
    pos_top10 = pos_filtered[:10]
    neg_top10 = neg_filtered[:10]
    results["top10_positive"] = [{"word": w, "count": c} for w, c in pos_top10]
    results["top10_negative"] = [{"word": w, "count": c} for w, c in neg_top10]
    print(f"\n正向 Top10 词: {', '.join(w for w,_ in pos_top10)}")
    print(f"负向 Top10 词: {', '.join(w for w,_ in neg_top10)}")

    # 3. PMI Top10
    pmi_top = _pmi(pos_words, neg_words, topk=10)
    results["pmi_top10"] = [{"word": w, "pmi": round(s, 2)} for w, s in pmi_top]
    print(f"\nPMI Top10: {', '.join(f'{w}({s:.1f})' for w,s in pmi_top)}")

    # 4. 词性分析
    all_tokens = []
    for t in raw_train[:5000]:
        all_tokens.extend(_tokenize(t))
    pos_dist = _pos_analysis(all_tokens)
    results["pos_distribution"] = pos_dist
    print(f"\n词性分布 (Top10):")
    for tag, ratio in list(pos_dist.items())[:10]:
        print(f"  {tag}: {ratio:.2%}")

    # 保存
    out_path = os.path.join(OUT_DIR, "data_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
