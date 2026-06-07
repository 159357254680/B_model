"""SentiWordNet 结果分析 —— 用 SentiWordNet 分析所有模型（除产生式系统外）的输出"""
import os
import json
import numpy as np
from config import B_DIR, RAW_TEST_PATH, RAW_LABELS_TEST_PATH, PARAMS_PATH

OUT_DIR = os.path.join(B_DIR, "analysis")
os.makedirs(OUT_DIR, exist_ok=True)


def _ensure_swn():
    import nltk
    nltk_data = os.path.expanduser("~/nltk_data")
    if os.path.exists(nltk_data):
        nltk.data.path.insert(0, nltk_data)
    try:
        from nltk.corpus import sentiwordnet as swn
        list(swn.all_senti_synsets())
    except LookupError:
        nltk.download("sentiwordnet", download_dir=nltk_data, quiet=True)
        nltk.download("wordnet", download_dir=nltk_data, quiet=True)


def _text_swn_scores(text):
    from nltk.corpus import sentiwordnet as swn
    words = text.lower().split()
    pos_total, neg_total, pos_words, neg_words = 0.0, 0.0, 0, 0
    count = 0
    for w in words:
        synsets = list(swn.senti_synsets(w))
        if synsets:
            p = sum(s.pos_score() for s in synsets) / len(synsets)
            n = sum(s.neg_score() for s in synsets) / len(synsets)
            pos_total += p
            neg_total += n
            if p > n and p > 0:
                pos_words += 1
            elif n > p and n > 0:
                neg_words += 1
            count += 1
    if count == 0:
        return {"pos_word_count": 0, "neg_word_count": 0, "pos_score": 0.0, "neg_score": 0.0, "sentiment": 0.0}
    return {
        "pos_word_count": pos_words,
        "neg_word_count": neg_words,
        "pos_score": round(pos_total / count, 4),
        "neg_score": round(neg_total / count, 4),
        "sentiment": round((pos_total - neg_total) / count, 4),
    }


def main():
    print("=" * 50)
    print("SentiWordNet 结果分析")
    print("=" * 50)
    _ensure_swn()

    # 读取原始测试文本和标签
    raw_test = np.load(RAW_TEST_PATH, allow_pickle=True)
    y_test = np.load(RAW_LABELS_TEST_PATH, allow_pickle=True)

    # 读取模型结果
    params = {}
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH, "r", encoding="utf-8") as f:
            params = json.load(f)
    all_results = params.get("all_model_results", {})
    if not all_results:
        print("未找到模型结果，请先运行 train.py")
        return

    # 读取预测和概率
    preds = None
    probas = None
    preds_path = os.path.join(B_DIR, "predictions.npy")
    probas_path = os.path.join(B_DIR, "probas.npy")
    if os.path.exists(preds_path):
        preds = np.load(preds_path)
    if os.path.exists(probas_path):
        probas = np.load(probas_path)

    # 排除产生式规则系统
    exclude = {"产生式规则系统", "大语言模型"}
    models_to_analyze = [n for n in all_results if n not in exclude and "error" not in all_results[n]]

    analysis = {}
    for model_name in models_to_analyze:
        print(f"\n分析 {model_name} ...")
        # 对于每个模型，取测试集中预测为正向的样本，计算 SentiWordNet 相关度
        pos_samples = []
        neg_samples = []
        for i in range(min(len(raw_test), 500)):  # 抽样500条加速
            swn_scores = _text_swn_scores(str(raw_test[i]))
            true_label = int(y_test[i])
            pos_samples.append({
                "true_label": true_label,
                "swn_pos_words": swn_scores["pos_word_count"],
                "swn_neg_words": swn_scores["neg_word_count"],
                "swn_sentiment": swn_scores["sentiment"],
            })

        pos_pred = [s for s in pos_samples if s["true_label"] == 1]
        neg_pred = [s for s in pos_samples if s["true_label"] == 0]

        avg_pos_words_in_pos = np.mean([s["swn_pos_words"] for s in pos_pred]) if pos_pred else 0
        avg_pos_words_in_neg = np.mean([s["swn_pos_words"] for s in neg_pred]) if neg_pred else 0
        avg_swn_sentiment_pos = np.mean([s["swn_sentiment"] for s in pos_pred]) if pos_pred else 0
        avg_swn_sentiment_neg = np.mean([s["swn_sentiment"] for s in neg_pred]) if neg_pred else 0

        # 计算 SentiWordNet 预测准确率（sentiment > 0 → 正向）
        swn_preds = [1 if s["swn_sentiment"] > 0 else 0 for s in pos_samples]
        swn_true = [s["true_label"] for s in pos_samples]
        swn_acc = sum(1 for p, t in zip(swn_preds, swn_true) if p == t) / len(swn_true)

        analysis[model_name] = {
            "avg_pos_words_true_positive": round(float(avg_pos_words_in_pos), 2),
            "avg_pos_words_true_negative": round(float(avg_pos_words_in_neg), 2),
            "avg_swn_sentiment_true_positive": round(float(avg_swn_sentiment_pos), 4),
            "avg_swn_sentiment_true_negative": round(float(avg_swn_sentiment_neg), 4),
            "sentiwordnet_baseline_accuracy": round(float(swn_acc), 4),
        }
        print(f"  真实正向: avg_pos_words={avg_pos_words_in_pos:.1f}, swn_sentiment={avg_swn_sentiment_pos:.4f}")
        print(f"  真实负向: avg_pos_words={avg_pos_words_in_neg:.1f}, swn_sentiment={avg_swn_sentiment_neg:.4f}")
        print(f"  SentiWordNet 基线准确率: {swn_acc:.4f}")

    # 汇总对比
    summary = {}
    if preds is not None and probas is not None:
        for i in range(min(len(raw_test), 500)):
            s = _text_swn_scores(str(raw_test[i]))
            s["true_label"] = int(y_test[i])
            s["best_pred"] = int(preds[i]) if i < len(preds) else -1
            s["best_confidence"] = float(probas[i][1]) if i < len(probas) else 0
            summary[str(i)] = s

    out_path = os.path.join(OUT_DIR, "sentiwordnet_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_analysis": analysis,
            "per_sample_summary": {k: summary[k] for k in list(summary.keys())[:20]},
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
