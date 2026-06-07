"""SentiWordNet 结果分析 —— 用 SentiWordNet 分析所有模型（除产生式系统外）的输出"""
import os
import json
import warnings
import numpy as np
from config import B_DIR, RAW_TEST_PATH, RAW_LABELS_TEST_PATH, PARAMS_PATH

warnings.filterwarnings("ignore", message=".*pathsec.*")

OUT_DIR = os.path.join(B_DIR, "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

ALL_PREDS_FILE = os.path.join(B_DIR, "all_predictions.json")


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
    pos_total, neg_total = 0.0, 0.0
    pos_words, neg_words, count = 0, 0, 0
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


def run_sentiwordnet_analysis():
    print("\n" + "=" * 50)
    print("SentiWordNet 结果分析")
    print("=" * 50)
    _ensure_swn()

    raw_test = np.load(RAW_TEST_PATH, allow_pickle=True)
    y_test = np.load(RAW_LABELS_TEST_PATH, allow_pickle=True)

    # 读取每个模型的预测
    if not os.path.exists(ALL_PREDS_FILE):
        print("未找到 all_predictions.json，跳过")
        return

    with open(ALL_PREDS_FILE, "r") as f:
        all_preds = json.load(f)

    exclude = {"产生式规则系统", "大语言模型", "大语言模型(结构化)"}
    analysis = {}

    for model_name, preds_list in all_preds.items():
        if model_name in exclude:
            continue
        preds = np.array(preds_list)
        n = min(len(raw_test), len(preds), 500)

        pred_pos = []
        pred_neg = []
        for i in range(n):
            swn = _text_swn_scores(str(raw_test[i]))
            if preds[i] == 1:
                pred_pos.append(swn)
            else:
                pred_neg.append(swn)

        avg_pos_in_pos = np.mean([s["pos_word_count"] for s in pred_pos]) if pred_pos else 0
        avg_pos_in_neg = np.mean([s["pos_word_count"] for s in pred_neg]) if pred_neg else 0
        avg_swn_in_pos = np.mean([s["sentiment"] for s in pred_pos]) if pred_pos else 0
        avg_swn_in_neg = np.mean([s["sentiment"] for s in pred_neg]) if pred_neg else 0

        analysis[model_name] = {
            "pred_positive_count": len(pred_pos),
            "pred_negative_count": len(pred_neg),
            "avg_pos_words_when_pred_positive": round(float(avg_pos_in_pos), 2),
            "avg_pos_words_when_pred_negative": round(float(avg_pos_in_neg), 2),
            "avg_swn_sentiment_when_pred_positive": round(float(avg_swn_in_pos), 4),
            "avg_swn_sentiment_when_pred_negative": round(float(avg_swn_in_neg), 4),
            "sentiment_gap": round(float(avg_swn_in_pos - avg_swn_in_neg), 4),
        }
        print(f"  {model_name}: pred_pos={len(pred_pos)}, "
              f"pos_words(pos_pred)={avg_pos_in_pos:.1f}, "
              f"pos_words(neg_pred)={avg_pos_in_neg:.1f}, "
              f"gap={avg_swn_in_pos - avg_swn_in_neg:.4f}")

    out_path = os.path.join(OUT_DIR, "sentiwordnet_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f"  结果已保存: {out_path}")


def main():
    run_sentiwordnet_analysis()


if __name__ == "__main__":
    main()
