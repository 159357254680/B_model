"""B 模块主程序 —— 读入 A 的数据，训练 8 种模型，自动运行全套分析，输出 C 需要的文件。

用法:
    python train.py              # 正常模式，读 A_output/，训练 + 数据分析 + 可视化
    python train.py --mock       # 先用 mock 数据自测

执行链路:
    ① 加载数据 → ② 训练 8 模型 → ③ 保存输出 → ④ 数据分析 → ⑤ SentiWordNet分析 → ⑥ 可视化
"""

import os
import json
import pickle
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics.cluster")
from scipy import sparse

import config


def load_a_output():
    print("=" * 50)
    print("B 模块: 加载 A 产出的数据")
    print("=" * 50)

    has_dev = os.path.exists(config.X_DEV_PATH)

    files = {
        "x_train": config.X_TRAIN_PATH,
        "y_train": config.Y_TRAIN_PATH,
        "x_test":  config.X_TEST_PATH,
        "y_test":  config.Y_TEST_PATH,
        "vectorizer": config.VECTORIZER_PATH,
    }
    if has_dev:
        files["x_dev"] = config.X_DEV_PATH
        files["y_dev"] = config.Y_DEV_PATH

    for name, path in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"缺少 {name} ({path})\n"
                f"请确认 A 模块已运行，或使用 --mock 生成模拟数据"
            )

    x_train = sparse.load_npz(config.X_TRAIN_PATH)
    y_train = np.load(config.Y_TRAIN_PATH)
    x_test  = sparse.load_npz(config.X_TEST_PATH)
    y_test  = np.load(config.Y_TEST_PATH)
    with open(config.VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    print(f"  x_train: {x_train.shape}, y_train: {y_train.shape}, "
          f"分布: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    if has_dev:
        x_dev = sparse.load_npz(config.X_DEV_PATH)
        y_dev = np.load(config.Y_DEV_PATH)
        print(f"  x_dev:   {x_dev.shape}, y_dev:   {y_dev.shape}, "
              f"分布: {dict(zip(*np.unique(y_dev, return_counts=True)))}")
    else:
        x_dev, y_dev = None, None
    print(f"  x_test:  {x_test.shape}, y_test:  {y_test.shape}, "
          f"分布: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    return x_train, y_train, x_dev, y_dev, x_test, y_test, vectorizer


def train_all_models(x_train, y_train, x_test, y_test, vectorizer):
    from models.rule_based import train_rule_based
    from models.naive_bayes import train_naive_bayes
    from models.logistic_regression import train_logistic_regression
    from models.feature_selection import train_with_feature_selection
    from models.logistic_regression_features import train_logistic_regression_features
    from models.sentiwordnet import train_sentiwordnet
    from models.llm import train_llm, train_llm_structured

    models = [
        ("产生式规则系统", train_rule_based),
        ("朴素贝叶斯",     train_naive_bayes),
        ("逻辑回归",       train_logistic_regression),
        ("特征选择+LR",    train_with_feature_selection),
        ("新特征+LR",      train_logistic_regression_features),
        ("SentiWordNet",   train_sentiwordnet),
        ("大语言模型",     train_llm),
        ("大语言模型(结构化)", train_llm_structured),
    ]

    all_results = {}
    all_preds_dict = {}
    best_model = None
    best_metrics = None
    best_preds = None
    best_probas = None
    best_f1 = -1

    nonpicklable = {"产生式规则系统", "大语言模型", "大语言模型(结构化)", "SentiWordNet"}
    kwargs = {"vectorizer": vectorizer}

    for name, train_fn in models:
        print(f"\n--- {name} ---")
        try:
            model, metrics, preds, probas = train_fn(x_train, y_train, x_test, y_test, **kwargs)
            all_results[name] = metrics
            all_preds_dict[name] = preds.tolist() if hasattr(preds, "tolist") else list(preds)
            print(f"  结果: acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}")

            if metrics["f1"] > best_f1 and name not in nonpicklable:
                best_f1 = metrics["f1"]
                best_model = model
                best_metrics = metrics
                best_preds = preds
                best_probas = probas
        except Exception as e:
            print(f"  [跳过] {e}")
            all_results[name] = {"error": str(e)}

    if best_model is None:
        ok = [n for n in all_results if "error" not in all_results[n]]
        if ok:
            best_metrics = all_results[ok[0]]
            best_preds = None
            best_probas = None
        else:
            raise RuntimeError("所有模型训练都失败了")

    return best_model, best_metrics, best_preds, best_probas, all_results, all_preds_dict


def save_b_output(model, metrics, preds, probas, all_results, all_preds):
    print("\n" + "=" * 50)
    print("B 模块: 保存产出到 B_output/")
    print("=" * 50)

    if model is not None:
        with open(config.MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        print(f"  best_model.pkl  ({type(model).__name__})")
    else:
        print("  [警告] 无可序列化模型，跳过 best_model.pkl")

    # 清理不可序列化字段（如 structured_samples 含文本的 dict）
    clean_results = {}
    for name, m in all_results.items():
        if "error" in m:
            clean_results[name] = m
        else:
            clean_results[name] = {k: v for k, v in m.items() if k != "structured_samples"}

    params_dict = {
        "model_type": type(model).__name__ if model is not None else "None",
        "metrics": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in metrics.items()},
        "all_model_results": clean_results,
    }
    with open(config.PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(params_dict, f, ensure_ascii=False, indent=2, default=str)
    print(f"  best_params.json")

    if preds is not None:
        np.save(config.PREDS_PATH, preds)
        print(f"  predictions.npy  shape={preds.shape}")

    if probas is not None:
        np.save(config.PROBAS_PATH, probas)
        print(f"  probas.npy  shape={probas.shape}")

    # 保存所有模型预测（供 SentiWordNet 分析用）
    all_preds_path = os.path.join(config.B_DIR, "all_predictions.json")
    with open(all_preds_path, "w") as f:
        json.dump(all_preds, f)
    print(f"  all_predictions.json  ({len(all_preds)} 个模型)")


def _summarize_llm_structured(all_results):
    """自动总结 LLM 结构化输出"""
    llm_key = "大语言模型(结构化)"
    if llm_key not in all_results or "error" in all_results[llm_key]:
        return
    m = all_results[llm_key]
    samples = m.get("structured_samples", [])
    if not samples:
        return

    topics = {}
    sentiments = {"正面": 0, "负面": 0, "中性": 0}
    confidences = []
    for s in samples:
        st = s.get("structured", {})
        if isinstance(st, dict):
            topic = st.get("主题", "未知")
            topics[topic] = topics.get(topic, 0) + 1
            sent = st.get("情感", "")
            if sent in sentiments:
                sentiments[sent] += 1
            conf = st.get("置信度", 0)
            if isinstance(conf, (int, float)):
                confidences.append(float(conf))

    summary = {
        "样本数": len(samples),
        "主题分布": dict(sorted(topics.items(), key=lambda x: -x[1])),
        "情感分布": sentiments,
        "平均置信度": round(np.mean(confidences), 3) if confidences else 0,
    }

    # 打印
    print(f"\n--- LLM 结构化输出总结 ---")
    print(f"  样本数: {summary['样本数']}")
    print(f"  主题分布: {summary['主题分布']}")
    print(f"  情感分布: {summary['情感分布']}")
    print(f"  平均置信度: {summary['平均置信度']}")

    # 保存
    out_dir = os.path.join(config.B_DIR, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "llm_structured_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="B 模块: 训练 + 全链路分析")
    parser.add_argument("--mock", action="store_true", help="先生成模拟 A 数据，再训练")
    args = parser.parse_args()

    if args.mock:
        print("[mock 模式] 生成模拟数据...")
        from mock_data import make_mock_data
        make_mock_data()

    # ① 加载数据
    x_train, y_train, _x_dev, _y_dev, x_test, y_test, vectorizer = load_a_output()

    # ② 训练
    best_model, best_metrics, best_preds, best_probas, all_results, all_preds = \
        train_all_models(x_train, y_train, x_test, y_test, vectorizer)

    # ③ 保存
    save_b_output(best_model, best_metrics, best_preds, best_probas, all_results, all_preds)

    # ④ 模型对比汇总
    print("\n" + "=" * 50)
    print("模型对比汇总")
    print("=" * 50)
    print(f"{'模型':<22} {'Accuracy':>9} {'F1':>9}")
    print("-" * 42)
    for name, m in all_results.items():
        if "error" in m:
            print(f"{name:<22} {'ERROR':>9} {'-':>9}")
        else:
            print(f"{name:<22} {m['accuracy']:>9.4f} {m['f1']:>9.4f}")
    print("-" * 42)
    best_name = best_metrics.get("model", type(best_model).__name__) if best_metrics else "N/A"
    print(f"最优模型: {best_name}")

    # ⑤ LLM 结构化输出总结
    _summarize_llm_structured(all_results)

    # ⑥ 数据分析
    from data_analysis import run_data_analysis
    run_data_analysis()

    # ⑦ SentiWordNet 结果分析
    from sentiwordnet_analysis import run_sentiwordnet_analysis
    run_sentiwordnet_analysis()

    # ⑧ 可视化
    print("\n" + "=" * 50)
    print("生成可视化图表")
    print("=" * 50)
    from analyze import main as run_analyze
    run_analyze()

    print("\n" + "=" * 50)
    print("全链路完成")
    print(f"  模型: B_output/best_model.pkl + best_params.json")
    print(f"  分析: B_output/analysis/data_analysis.json")
    print(f"        B_output/analysis/sentiwordnet_analysis.json")
    print(f"        B_output/analysis/llm_structured_summary.json")
    print(f"  图表: B_output/charts/")
    print("=" * 50)


if __name__ == "__main__":
    main()
