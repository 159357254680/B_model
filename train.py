"""B 模块主程序 —— 读入 A 的数据，训练 8 种模型，输出 C 需要的文件。

用法:
    python train.py              # 正常模式，读 A_output/，写 B_output/
    python train.py --mock       # 先用 mock 数据自测，然后训练

和 A、C 的接口约定见 config.py
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

    # 尝试加载 dev 集（新格式），不存在则回退到旧格式
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

    return best_model, best_metrics, best_preds, best_probas, all_results


def save_b_output(model, metrics, preds, probas, all_results):
    print("\n" + "=" * 50)
    print("B 模块: 保存产出到 B_output/")
    print("=" * 50)

    if model is not None:
        with open(config.MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        print(f"  best_model.pkl  ({type(model).__name__})")
    else:
        print("  [警告] 无可序列化模型，跳过 best_model.pkl")

    params_dict = {
        "model_type": type(model).__name__ if model is not None else "None",
        "metrics": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in metrics.items()},
        "all_model_results": all_results,
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


def main():
    parser = argparse.ArgumentParser(description="B 模块: 8 模型训练")
    parser.add_argument("--mock", action="store_true",
                        help="先生成模拟 A 数据，再训练")
    args = parser.parse_args()

    if args.mock:
        print("[mock 模式] 生成模拟数据...")
        from mock_data import make_mock_data
        make_mock_data()

    x_train, y_train, _x_dev, _y_dev, x_test, y_test, vectorizer = load_a_output()
    best_model, best_metrics, best_preds, best_probas, all_results = \
        train_all_models(x_train, y_train, x_test, y_test, vectorizer)
    save_b_output(best_model, best_metrics, best_preds, best_probas, all_results)

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
    print(f"B_output/ 已就绪，可交付给 C 模块。")


if __name__ == "__main__":
    main()
