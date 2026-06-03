"""B 模块主程序 —— 读入 A 的数据，训练模型，输出 C 需要的文件。

用法:
    python train.py              # 正常模式，读 A_output/，写 B_output/
    python train.py --mock       # 先用 mock 数据自测，然后训练

和 A、C 的接口约定见 config.py
"""

import os
import sys
import json
import pickle
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics.cluster")
from scipy import sparse

import config


def load_a_output():
    """从 A 模块加载数据。严格按照 config 里的路径读取。"""
    print("=" * 50)
    print("B 模块: 加载 A 产出的数据")
    print("=" * 50)

    files = {
        "x_train": config.X_TRAIN_PATH,
        "y_train": config.Y_TRAIN_PATH,
        "x_test":  config.X_TEST_PATH,
        "y_test":  config.Y_TEST_PATH,
        "vectorizer": config.VECTORIZER_PATH,
    }

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
    print(f"  x_test:  {x_test.shape}, y_test:  {y_test.shape}, "
          f"分布: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    return x_train, y_train, x_test, y_test, vectorizer


def train_all_models(x_train, y_train, x_test, y_test, vectorizer):
    """训练全部 4 种模型，返回最好的那个。"""
    from models.rule_based import train_rule_based
    from models.naive_bayes import train_naive_bayes
    from models.logistic_regression import train_logistic_regression
    from models.feature_selection import train_with_feature_selection

    models = [
        ("产生式规则系统", train_rule_based),
        ("朴素贝叶斯",     train_naive_bayes),
        ("逻辑回归",       train_logistic_regression),
        ("特征选择+LR",    train_with_feature_selection),
    ]

    all_results = {}
    best_model = None
    best_metrics = None
    best_preds = None
    best_probas = None
    best_f1 = -1

    kwargs = {"vectorizer": vectorizer}

    for name, train_fn in models:
        print(f"\n--- {name} ---")
        try:
            model, metrics, preds, probas = train_fn(x_train, y_train, x_test, y_test, **kwargs)
            all_results[name] = metrics
            print(f"  结果: acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}")

            # 规则系统的 model 不能 pickle，单独处理
            if metrics["f1"] > best_f1 and name != "产生式规则系统":
                best_f1 = metrics["f1"]
                best_model = model
                best_metrics = metrics
                best_preds = preds
                best_probas = probas
        except Exception as e:
            print(f"  [跳过] {e}")
            all_results[name] = {"error": str(e)}

    # 如果全部失败，无法继续
    if best_model is None:
        # 如果只有规则系统成功了
        for name, train_fn in models:
            if name in all_results and "error" not in all_results[name]:
                best_metrics = all_results[name]
                break
        if best_metrics is None:
            raise RuntimeError("所有模型训练都失败了")

    return best_model, best_metrics, best_preds, best_probas, all_results


def save_b_output(model, metrics, preds, probas, all_results):
    """保存 B 模块的产出到 B_output/，供 C 模块使用。"""
    print("\n" + "=" * 50)
    print("B 模块: 保存产出到 B_output/")
    print("=" * 50)

    # 1. 最优模型
    with open(config.MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"  best_model.pkl  ({type(model).__name__})")

    # 2. 参数和指标
    params_dict = {
        "model_type": type(model).__name__,
        "metrics": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in metrics.items()},
        "all_model_results": all_results,
    }
    with open(config.PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(params_dict, f, ensure_ascii=False, indent=2, default=str)
    print(f"  best_params.json")

    # 3. 预测结果
    if preds is not None:
        np.save(config.PREDS_PATH, preds)
        print(f"  predictions.npy  shape={preds.shape}")

    # 4. 预测概率（可选，方便 C 画 ROC）
    if probas is not None:
        np.save(config.PROBAS_PATH, probas)
        print(f"  probas.npy  shape={probas.shape}")


def main():
    parser = argparse.ArgumentParser(description="B 模块: 模型训练")
    parser.add_argument("--mock", action="store_true",
                        help="先生成模拟 A 数据，再训练")
    args = parser.parse_args()

    if args.mock:
        print("[mock 模式] 生成模拟数据...")
        from mock_data import make_mock_data
        make_mock_data()

    # 1. 加载 A 数据
    x_train, y_train, x_test, y_test, vectorizer = load_a_output()

    # 2. 训练全部模型，选最好
    best_model, best_metrics, best_preds, best_probas, all_results = \
        train_all_models(x_train, y_train, x_test, y_test, vectorizer)

    # 3. 保存产出
    save_b_output(best_model, best_metrics, best_preds, best_probas, all_results)

    # 4. 总结
    print("\n" + "=" * 50)
    print("模型对比汇总")
    print("=" * 50)
    print(f"{'模型':<20} {'Accuracy':>9} {'F1':>9}")
    print("-" * 40)
    for name, m in all_results.items():
        if "error" in m:
            print(f"{name:<20} {'ERROR':>9} {'-':>9}")
        else:
            print(f"{name:<20} {m['accuracy']:>9.4f} {m['f1']:>9.4f}")
    print("-" * 40)
    best_name = best_metrics.get("model", type(best_model).__name__) if best_metrics else "N/A"
    print(f"最优模型: {best_name}")
    print(f"B_output/ 已就绪，可交付给 C 模块。")


if __name__ == "__main__":
    main()
