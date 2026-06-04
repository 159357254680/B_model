"""B 模块主程序 —— 读入 A 的数据，训练逻辑回归模型，输出 C 需要的文件。

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
from models.logistic_regression import train_logistic_regression


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


def save_b_output(model, metrics, preds, probas):
    """保存 B 模块的产出到 B_output/，供 C 模块使用。"""
    print("\n" + "=" * 50)
    print("B 模块: 保存产出到 B_output/")
    print("=" * 50)

    with open(config.MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"  best_model.pkl  ({type(model).__name__})")

    params_dict = {
        "model_type": type(model).__name__,
        "metrics": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in metrics.items()},
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
    parser = argparse.ArgumentParser(description="B 模块: 逻辑回归模型训练")
    parser.add_argument("--mock", action="store_true",
                        help="先生成模拟 A 数据，再训练")
    args = parser.parse_args()

    if args.mock:
        print("[mock 模式] 生成模拟数据...")
        from mock_data import make_mock_data
        make_mock_data()

    x_train, y_train, x_test, y_test, vectorizer = load_a_output()

    print(f"\n--- 逻辑回归 ---")
    model, metrics, preds, probas = train_logistic_regression(
        x_train, y_train, x_test, y_test, vectorizer=vectorizer
    )
    print(f"  结果: acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}")

    save_b_output(model, metrics, preds, probas)

    print("\n" + "=" * 50)
    print(f"B_output/ 已就绪，可交付给 C 模块。")


if __name__ == "__main__":
    main()
