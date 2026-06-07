"""B 模块配置 —— 与 A、C 的接口约定"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# A 的产出 = B 的输入（契约路径，不要改）
A_DIR = os.path.join(BASE_DIR, "A_output")
X_TRAIN_PATH = os.path.join(A_DIR, "x_train.npz")
Y_TRAIN_PATH = os.path.join(A_DIR, "y_train.npy")
X_DEV_PATH   = os.path.join(A_DIR, "x_dev.npz")
Y_DEV_PATH   = os.path.join(A_DIR, "y_dev.npy")
X_TEST_PATH  = os.path.join(A_DIR, "x_test.npz")
Y_TEST_PATH  = os.path.join(A_DIR, "y_test.npy")
VECTORIZER_PATH = os.path.join(A_DIR, "vectorizer.pkl")
RAW_TRAIN_PATH = os.path.join(A_DIR, "raw_train.npy")
RAW_DEV_PATH   = os.path.join(A_DIR, "raw_dev.npy")
RAW_TEST_PATH  = os.path.join(A_DIR, "raw_test.npy")
RAW_LABELS_TRAIN_PATH = os.path.join(A_DIR, "raw_labels_train.npy")
RAW_LABELS_DEV_PATH   = os.path.join(A_DIR, "raw_labels_dev.npy")
RAW_LABELS_TEST_PATH  = os.path.join(A_DIR, "raw_labels_test.npy")

# B 的产出 = C 的输入（契约路径，不要改）
B_DIR = os.path.join(BASE_DIR, "B_output")
MODEL_PATH   = os.path.join(B_DIR, "best_model.pkl")
PARAMS_PATH  = os.path.join(B_DIR, "best_params.json")
PREDS_PATH   = os.path.join(B_DIR, "predictions.npy")
PROBAS_PATH  = os.path.join(B_DIR, "probas.npy")
ALL_PREDS_PATH = os.path.join(B_DIR, "all_predictions.json")

os.makedirs(B_DIR, exist_ok=True)
os.makedirs(A_DIR, exist_ok=True)
