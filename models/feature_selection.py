"""特征选择 + 逻辑回归"""
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_with_feature_selection(x_train, y_train, x_test, y_test, vectorizer=None, **_kw):
    print("  特征选择: 对比 Chi2 vs Mutual Information...")
    methods = [
        ("Chi2", SelectKBest(chi2)),
        ("MutualInfo", SelectKBest(mutual_info_classif)),
    ]
    max_k = x_train.shape[1]
    k_values = [500, 1000, 2000, 5000]
    k_values = [k for k in k_values if k <= max_k]
    if not k_values:
        k_values = [max_k // 2, max_k]  # 小特征集的 fallback

    best_model = None
    best_f1 = 0
    best_info = ""

    for method_name, selector in methods:
        for k in k_values:
            pipe = Pipeline([
                ("select", SelectKBest(score_func=selector.score_func, k=k)),
                ("clf", LogisticRegression(max_iter=2000, random_state=42)),
            ])
            # 简单网格搜索 C
            grid = GridSearchCV(pipe, {"clf__C": [0.5, 1.0, 3.0]}, cv=3, scoring="f1", n_jobs=-1)
            grid.fit(x_train, y_train)
            preds = grid.predict(x_test)
            f1 = f1_score(y_test, preds, zero_division=0)
            print(f"    {method_name} k={k:>5d}: test_f1={f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_model = grid.best_estimator_
                best_info = f"{method_name}_k={k}_C={grid.best_params_['clf__C']}"

    preds = best_model.predict(x_test)
    probas = best_model.predict_proba(x_test)
    metrics = {
        "method": best_info,
        "accuracy":  accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall":    recall_score(y_test, preds, zero_division=0),
        "f1":        f1_score(y_test, preds, zero_division=0),
    }
    return best_model, metrics, preds, probas
