"""逻辑回归分类器"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_logistic_regression(x_train, y_train, x_test, y_test, vectorizer=None, **_kw):
    print("  逻辑回归: 网格搜索 C...")
    lr = LogisticRegression(max_iter=2000, random_state=42)
    grid = GridSearchCV(
        lr, {"C": [0.1, 0.5, 1.0, 3.0, 5.0, 10.0]},
        cv=3, scoring="f1", n_jobs=1
    )
    grid.fit(x_train, y_train)
    print(f"    best_C={grid.best_params_['C']:.2f}, best_cv_f1={grid.best_score_:.4f}")

    best_model = grid.best_estimator_
    preds = best_model.predict(x_test)
    probas = best_model.predict_proba(x_test)

    # 提取 Top 特征（供 C 画图）
    top_features = {}
    if vectorizer is not None and hasattr(best_model, "coef_"):
        names = vectorizer.get_feature_names_out()
        coef = best_model.coef_[0]
        top_idx = np.argsort(coef)
        top_features["positive"] = [(names[i], float(coef[i])) for i in top_idx[-10:][::-1]]
        top_features["negative"] = [(names[i], float(coef[i])) for i in top_idx[:10]]

    metrics = {
        "accuracy":  accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall":    recall_score(y_test, preds, zero_division=0),
        "f1":        f1_score(y_test, preds, zero_division=0),
        "best_C":    grid.best_params_["C"],
        "top_features": top_features,
    }
    return best_model, metrics, preds, probas
