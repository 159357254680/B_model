"""朴素贝叶斯分类器"""
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_naive_bayes(x_train, y_train, x_test, y_test, **_kw):
    print("  朴素贝叶斯: 网格搜索 alpha...")
    candidates = [
        ("MultinomialNB", MultinomialNB()),
        ("ComplementNB", ComplementNB()),
    ]

    best_model = None
    best_f1 = 0
    best_name = ""

    for name, model in candidates:
        grid = GridSearchCV(
            model, {"alpha": [0.05, 0.1, 0.5, 1.0, 2.0]},
            cv=3, scoring="f1", n_jobs=-1
        )
        grid.fit(x_train, y_train)
        preds = grid.predict(x_test)
        f1 = f1_score(y_test, preds, zero_division=0)
        print(f"    {name}: best_alpha={grid.best_params_['alpha']}, test_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_model = grid.best_estimator_
            best_name = name

    preds = best_model.predict(x_test)
    probas = best_model.predict_proba(x_test)
    metrics = {
        "model": best_name,
        "accuracy":  accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall":    recall_score(y_test, preds, zero_division=0),
        "f1":        f1_score(y_test, preds, zero_division=0),
        "best_alpha": best_model.alpha,
    }
    return best_model, metrics, preds, probas
