"""逻辑回归 + 新特征工程 —— 在 TF-IDF 基础上设计至少两个新特征"""
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


POS_WORDS = {
    "excellent", "amazing", "wonderful", "great", "fantastic", "brilliant",
    "outstanding", "superb", "perfect", "loved", "enjoyed", "beautiful",
    "masterpiece", "incredible", "hilarious", "touching", "engaging",
    "remarkable", "exceptional", "delightful", "impressive", "riveting",
    "best", "must-see", "favorite", "gem", "classic", "refreshing",
    "clever", "intelligent", "moving", "powerful", "unforgettable",
    "magnificent", "captivating", "entertaining", "compelling",
    "flawless", "stunning", "thrilling", "heartwarming", "worth", "recommend",
}
NEG_WORDS = {
    "terrible", "awful", "horrible", "boring", "worst", "waste",
    "disappointing", "dull", "poor", "bad", "stupid", "ridiculous",
    "dreadful", "lame", "cheesy", "unwatchable", "crap", "garbage",
    "trash", "pathetic", "mediocre", "unfunny", "pointless", "annoying",
    "cliche", "predictable", "forgettable", "atrocious", "overrated",
    "disappointment", "mess", "torture", "cringe", "nonsense", "absurd",
}


def _texts_from_sparse(x, vectorizer):
    texts = vectorizer.inverse_transform(x)
    return [" ".join(words) for words in texts]


def _extract_features(texts):
    feats = np.zeros((len(texts), 8))
    for i, t in enumerate(texts):
        words = t.split()
        n_words = len(words)
        if n_words == 0:
            continue
        n_chars = len(t)
        pos_cnt = sum(1 for w in words if w.lower() in POS_WORDS)
        neg_cnt = sum(1 for w in words if w.lower() in NEG_WORDS)
        excl = t.count("!")
        ques = t.count("?")
        avg_wlen = n_chars / n_words
        upper_ratio = sum(1 for w in words if w.isupper() and len(w) > 1) / n_words
        feats[i] = [
            np.log1p(n_words),
            np.log1p(n_chars),
            pos_cnt / n_words,
            neg_cnt / n_words,
            excl / n_words,
            ques / n_words,
            avg_wlen,
            upper_ratio,
        ]
    return feats


def train_logistic_regression_features(x_train, y_train, x_test, y_test, vectorizer=None, **_kw):
    print("  逻辑回归+新特征: TF-IDF + 手工特征 ...")
    if vectorizer is None:
        raise ValueError("新特征工程需要 vectorizer 还原文本")

    train_texts = _texts_from_sparse(x_train, vectorizer)
    test_texts = _texts_from_sparse(x_test, vectorizer)

    print("    提取手工特征: word_count, char_count, pos_ratio, neg_ratio, "
          "excl_ratio, ques_ratio, avg_word_len, upper_ratio")
    train_feats = _extract_features(train_texts)
    test_feats = _extract_features(test_texts)

    scaler = StandardScaler()
    train_feats_scaled = scaler.fit_transform(train_feats)
    test_feats_scaled = scaler.transform(test_feats)

    train_feats_sparse = sparse.csr_matrix(train_feats_scaled)
    test_feats_sparse = sparse.csr_matrix(test_feats_scaled)

    x_train_combined = sparse.hstack([x_train, train_feats_sparse], format="csr")
    x_test_combined = sparse.hstack([x_test, test_feats_sparse], format="csr")
    print(f"    组合特征维度: {x_train_combined.shape[1]} (TF-IDF + 8 新特征)")

    lr = LogisticRegression(max_iter=2000, random_state=42)
    grid = GridSearchCV(
        lr, {"C": [0.1, 0.5, 1.0, 3.0, 5.0, 10.0]},
        cv=3, scoring="f1", n_jobs=-1,
    )
    grid.fit(x_train_combined, y_train)
    print(f"    best_C={grid.best_params_['C']:.2f}, best_cv_f1={grid.best_score_:.4f}")

    best_model = grid.best_estimator_
    preds = best_model.predict(x_test_combined)
    probas = best_model.predict_proba(x_test_combined)

    feature_names = [
        "log_word_count", "log_char_count", "pos_word_ratio", "neg_word_ratio",
        "excl_ratio", "ques_ratio", "avg_word_len", "upper_ratio",
    ]
    new_feat_coef = best_model.coef_[0][-8:]
    top_new = sorted(zip(feature_names, new_feat_coef), key=lambda x: -abs(x[1]))

    metrics = {
        "model": "LR+NewFeatures",
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "best_C": grid.best_params_["C"],
        "new_features": 8,
        "new_feature_importance": [(n, round(float(c), 4)) for n, c in top_new],
    }
    return best_model, metrics, preds, probas
