"""产生式规则系统 —— VADER 情感词典 + 自定义规则"""
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class RuleBasedClassifier:
    POS = {"excellent", "amazing", "wonderful", "great", "fantastic", "brilliant",
           "outstanding", "superb", "perfect", "loved", "enjoyed", "beautiful",
           "masterpiece", "incredible", "hilarious", "touching", "engaging",
           "remarkable", "exceptional", "delightful", "impressive", "riveting",
           "best", "must-see", "favorite", "gem", "classic", "refreshing",
           "clever", "intelligent", "moving", "powerful", "unforgettable",
           "magnificent", "captivating", "entertaining", "compelling",
           "flawless", "stunning", "thrilling", "heartwarming", "worth", "recommend"}
    NEG = {"terrible", "awful", "horrible", "boring", "worst", "waste",
           "disappointing", "dull", "poor", "bad", "stupid", "ridiculous",
           "dreadful", "lame", "cheesy", "unwatchable", "crap", "garbage",
           "trash", "pathetic", "mediocre", "unfunny", "pointless", "annoying",
           "cliche", "predictable", "forgettable", "atrocious", "overrated",
           "disappointment", "mess", "torture", "cringe", "nonsense", "absurd"}

    def __init__(self):
        import nltk
        import os
        nltk_data = os.path.expanduser("~/nltk_data")
        if os.path.exists(nltk_data):
            nltk.data.path.insert(0, nltk_data)
        try:
            self.vader = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", download_dir=nltk_data, quiet=True)
            self.vader = SentimentIntensityAnalyzer()

    def predict(self, texts):
        preds = []
        for text in texts:
            vader_score = self.vader.polarity_scores(text)["compound"]
            words = set(text.lower().split())
            pos_hits = len(words & self.POS)
            neg_hits = len(words & self.NEG)
            lexicon_score = (pos_hits - neg_hits) / max(pos_hits + neg_hits, 1)
            final = 0.5 * vader_score + 0.5 * lexicon_score
            preds.append(1 if final >= 0 else 0)
        return np.array(preds)


def train_rule_based(x_train, y_train, x_test, y_test, vectorizer=None):
    """规则系统不需要训练，直接用 VADER + 关键词做推理。
    但需要 vectorizer 来把稀疏矩阵还原成文本。
    """
    # 从 TF-IDF 稀疏矩阵重建文本
    if vectorizer is not None:
        texts = vectorizer.inverse_transform(x_test)
        texts = [" ".join(words) for words in texts]
    else:
        raise ValueError("规则系统需要 vectorizer.pkl 来还原文本")

    model = RuleBasedClassifier()
    preds = model.predict(texts)
    probas = np.column_stack([1 - preds, preds]).astype(float)

    metrics = {
        "accuracy":  accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall":    recall_score(y_test, preds, zero_division=0),
        "f1":        f1_score(y_test, preds, zero_division=0),
    }
    return model, metrics, preds, probas
