"""产生式规则系统 —— VADER + SentiWordNet + WordNet + 自定义规则"""
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def _ensure_nltk_resources():
    import nltk
    nltk_data = os.path.expanduser("~/nltk_data")
    if os.path.exists(nltk_data):
        nltk.data.path.insert(0, nltk_data)
    for res in ["vader_lexicon", "sentiwordnet", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{res}") if res != "vader_lexicon" else nltk.data.find(f"sentiment/{res}")
        except LookupError:
            nltk.download(res, download_dir=nltk_data, quiet=True)


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
        _ensure_nltk_resources()
        from nltk.sentiment import SentimentIntensityAnalyzer
        self.vader = SentimentIntensityAnalyzer()

    def _sentiwordnet_score(self, text):
        from nltk.corpus import sentiwordnet as swn
        words = text.lower().split()
        pos_total, neg_total = 0.0, 0.0
        count = 0
        for w in words:
            synsets = list(swn.senti_synsets(w))
            if synsets:
                pos_total += sum(s.pos_score() for s in synsets) / len(synsets)
                neg_total += sum(s.neg_score() for s in synsets) / len(synsets)
                count += 1
        if count == 0:
            return 0.0
        return (pos_total - neg_total) / count

    def predict(self, texts):
        preds = []
        for text in texts:
            vader_score = self.vader.polarity_scores(text)["compound"]
            words = set(text.lower().split())
            pos_hits = len(words & self.POS)
            neg_hits = len(words & self.NEG)
            lexicon_score = (pos_hits - neg_hits) / max(pos_hits + neg_hits, 1)
            swn_score = self._sentiwordnet_score(text)
            final = 0.4 * vader_score + 0.3 * lexicon_score + 0.3 * swn_score
            preds.append(1 if final >= 0 else 0)
        return np.array(preds)


def train_rule_based(x_train, y_train, x_test, y_test, vectorizer=None):
    if vectorizer is not None:
        texts = vectorizer.inverse_transform(x_test)
        texts = [" ".join(words) for words in texts]
    else:
        raise ValueError("规则系统需要 vectorizer.pkl 来还原文本")

    model = RuleBasedClassifier()
    preds = model.predict(texts)
    probas = np.column_stack([1 - preds, preds]).astype(float)

    metrics = {
        "model": "RuleBased(VADER+SentiWordNet+Lexicon)",
        "accuracy":  accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall":    recall_score(y_test, preds, zero_division=0),
        "f1":        f1_score(y_test, preds, zero_division=0),
    }
    return model, metrics, preds, probas
