"""大语言模型分类器 —— 调用 API 做情感分类（支持 DeepSeek / OpenAI 兼容接口）"""
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from openai import OpenAI


def train_llm(x_train, y_train, x_test, y_test, vectorizer=None, **_kw):
    print("  大语言模型: 调用 API 做情感分类 ...")

    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
    model = os.environ.get("LLM_MODEL", "deepseek-chat")

    if not api_key:
        raise RuntimeError("请设置 LLM_API_KEY 环境变量")

    if vectorizer is None:
        raise ValueError("LLM 需要 vectorizer 还原文本")

    texts = vectorizer.inverse_transform(x_test)
    texts = [" ".join(words) for words in texts]

    n = min(200, len(texts))
    idx = np.random.RandomState(42).choice(len(texts), n, replace=False)
    sampled_texts = [texts[i][:300] for i in idx]
    sampled_labels = y_test[idx]

    client = OpenAI(api_key=api_key, base_url=base_url)
    preds = []
    for i, text in enumerate(sampled_texts):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "判断评论情感：1正面，0负面。只回复数字。"},
                    {"role": "user", "content": text},
                ],
                temperature=0,
                max_tokens=2,
            )
            label = resp.choices[0].message.content.strip()
            preds.append(1 if "1" in label else 0)
        except Exception:
            preds.append(0)
        if (i + 1) % 50 == 0:
            print(f"  LLM 推理: {i+1}/{n}")

    preds = np.array(preds)
    probas = np.column_stack([1 - preds, preds]).astype(float)
    metrics = {
        "model": f"LLM({model})",
        "sample_size": n,
        "accuracy": accuracy_score(sampled_labels, preds),
        "precision": precision_score(sampled_labels, preds, zero_division=0),
        "recall": recall_score(sampled_labels, preds, zero_division=0),
        "f1": f1_score(sampled_labels, preds, zero_division=0),
    }
    return None, metrics, preds, probas
