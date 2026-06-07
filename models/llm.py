"""大语言模型分类器 —— 调用 API 做情感分类（支持 DeepSeek / OpenAI 兼容接口）
包含两种模式：
  1. 简单 prompt（200条）：0/1 分类
  2. 结构化输出（20条）：JSON 模式 + 框架表示法
"""
import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def _get_client():
    api_key = os.getenv("LLM_API_KEY", "")
    base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("LLM_MODEL", "deepseek-chat")
    if not api_key:
        raise RuntimeError("请设置 LLM_API_KEY 环境变量")
    return OpenAI(api_key=api_key, base_url=base_url), model


def train_llm(x_train, y_train, x_test, y_test, vectorizer=None, **_kw):
    """简单 prompt：200 条，0/1 分类"""
    print("  大语言模型: 调用 API 做情感分类 ...")

    api_key = os.getenv("LLM_API_KEY", "")
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

    client, model = _get_client()
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


def train_llm_structured(x_train, y_train, x_test, y_test, vectorizer=None, **_kw):
    """结构化输出：20 条，JSON 模式 + 框架表示法（主题、情感、置信度）"""
    print("  大语言模型(结构化输出): JSON 模式 + 框架表示法 ...")

    api_key = os.getenv("LLM_API_KEY", "")
    if not api_key:
        raise RuntimeError("请设置 LLM_API_KEY 环境变量")

    if vectorizer is None:
        raise ValueError("LLM 需要 vectorizer 还原文本")

    texts = vectorizer.inverse_transform(x_test)
    texts = [" ".join(words) for words in texts]

    n = min(20, len(texts))
    idx = np.random.RandomState(42).choice(len(texts), n, replace=False)
    sampled_texts = [texts[i][:400] for i in idx]
    sampled_labels = y_test[idx]

    system_prompt = (
        "你是一个电影评论分析助手。对于每条评论，用 JSON 框架表示法输出以下信息：\n"
        '{"主题": "电影的具体方面(如演技、剧情、特效、导演、配乐等)", '
        '"情感": "正面/负面/中性", '
        '"置信度": 0.0-1.0之间的小数, '
        '"关键词": ["关键词1", "关键词2"], '
        '"简要理由": "一句话说明判断依据"}'
    )

    client, model = _get_client()
    results = []
    preds = []
    for i, text in enumerate(sampled_texts):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content.strip()
            parsed = json.loads(content)
            sentiment = parsed.get("情感", "")
            if "正面" in str(sentiment):
                preds.append(1)
            elif "负面" in str(sentiment):
                preds.append(0)
            else:
                preds.append(1 if parsed.get("置信度", 0) >= 0.5 else 0)
            results.append({"text": text[:200], "label": int(sampled_labels[i]), "prediction": preds[-1], "structured": parsed})
        except Exception:
            preds.append(0)
            results.append({"text": text[:200], "label": int(sampled_labels[i]), "prediction": 0, "structured": {"error": "API异常"}})
        if (i + 1) % 10 == 0:
            print(f"  LLM 结构化推理: {i+1}/{n}")

    preds = np.array(preds)
    probas = np.column_stack([1 - preds, preds]).astype(float)
    metrics = {
        "model": f"LLM-Structured({model})",
        "sample_size": n,
        "accuracy": accuracy_score(sampled_labels, preds),
        "precision": precision_score(sampled_labels, preds, zero_division=0),
        "recall": recall_score(sampled_labels, preds, zero_division=0),
        "f1": f1_score(sampled_labels, preds, zero_division=0),
        "structured_samples": results,
    }
    return None, metrics, preds, probas
