"""B 模块可视化 —— 读取 B_output 结果，生成模型对比图"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PARAMS_PATH, B_DIR


def main():
    os.makedirs(os.path.join(B_DIR, "charts"), exist_ok=True)
    OUT = os.path.join(B_DIR, "charts")

    params = {}
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH, "r", encoding="utf-8") as f:
            params = json.load(f)

    all_results = params.get("all_model_results", {})
    if not all_results:
        print("  B_output/best_params.json 中没有模型结果，跳过可视化")
        return

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "Arial Unicode MS", "Heiti SC", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    models_order = [
        "产生式规则系统", "朴素贝叶斯", "逻辑回归",
        "特征选择+LR", "新特征+LR", "SentiWordNet",
        "大语言模型", "大语言模型(结构化)",
    ]

    # Chart 1: Model accuracy & F1 comparison
    names, accs, f1s = [], [], []
    for name in models_order:
        if name in all_results and "error" not in all_results[name]:
            names.append(name)
            accs.append(all_results[name]["accuracy"] * 100)
            f1s.append(all_results[name]["f1"] * 100)

    if names:
        x = np.arange(len(names))
        w = 0.35
        fig, ax = plt.subplots(figsize=(16, 6))
        bars1 = ax.bar(x - w/2, accs, w, label="Accuracy (%)", color="#4C72B0", edgecolor="white")
        bars2 = ax.bar(x + w/2, f1s, w, label="F1 (%)", color="#DD8452", edgecolor="white")
        for b in bars1:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, f"{b.get_height():.1f}",
                    ha="center", va="bottom", fontsize=7)
        for b in bars2:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, f"{b.get_height():.1f}",
                    ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Score (%)")
        ax.set_title("Model Performance Comparison")
        ax.legend(loc="lower right")
        ax.set_ylim(0, max(max(accs), max(f1s)) * 1.15)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "model_comparison.png"), dpi=150)
        plt.close()
        print(f"  {OUT}/model_comparison.png")

    # Chart 2: New feature importance
    feat_imp = None
    for name in ["新特征+LR"]:
        if name in all_results and "new_feature_importance" in all_results[name]:
            feat_imp = all_results[name]["new_feature_importance"]
            break
    if feat_imp:
        f_names = [x[0] for x in feat_imp]
        f_vals = [x[1] for x in feat_imp]
        colors = ["#4C72B0" if v >= 0 else "#DD8452" for v in f_vals]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(range(len(f_names)), f_vals, color=colors, edgecolor="white", height=0.6)
        ax.set_yticks(range(len(f_names)))
        ax.set_yticklabels(f_names, fontsize=10)
        ax.set_xlabel("Coefficient")
        ax.set_title("New Feature Importance (LR+NewFeatures)")
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "new_feature_importance.png"), dpi=150)
        plt.close()
        print(f"  {OUT}/new_feature_importance.png")

    # Chart 3: SentiWordNet model summary
    swn_name = "SentiWordNet"
    if swn_name in all_results and "error" not in all_results[swn_name]:
        swn = all_results[swn_name]
        labels = ["Avg Pos Score", "Avg Neg Score", "Coverage"]
        vals = [swn.get("avg_pos_score", 0), swn.get("avg_neg_score", 0), swn.get("avg_coverage", 0)]
        fig, ax = plt.subplots(figsize=(6, 4))
        colors_bar = ["#55A868", "#C44E52", "#8C8C8C"]
        ax.bar(labels, vals, color=colors_bar, edgecolor="white", width=0.5)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
        ax.set_ylabel("Score")
        ax.set_title("SentiWordNet Analysis Summary")
        ax.set_ylim(0, max(vals) * 1.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "sentiwordnet_summary.png"), dpi=150)
        plt.close()
        print(f"  {OUT}/sentiwordnet_summary.png")

    # Chart 4: LLM structured output topics
    llm_struct_name = "大语言模型(结构化)"
    if llm_struct_name in all_results and "error" not in all_results[llm_struct_name]:
        samples = all_results[llm_struct_name].get("structured_samples", [])
        if samples:
            topics = {}
            for s in samples:
                if isinstance(s.get("structured"), dict):
                    topic = s["structured"].get("主题", "未知")
                    topics[topic] = topics.get(topic, 0) + 1
            if topics:
                fig, ax = plt.subplots(figsize=(8, 5))
                sorted_topics = sorted(topics.items(), key=lambda x: -x[1])
                t_names = [t[0][:15] for t in sorted_topics]
                t_vals = [t[1] for t in sorted_topics]
                ax.bar(range(len(t_names)), t_vals, color="#8E6CD0", edgecolor="white", width=0.6)
                ax.set_xticks(range(len(t_names)))
                ax.set_xticklabels(t_names, rotation=30, ha="right", fontsize=9)
                ax.set_ylabel("Count")
                ax.set_title("LLM Structured Output: Topic Distribution")
                plt.tight_layout()
                plt.savefig(os.path.join(OUT, "llm_structured_topics.png"), dpi=150)
                plt.close()
                print(f"  {OUT}/llm_structured_topics.png")

    print("  可视化图表已保存到 B_output/charts/")


if __name__ == "__main__":
    main()
