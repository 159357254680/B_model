# B 模块：模型训练

## 概述

B 模块负责从 A 模块接收预处理后的数据，训练并对比 8 种分类模型，选出可序列化的最优模型交付给 C 模块。

**核心定位**：A 数据预处理 → B 训练 + 全链路自动分析。`python train.py` 一条命令完成所有步骤。

## 项目结构

```
B_model/
├── config.py                       # 路径和接口约定
├── a_preprocess.py                 # A 模块：IMDB 真实数据预处理（70/15/15 三分）
├── train.py                        # B 模块：8 模型训练
├── data_analysis.py                # 数据分析：分布、Top10词频、PMI、词性
├── analyze.py                      # 可视化：模型对比、特征重要性、主题分布
├── sentiwordnet_analysis.py        # SentiWordNet 结果分析
├── mock_data.py                    # 模拟数据（快速自测）
├── requirements.txt                # Python 依赖
├── .env.example                    # LLM 配置模板
├── models/                         # 8 种模型实现
│   ├── rule_based.py                # 产生式规则系统（VADER + SentiWordNet + 关键词）
│   ├── naive_bayes.py               # 朴素贝叶斯（MultinomialNB / ComplementNB）
│   ├── logistic_regression.py       # 逻辑回归（GridSearchCV 选 C）
│   ├── feature_selection.py         # 特征选择 + LR（Chi2 / 互信息）
│   ├── logistic_regression_features.py  # 新特征工程 + LR（8个手工特征）
│   ├── sentiwordnet.py              # SentiWordNet 情感分类器
│   └── llm.py                       # 大语言模型（简单 + 结构化输出两种模式）
├── A_output/                       # A 模块输出（B 读取）
├── B_output/                       # B 模块输出（C 读取）
│   ├── charts/                     # 可视化图表
│   └── analysis/                   # 数据分析结果
└── data/                           # IMDB 原始数据缓存
```

## 数据流

```
 A 模块                             B 模块                             C 模块
────────                          ──────────                         ─────────
x_train.npz ──┐
x_dev.npz   ──┤
x_test.npz  ──┼──→ train.py ──────→ best_model.pkl    ──→ 加载模型评估
y_*.npy     ──┤                    best_params.json        计算指标
vectorizer.pkl ┘                    predictions.npy        画图制表
raw_*.npy  ───→ data_analysis.py   probas.npy             错误分析
               sentiwordnet_analysis.py
```

## 使用方式

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

依赖项：numpy, scipy, scikit-learn, nltk, openai, python-dotenv, matplotlib

### 2. 运行全链路（一条命令）

```bash
# 真实数据
python a_preprocess.py   # 先下载 IMDB 数据
python train.py          # 自动完成：训练 + 数据分析 + SentiWordNet分析 + 可视化

# 模拟数据
python train.py --mock   # 一步完成（含自动分析）
```

`train.py` 自动串行执行：加载数据 → 训练 8 模型 → 保存输出 → LLM结构化总结 → 数据分析 → SentiWordNet结果分析 → 可视化图表。

### 3. 输出文件

| 产物 | 路径 | 说明 |
|------|------|------|
| 最优模型 | `B_output/best_model.pkl` | F1 最高可序列化模型 |
| 全部指标 | `B_output/best_params.json` | 所有模型指标对比 |
| 预测结果 | `B_output/predictions.npy` + `probas.npy` | 测试集预测 |
| 数据分析 | `B_output/analysis/data_analysis.json` | 分布/Top词/PMI/词性 |
| SentiWordNet分析 | `B_output/analysis/sentiwordnet_analysis.json` | 每个模型的正向词密度对比 |
| LLM总结 | `B_output/analysis/llm_structured_summary.json` | 结构化输出主题/情感/置信度 |
| 对比图 | `B_output/charts/model_comparison.png` | 模型 Accuracy/F1 柱状图 |
| 特征图 | `B_output/charts/new_feature_importance.png` | 新特征系数 |
| SentiWordNet图 | `B_output/charts/sentiwordnet_summary.png` | SWN 摘要 |
| LLM主题图 | `B_output/charts/llm_structured_topics.png` | 结构化输出主题分布 |

## 接口契约

### A → B（A 必须产出的文件）

| 文件 | 格式 | 内容 |
|------|------|------|
| `x_train.npz` | scipy 稀疏矩阵 | 训练集 TF-IDF 特征 |
| `y_train.npy` | numpy 数组 | 训练集标签 (0/1) |
| `x_dev.npz` | scipy 稀疏矩阵 | 开发集 TF-IDF 特征 |
| `y_dev.npy` | numpy 数组 | 开发集标签 (0/1) |
| `x_test.npz` | scipy 稀疏矩阵 | 测试集 TF-IDF 特征 |
| `y_test.npy` | numpy 数组 | 测试集标签 (0/1) |
| `vectorizer.pkl` | pickle 对象 | TfidfVectorizer 实例 |
| `raw_*.npy` | numpy 数组 | 原始文本（供分析用）|
| `raw_labels_*.npy` | numpy 数组 | 原始标签（供分析用）|

### B → C（B 会产出的文件）

| 文件 | 格式 | 内容 |
|------|------|------|
| `best_model.pkl` | pickle 对象 | F1 最高的可序列化模型 |
| `best_params.json` | JSON | 最优模型参数 + 全部模型指标对比 + LLM 结构化样本 |
| `predictions.npy` | numpy 数组 | 测试集预测标签 |
| `probas.npy` | numpy 数组 | 测试集预测概率 |
| `charts/` | PNG | 4 张可视化图表 |
| `analysis/` | JSON | 数据分析 + SentiWordNet 分析 |

## 8 种模型

| 模型 | 需要训练 | 可 pickle | 说明 |
|------|---------|----------|------|
| 产生式规则系统 | 否 | 否 | VADER + SentiWordNet + 自定义关键词，三路加权打分 |
| 朴素贝叶斯 | 是 | 是 | MultinomialNB / ComplementNB，网格搜索 alpha |
| 逻辑回归 | 是 | 是 | L2 正则，网格搜索 C，可提取特征权重 |
| 特征选择+LR | 是 | 是 | Chi2 / 互信息 + SelectKBest + LR Pipeline |
| 新特征+LR | 是 | 是 | TF-IDF + 8 个手工特征（词数、情感词密度、标点等）|
| SentiWordNet | 否 | 否 | WordNet 情感词典打分，计算 pos/neg/obj 分数 |
| 大语言模型 | 否 | 否 | 简单 prompt：200条 0/1 分类 |
| 大语言模型(结构化) | 否 | 否 | JSON 模式 + 框架表示法（主题/情感/置信度/关键词/理由），20条 |

## 大语言模型配置

复制 `.env.example` 为 `.env` 并填入 API Key：

```bash
cp .env.example .env
# 编辑 .env 填入 LLM_API_KEY
```

支持 OpenAI 兼容厂商（DeepSeek、智谱、通义千问、Moonshot 等），改 `LLM_BASE_URL` 和 `LLM_MODEL` 即可。不设 key 时两个 LLM 模型自动跳过。

### 结构化输出

`大语言模型(结构化)` 使用 JSON 模式 + 框架表示法，输出格式：

```json
{
  "主题": "演技",
  "情感": "正面",
  "置信度": 0.85,
  "关键词": ["acting", "brilliant"],
  "简要理由": "演员表演出色，情感表达真实"
}
```

## 执行流程

```
a_preprocess.py → 下载 IMDB → 清洗 → 70/15/15 三分 → TF-IDF → A_output/

train.py (全链路):
  ├── ① 加载 A_output/ 数据
  ├── ② 训练 8 个模型，保存所有预测
  ├── ③ 模型对比汇总表
  ├── ④ LLM 结构化输出总结（需 API key）
  ├── ⑤ 数据分析（分布 / Top10词 / PMI / 词性）
  ├── ⑥ SentiWordNet 结果分析（按模型预测分组）
  └── ⑦ 可视化图表（4 张 PNG）

模型                         Accuracy        F1
----------------------------------------------
产生式规则系统                  0.6933    0.7473
朴素贝叶斯                    0.8589    0.8602
逻辑回归                      0.8917    0.8925  ← 最优
特征选择+LR                   0.8917    0.8925
新特征+LR                    0.8879    0.8882
SentiWordNet                0.6125    0.7096
大语言模型                     (需API)    (需API)
大语言模型(结构化)               (需API)    (需API)
----------------------------------------------
最优模型: LogisticRegression
```

## 常见问题

**Q: mock 数据为什么准确率 100%？**
mock 数据用固定的正/负面词汇生成，信号太强。真实 IMDB 数据参考范围见上方。

**Q: 和 A、C 的路径不一样怎么办？**
修改 `config.py` 中的路径常量。

**Q: LLM 怎么换厂商？**
改 `.env` 中的 `LLM_BASE_URL` 和 `LLM_MODEL` 即可，接口都是 OpenAI 兼容的。

**Q: 为什么 train/dev/test 是 70/15/15？**
实验要求训练集、开发集、测试集三分。70% 训练、15% 开发（模型选择）、15% 测试（最终评估）。
