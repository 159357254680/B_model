# B 模块：模型训练

## 概述

B 模块负责从 A 模块接收预处理后的数据，训练并对比 8 种分类模型，选出可序列化的最优模型交付给 C 模块。

**核心定位**：A 数据预处理 → B 训练 + 分析 → C 评估。含数据分析和 SentiWordNet 结果分析。

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

### 2. 数据准备

**真实数据（推荐）**：

```bash
python a_preprocess.py
```

自动下载 IMDB 数据集（~80MB, 5万条），清洗、70/15/15 三分、TF-IDF 向量化，输出 14 个文件到 `A_output/`。

**模拟数据（快速自测）**：

```bash
python train.py --mock
```

### 3. 数据分析

```bash
python data_analysis.py
```

输出 `B_output/analysis/data_analysis.json`，包含：
- 训练/开发/测试集分布
- 正负向情感 Top10 高频词
- PMI（点互信息）Top10 词
- 词性分布（NN, JJ, RB 等）

### 4. 训练

```bash
python train.py
```

### 5. 可视化

```bash
python analyze.py
```

输出 `B_output/charts/`：
- `model_comparison.png` — 所有模型 Accuracy/F1 对比柱状图
- `new_feature_importance.png` — 新特征系数重要性
- `sentiwordnet_summary.png` — SentiWordNet 分析摘要
- `llm_structured_topics.png` — LLM 结构化输出主题分布（需 API key）

### 6. SentiWordNet 结果分析

```bash
python sentiwordnet_analysis.py
```

用 SentiWordNet 分析所有模型（除产生式系统外）在测试集上的输出，计算正向词数量与预测正向的相关度。输出 `B_output/analysis/sentiwordnet_analysis.json`。

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
a_preprocess.py     → 下载 IMDB → 清洗 → 70/15/15 三分 → TF-IDF → A_output/
data_analysis.py    → 分布/Top词/PMI/词性 → B_output/analysis/
train.py            → 8 模型训练 → 选最优 → B_output/
analyze.py          → 4 张对比图 → B_output/charts/
sentiwordnet_analysis.py → SentiWordNet 交叉分析 → B_output/analysis/

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
