# B 模块：模型训练

## 概述

B 模块负责从 A 模块接收预处理后的数据，训练并对比 7 种分类模型，选出可序列化的最优模型交付给 C 模块。

**核心定位**：读入 → 训练 → 输出 + 可视化分析。

## 项目结构

```
B_model/
├── config.py               # 路径和接口约定
├── train.py                # 主程序入口
├── analyze.py              # 可视化分析（模型对比图）
├── mock_data.py            # 模拟 A 模块数据（独立测试用）
├── requirements.txt        # Python 依赖
├── models/                 # 7 种模型实现
│   ├── rule_based.py                # 产生式规则系统
│   ├── naive_bayes.py               # 朴素贝叶斯
│   ├── logistic_regression.py       # 逻辑回归
│   ├── feature_selection.py         # 特征选择 + LR
│   ├── logistic_regression_features.py  # 新特征工程 + LR
│   ├── sentiwordnet.py              # SentiWordNet 情感分析
│   └── llm.py                       # 大语言模型
├── A_output/               # A 模块的输出（B 读取）
└── B_output/               # B 模块的输出（C 读取）
    └── charts/             # 可视化图表输出
```

## 数据流

```
 A 模块                        B 模块                        C 模块
────────                     ──────────                    ─────────
x_train.npz ──┐
y_train.npy ──┤
x_test.npz  ──┼──→ train.py ──→ best_model.pkl    ──→ 加载模型评估
y_test.npy  ──┤               best_params.json        计算指标
vectorizer.pkl ──┘            predictions.npy         画图制表
                              probas.npy               错误分析
```

## 接口契约

### A → B（A 必须产出的 5 个文件）

| 文件 | 格式 | 内容 |
|------|------|------|
| `x_train.npz` | scipy 稀疏矩阵 | 训练集 TF-IDF 特征 |
| `y_train.npy` | numpy 数组 | 训练集标签 (0/1) |
| `x_test.npz` | scipy 稀疏矩阵 | 测试集 TF-IDF 特征 |
| `y_test.npy` | numpy 数组 | 测试集标签 (0/1) |
| `vectorizer.pkl` | pickle 对象 | TfidfVectorizer 实例 |

### B → C（B 会产出的 4 个文件）

| 文件 | 格式 | 内容 |
|------|------|------|
| `best_model.pkl` | pickle 对象 | F1 最高的可序列化模型 |
| `best_params.json` | JSON | 最优模型参数 + 全部模型指标对比 |
| `predictions.npy` | numpy 数组 | 测试集预测标签 |
| `probas.npy` | numpy 数组 | 测试集预测概率 |

## 使用方式

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 独立测试（不需要 A 模块）

```bash
python train.py --mock
```

会自动在 `A_output/` 生成模拟数据，训练全部模型，结果写入 `B_output/`。

### 3. 正式运行

```bash
python train.py
```

前提是 A 已将 5 个文件放入 `A_output/`。

### 4. 查看结果

`B_output/best_params.json` 包含最优模型参数和所有模型指标对比：

```json
{
  "model_type": "LogisticRegression",
  "metrics": { "accuracy": 0.8893, "f1": 0.8893, "best_C": 1.0 },
  "all_model_results": {
    "产生式规则系统": { "accuracy": 0.729, "f1": 0.762 },
    "朴素贝叶斯":     { "accuracy": 0.856, "f1": 0.853 },
    "逻辑回归":       { "accuracy": 0.889, "f1": 0.889 },
    "特征选择+LR":    { "accuracy": 0.877, "f1": 0.877 },
    "新特征+LR":      { "accuracy": 0.892, "f1": 0.892 },
    "SentiWordNet":   { "accuracy": 0.681, "f1": 0.712 },
    "大语言模型":     { "accuracy": 0.852, "f1": 0.848 }
  }
}
```

## 7 种模型

| 模型 | 需要训练 | 可 pickle | 说明 |
|------|---------|----------|------|
| 产生式规则系统 | 否 | 否 | VADER 情感词典 + 关键词匹配 |
| 朴素贝叶斯 | 是 | 是 | MultinomialNB / ComplementNB，网格搜索 alpha |
| 逻辑回归 | 是 | 是 | L2 正则，网格搜索 C，可提取特征权重 |
| 特征选择+LR | 是 | 是 | Chi2 / 互信息 + SelectKBest + LR Pipeline |
| 新特征+LR | 是 | 是 | TF-IDF + 8 个手工特征（词数、情感词密度、标点等）|
| SentiWordNet | 否 | 否 | WordNet 情感词典打分，计算 pos/neg/obj 分数 |
| 大语言模型 | 否 | 否 | 调用 OpenAI 兼容 API（默认 DeepSeek）|

规则系统、SentiWordNet 和 LLM 返回的 model 不可 pickle，不参与最优模型选择，仅在指标对比中展示。

## 大语言模型配置

默认使用 DeepSeek，复制 `.env.example` 为 `.env` 并填入 API Key：

```bash
cp .env.example .env
# 编辑 .env 填入你的 LLM_API_KEY
```

也可直接设置环境变量：

```bash
export LLM_API_KEY="your-key"
export LLM_BASE_URL="https://api.deepseek.com"
export LLM_MODEL="deepseek-chat"
```

支持 OpenAI 兼容厂商（智谱、通义千问、Moonshot 等），改 `LLM_BASE_URL` 和 `LLM_MODEL` 即可。不设 key 时 LLM 自动跳过。

## 可视化分析

训练完成后运行 `analyze.py` 生成对比图表：

```bash
python analyze.py
```

会在 `B_output/charts/` 输出：
- `model_comparison.png` — 所有模型 Accuracy/F1 对比柱状图
- `new_feature_importance.png` — 新特征系数重要性（需新特征+LR 模型有结果）
- `sentiwordnet_summary.png` — SentiWordNet 分析摘要

## 执行流程

```
train.py
  │
  ├── ① load_a_output()
  │     从 A_output/ 读取数据，校验文件完整性
  │
  ├── ② train_all_models()
  │     依次执行 7 个模型，记录指标到 all_model_results
  │     按 F1 选出最优的可序列化模型
  │
  └── ③ save_b_output()
        保存最优模型 + 全部指标 + 预测值 + 概率到 B_output/

analyze.py  ← 训练后运行，读取 B_output/ 生成图表

模型                   Accuracy        F1
----------------------------------------
产生式规则系统            0.7290    0.7620
朴素贝叶斯              0.8560    0.8530
逻辑回归                0.8893    0.8893  ← 最优
特征选择+LR             0.8770    0.8770
新特征+LR              0.8920    0.8920
SentiWordNet           0.6810    0.7120
大语言模型              0.8520    0.8480
----------------------------------------
最优模型: LogisticRegression
```

## 常见问题

**Q: mock 数据为什么准确率 100%？**
mock 数据用固定的正/负面词汇生成，信号太强。真实 IMDB 数据集参考范围见上方示例。

**Q: 和 A、C 的路径不一样怎么办？**
修改 `config.py` 中的路径常量。

**Q: LLM 怎么换厂商？**
改 `LLM_BASE_URL` 和 `LLM_MODEL` 环境变量即可，接口都是 OpenAI 兼容的。
