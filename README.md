# B 模块：模型训练

## 概述

B 模块负责从 A 模块接收预处理后的数据，训练并对比多种分类模型，最终选出最优模型交付给 C 模块做评估和展示。

**核心定位**：B 是数据管道中间的一环，不自己生产数据，不自己画图表，只做「读入 → 训练 → 输出」这一件事。

## 项目结构

```
B_model/
├── config.py               # 路径和接口约定
├── train.py                # 主程序入口
├── mock_data.py            # 模拟A模块数据（独立测试用）
├── README.md               # 本文档
├── models/                 # 模型实现
│   ├── __init__.py
│   ├── rule_based.py           # 产生式规则系统
│   ├── naive_bayes.py          # 朴素贝叶斯
│   ├── logistic_regression.py  # 逻辑回归
│   └── feature_selection.py    # 特征选择 + LR
├── A_output/               # A 模块放在这里的文件（B 读取）
└── B_output/               # B 模块的输出（C 读取）
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

## 接口契约（必须和 A、C 统一）

### A → B（A 必须产出的 5 个文件）

| 文件 | 格式 | 内容 | 注意 |
|------|------|------|------|
| `x_train.npz` | scipy 稀疏矩阵 | 训练集 TF-IDF 特征 | 用 `scipy.sparse.save_npz` 保存 |
| `y_train.npy` | numpy 数组 | 训练集标签，shape=(n,) | 值必须是 0 或 1 |
| `x_test.npz` | scipy 稀疏矩阵 | 测试集 TF-IDF 特征 | 同上 |
| `y_test.npy` | numpy 数组 | 测试集标签，shape=(n,) | 值必须是 0 或 1 |
| `vectorizer.pkl` | pickle 对象 | `TfidfVectorizer` 实例 | **必须传**，规则系统需要它还原文本 |

### B → C（B 会产出的 4 个文件）

| 文件 | 格式 | 内容 |
|------|------|------|
| `best_model.pkl` | pickle 对象 | 4 个模型中 F1 最高的那个模型 |
| `best_params.json` | JSON | 所有模型的指标、最优超参数、LR 特征词 |
| `predictions.npy` | numpy 数组 | 测试集预测标签，shape=(n,) |
| `probas.npy` | numpy 数组 | 测试集预测概率，shape=(n, 2) |

### 标签编码约定

```
0 = Negative（负面）
1 = Positive（正面）
```

## 使用方式

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 独立测试（不需要 A 模块）

还没拿到 A 的数据时，先用 `--mock` 生成模拟数据自测：

```bash
python train.py --mock
```

这条命令会：
1. 调用 `mock_data.py` 在 `A_output/` 下生成 5 个模拟文件
2. 自动加载这些文件，训练全部模型
3. 把结果保存到 `B_output/`

### 3. 正式运行

A 把数据放到 `A_output/` 后：

```bash
python train.py
```

### 4. 查看结果

训练完成后查看 `B_output/best_params.json`：
```json
{
  "model_type": "LogisticRegression",
  "metrics": {
    "accuracy": 0.8893,
    "precision": 0.8846,
    "recall": 0.8941,
    "f1": 0.8893,
    "best_C": 1.0,
    "top_features": {
      "positive": [["great", 7.015], ["excellent", 6.190], ...],
      "negative": [["worst", -8.651], ["bad", -7.616], ...]
    }
  },
  "all_model_results": {
    "产生式规则系统": { "accuracy": 0.729, "f1": 0.762, ... },
    "朴素贝叶斯":     { "accuracy": 0.856, "f1": 0.853, ... },
    "逻辑回归":       { "accuracy": 0.889, "f1": 0.889, ... },
    "特征选择+LR":    { "accuracy": 0.877, "f1": 0.877, ... }
  }
}
```

## 模型详解

### 产生式规则系统
- **原理**：手动构建 60 个正面词和 60 个负面词的情感词典，结合 NLTK VADER 预训练模型打分，加权融合后判定
- **特点**：不需要训练，直接推理，速度快；但准确率受限于词典覆盖度
- **依赖**：NLTK `vader_lexicon`，`vectorizer.pkl`（还原文本）

### 朴素贝叶斯
- **原理**：对 TF-IDF 特征用 MultinomialNB 和 ComplementNB 两种变体，网格搜索平滑参数 alpha
- **特点**：训练快，适合文本稀疏特征；ComplementNB 对类别不平衡更鲁棒
- **搜索空间**：alpha ∈ {0.05, 0.1, 0.5, 1.0, 2.0}，3 折交叉验证

### 逻辑回归
- **原理**：线性分类器 + L2 正则化，网格搜索正则强度 C
- **特点**：效果通常最好，能提取每个特征词的权重系数供 C 解释和可视化
- **搜索空间**：C ∈ {0.1, 0.5, 1.0, 3.0, 5.0, 10.0}，3 折交叉验证

### 特征选择 + LR
- **原理**：先用卡方检验 / 互信息筛选出最有效的 k 个特征，再用逻辑回归做分类
- **特点**：减少噪音特征、加快训练、便于分析哪些词最重要
- **搜索空间**：选择方法 × {卡方, 互信息}；k × {500, 1000, 2000, 5000}；C × {0.5, 1.0, 3.0}

## `train.py` 执行流程

```
train.py
  │
  ├── ① load_a_output()
  │     从 A_output/ 读取稀疏矩阵、标签、vectorizer
  │     校验文件是否存在、shape 是否匹配
  │
  ├── ② train_all_models()
  │     依次执行 4 个模型的训练函数：
  │     ├── train_rule_based()
  │     │   用 vectorizer 还原文本 → VADER + 词典打分 → 预测
  │     ├── train_naive_bayes()
  │     │   网格搜索 alpha → 对比 MultinomialNB vs ComplementNB → 选最优
  │     ├── train_logistic_regression()
  │     │   网格搜索 C → 训练 → 提取特征词权重
  │     └── train_with_feature_selection()
  │         对比 Chi2 vs 互信息 × 多种 k → Pipeline → 选最优
  │     记录每个模型的指标，按 F1 排序
  │
  ├── ③ save_b_output()
  │     保存最优模型、所有指标、预测值、概率到 B_output/
  │
  └── ④ 打印汇总表格
```

## 错误处理

| 情况 | 行为 |
|------|------|
| A_output/ 缺少某个文件 | 报错并明确提示缺少哪个文件 |
| 某个模型训练失败 | 跳过该模型继续训其他的，在结果中标记 `ERROR` |
| 所有模型都失败 | 抛出 `RuntimeError` 终止 |
| 特征数少于 k 值 | 自动降级为 `[n_features/2, n_features]` |
| 所有预测是同一类 | 规则系统正常输出，但 C 那边 precision/recall 会为 0 |

## 常见问题

**Q: 为什么规则系统需要 vectorizer？**
A: TF-IDF 矩阵是数值化的词频向量，规则系统需要原始文本才能做关键词匹配和 VADER 打分。`vectorizer.inverse_transform()` 可以把稀疏矩阵还原回词序列。

**Q: mock 数据为什么准确率 100%？**
A: mock 数据用固定的正/负面词汇生成，信号太强。真实 IMDB 数据集的结果参考范围：朴素贝叶斯 ~85%，逻辑回归 ~89%，规则系统 ~73%。

**Q: B 模块能不能改模型？**
A: 可以。在 `models/` 下新建一个 `.py` 文件，实现一个 `train_xxx(x_train, y_train, x_test, y_test, vectorizer=None)` 函数，返回 `(model, metrics_dict, predictions, probas)` 即可。然后在 `train.py` 的 `train_all_models()` 里加一行调用。

**Q: 和 A、C 的路径不一样怎么办？**
A: 修改 `config.py` 中的路径常量即可，不要硬编码任何路径在模型代码里。
