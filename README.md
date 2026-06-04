# B 模块：模型训练

## 概述

B 模块负责从 A 模块接收预处理后的数据，训练逻辑回归分类模型，输出给 C 模块做评估和展示。

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
│   └── logistic_regression.py  # 逻辑回归
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
| `vectorizer.pkl` | pickle 对象 | `TfidfVectorizer` 实例 | 用于提取特征词名称 |

### B → C（B 会产出的 4 个文件）

| 文件 | 格式 | 内容 |
|------|------|------|
| `best_model.pkl` | pickle 对象 | 训练好的逻辑回归模型 |
| `best_params.json` | JSON | 模型指标、最优超参数、Top 特征词 |
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
2. 自动加载这些文件，训练模型
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
      "positive": [["great", 7.015], ["excellent", 6.190]],
      "negative": [["worst", -8.651], ["bad", -7.616]]
    }
  }
}
```

## 模型说明

逻辑回归 —— 线性分类器 + L2 正则化，网格搜索正则强度 C。
- **搜索空间**：C ∈ {0.1, 0.5, 1.0, 3.0, 5.0, 10.0}，3 折交叉验证
- **特点**：对 TF-IDF 稀疏特征效果好，能提取每个特征词的权重系数供 C 解释和可视化

## `train.py` 执行流程

```
train.py
  │
  ├── ① load_a_output()
  │     从 A_output/ 读取稀疏矩阵、标签、vectorizer
  │     校验文件是否存在
  │
  ├── ② train_logistic_regression()
  │     网格搜索 C → 训练 → 提取特征词权重 → 预测
  │
  └── ③ save_b_output()
        保存模型、指标、预测值、概率到 B_output/
```

## 错误处理

| 情况 | 行为 |
|------|------|
| A_output/ 缺少某个文件 | 报错并明确提示缺少哪个文件 |
| 模型训练失败 | 异常直接抛出，终止运行 |

## 常见问题

**Q: mock 数据为什么准确率很高？**
A: mock 数据用固定的正/负面词汇生成，信号太强。真实 IMDB 数据集逻辑回归 ~89%。

**Q: 和 A、C 的路径不一样怎么办？**
A: 修改 `config.py` 中的路径常量即可，不要硬编码任何路径在模型代码里。
