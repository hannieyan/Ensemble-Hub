# Correctness Prediction Analysis Script

## 概述

`correctness_prediction_analysis.py` 是一个机器学习脚本，用于分析是否可以通过7B模型计算的PPL（困惑度）和Entropy（熵）指标来预测数学题答案的正确性。该脚本实现了多种预测模型，并提供详细的性能分析和可视化。

## 核心思想

### 假设
- **低PPL（困惑度）**：模型对文本更确定，可能表示答案更流畅/正确
- **适中Entropy（熵）**：既不过于确定（可能过拟合）也不过于不确定
- **Token序列特征**：正确答案可能具有不同的统计特性

### 特征工程
脚本分析每个回答的**前100个token**（可通过`--n-tokens`调整），从中提取17个特征：

**重要**：当使用问题上下文时，100个token限制仅适用于回答部分，问题用作上下文但不计入token计数。

#### PPL特征（7个）
- `ppl_mean`: 平均困惑度
- `ppl_std`: 困惑度标准差
- `ppl_median`: 困惑度中位数
- `ppl_max/min`: 最大/最小困惑度
- `ppl_q25/q75`: 25%/75%分位数

#### Entropy特征（7个）
- `entropy_mean`: 平均熵
- `entropy_std`: 熵标准差
- `entropy_median`: 熵中位数
- `entropy_max/min`: 最大/最小熵
- `entropy_q25/q75`: 25%/75%分位数

#### 趋势特征（2个）
- `ppl_trend`: 前20 vs 后20个token的PPL变化（限制在分析的token范围内）
- `entropy_trend`: 前20 vs 后20个token的Entropy变化（限制在分析的token范围内）

#### 其他特征（2个）
- `response_length`: 回答长度（词数）
- `token_count`: 实际分析的token数量（≤ n_tokens，默认≤100）

## 预测模型

### 1. 阈值模型（Threshold）
- **方法**：寻找最优PPL阈值来分类
- **逻辑**：PPL < threshold → 正确
- **优点**：简单、可解释
- **适用**：快速筛选

### 2. 逻辑回归（Logistic Regression）
- **方法**：线性分类器，使用所有特征
- **优点**：系数可解释、训练快速
- **输出**：概率预测 + 特征重要性

### 3. 随机森林（Random Forest）
- **方法**：集成树模型
- **优点**：处理非线性关系、特征重要性
- **适用**：复杂模式识别

## 使用方法

### 基本用法
```bash
# 使用默认base.jsonl文件分析
python scripts/correctness_prediction_analysis.py

# 分析enhanced模型的结果
python scripts/correctness_prediction_analysis.py --data saves/enhanced.jsonl

# 只分析回答部分（不包含问题）
python scripts/correctness_prediction_analysis.py --response-only

# 分析更多token
python scripts/correctness_prediction_analysis.py --n-tokens 200
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | `saves/base.jsonl` | 输入数据文件路径 |
| `--output-dir` | `saves/correctness_prediction_output` | 输出目录 |
| `--model` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 用于特征提取的模型 |
| `--n-tokens` | `100` | 分析的token数量 |
| `--response-only` | `False` | 是否只分析回答文本 |

## 输出文件

### 1. 可视化图表
- `correctness_prediction_overview.png`：总览图表
  - PPL/Entropy分布对比
  - 散点图分析
  - 模型性能对比
- `performance_curves.png`：性能曲线
  - ROC曲线
  - Precision-Recall曲线
- `feature_importance.png`：特征重要性
  - 逻辑回归系数
  - 随机森林重要性
- `confusion_matrices.png`：混淆矩阵

### 2. 数据文件
- `prediction_summary.json`：预测结果摘要
- `feature_data.jsonl`：完整特征数据

## 结果解释

### 典型输出示例
```
=== Model Performance ===

1. Threshold-based Predictor (PPL mean)
Best threshold: PPL < 85.23
Test accuracy: 0.647

2. Logistic Regression
Test accuracy: 0.724
Top 5 features (by coefficient magnitude):
  ppl_mean: -0.892
  entropy_std: 0.445
  ppl_q75: -0.321
  response_length: 0.284
  entropy_mean: -0.201

3. Random Forest
Test accuracy: 0.756
Top 5 features (by importance):
  ppl_mean: 0.187
  entropy_mean: 0.143
  ppl_std: 0.089
  response_length: 0.076
  ppl_median: 0.068

4. Cross-validation scores (5-fold)
Logistic Regression: 0.701 (+/- 0.048)
Random Forest: 0.739 (+/- 0.052)
```

### 关键发现

#### 1. PPL是最重要的预测因子
- **负相关**：PPL越低，正确概率越高
- **阈值效应**：存在明显的分界点
- **解释**：正确答案通常更流畅，PPL更低

#### 2. Entropy提供补充信息
- **复杂关系**：不是简单的线性关系
- **标准差重要**：熵的变化比平均值更有预测性
- **平衡性**：既不能太确定也不能太不确定

#### 3. 回答长度有影响
- **长度效应**：更长的回答可能更正确（更详细的推理）
- **但需谨慎**：可能只是相关性，不是因果性

### 模型比较

| 模型 | 准确率 | 优点 | 缺点 |
|------|--------|------|------|
| 阈值 | ~65% | 简单、快速、可解释 | 只用一个特征，性能有限 |
| 逻辑回归 | ~72% | 线性可解释、训练快 | 假设线性关系 |
| 随机森林 | ~76% | 处理非线性、特征重要性 | 黑盒模型、过拟合风险 |

## 应用场景

### 1. 答案质量筛选
```python
# 使用训练好的模型预测新答案
if ppl_mean < threshold:
    print("可能正确，建议采用")
else:
    print("可能错误，需要检查")
```

### 2. 模型选择策略
- **高PPL答案**：可能需要更强的模型
- **低PPL但错误**：可能是系统性偏见
- **中等PPL**：最难预测的区域

### 3. 训练数据筛选
- 过滤明显错误的训练样本
- 识别需要人工检查的边界案例

## 局限性和注意事项

### 1. 数据依赖
- **领域特异性**：在数学题上训练，可能不适用其他领域
- **模型特异性**：基于特定7B模型的特征
- **Token限制**：只分析前100个token，长答案的后续部分不被考虑

### 2. 因果关系
- **相关vs因果**：低PPL和正确性可能都由第三因素导致
- **模型偏见**：7B模型的偏见可能影响预测

### 3. 阈值调整
- **数据集差异**：不同数据集的最优阈值可能不同
- **需重新校准**：在新数据上使用前应重新评估

### 4. 性能上限
- **天花板效应**：基于统计特征的预测有天然上限
- **复杂推理**：无法捕捉深层逻辑错误

## 扩展方向

### 1. 特征增强
- **语义特征**：使用embedding相似度
- **结构特征**：数学公式结构分析
- **时序特征**：token生成的时序信息

### 2. 模型改进
- **集成方法**：结合多个模型的预测
- **深度学习**：使用神经网络进行端到端学习
- **主动学习**：优先标注最不确定的样本

### 3. 多模态扩展
- **多模型融合**：结合不同规模模型的判断
- **人工反馈**：结合人类专家的判断

## 技术细节

### 环境要求
- Python 3.7+
- PyTorch（用于DeepSeek模型）
- scikit-learn（机器学习模型）
- matplotlib, seaborn（可视化）

### 计算资源
- **GPU推荐**：特征提取需要运行7B模型
- **内存需求**：至少16GB RAM用于模型加载
- **时间复杂度**：O(n×k)，n为样本数，k为token数

### 可复现性
- 使用固定随机种子（random_state=42）
- 标准化的数据分割比例（80%训练，20%测试）
- 详细的超参数记录