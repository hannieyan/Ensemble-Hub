# PPL Distribution Analysis Script

## 概述

`ppl_distribution_analysis.py` 是一个专门用于分析和可视化improvement和regression样本中困惑度（PPL）分布的工具。该脚本通过将样本按base模型的PPL从小到大排序，直观地展示了模型改进和退化的模式。

## 主要功能

### 1. PPL分布分析
- **自动数据加载**：复用`complete_token_analysis.py`的数据加载和匹配功能
- **PPL计算**：使用DeepSeek模型计算每个样本回答部分前100个token的平均困惑度
- **排序展示**：按base模型PPL从小到大排序，便于观察趋势

**重要**：分析的100个token仅限于回答部分，问题文本用作上下文但不计入token计数。

### 2. 可视化输出

#### 2.1 柱状图分布（ppl_distribution_sorted.png）
- **上图**：Improvement cases的PPL分布
  - 蓝色柱：Base模型的PPL
  - 绿色柱：Enhanced模型的PPL
- **下图**：Regression cases的PPL分布
  - 蓝色柱：Base模型的PPL
  - 红色柱：Enhanced模型的PPL

#### 2.2 趋势线图（ppl_distribution_trends.png）
- **折线图**：更清晰地展示PPL变化趋势
- **阴影区域**：
  - 绿色阴影：PPL下降（改进）
  - 红色阴影：PPL上升（恶化）

#### 2.3 统计摘要图（ppl_summary_statistics.png）
- **均值对比**：展示improvements和regressions的平均PPL及标准差
- **改进统计**：显示PPL下降和上升的样本数量

## 使用方法

### 基本用法
```bash
# 使用默认参数分析
python scripts/ppl_distribution_analysis.py

# 分析更多token（默认100）
python scripts/ppl_distribution_analysis.py --n-tokens 200

# 只分析回答部分（不包含问题）
python scripts/ppl_distribution_analysis.py --response-only

# 指定输出目录
python scripts/ppl_distribution_analysis.py --output-dir my_ppl_analysis
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--base` | `saves/base.jsonl` | Base模型结果文件路径 |
| `--enhanced` | `saves/enhanced.jsonl` | Enhanced模型结果文件路径 |
| `--output-dir` | `saves/ppl_distribution_output` | 输出目录 |
| `--model` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 用于PPL计算的模型 |
| `--n-tokens` | `100` | 分析的token数量 |
| `--response-only` | `False` | 是否只分析回答文本 |

## 输出文件

### 1. 可视化文件
- `ppl_distribution_sorted.png`：柱状图展示PPL分布
- `ppl_distribution_trends.png`：折线图展示PPL变化趋势
- `ppl_summary_statistics.png`：统计摘要图

### 2. 数据文件
- `ppl_distribution_results.json`：包含所有样本的PPL数据，按base PPL排序

## 分析见解

### 通过该分析可以观察到：

1. **Improvement模式**：
   - Base PPL较高的样本更容易被改进
   - Enhanced模型在困难样本上的表现提升更明显

2. **Regression模式**：
   - Base PPL较低的简单样本可能出现退化
   - Enhanced模型可能在某些简单任务上过度复杂化

3. **整体趋势**：
   - PPL分布的形状反映了模型的改进策略
   - 可以识别哪些PPL范围的样本最受益于增强

## 典型输出示例

```
=== Detailed PPL Statistics ===

Improvement Cases (n=58):
  Base PPL: 95.43 ± 45.67
  Enhanced PPL: 104.11 ± 42.31
  PPL Change: 8.69
  Cases with PPL reduction: 42 (72.4%)
  Cases with PPL increase: 16 (27.6%)

Regression Cases (n=28):
  Base PPL: 65.55 ± 38.92
  Enhanced PPL: 104.85 ± 52.14
  PPL Change: 39.30
  Cases with PPL reduction: 5 (17.9%)
  Cases with PPL increase: 23 (82.1%)
```

## 应用场景

1. **模型评估**：快速了解模型增强的效果分布
2. **错误分析**：识别哪些类型的样本容易出现退化
3. **改进方向**：指导后续模型优化的重点
4. **阈值设定**：帮助确定何时使用base或enhanced模型

## 注意事项

1. **计算资源**：PPL计算需要GPU，建议使用CUDA
2. **内存使用**：大量样本分析可能需要较多内存
3. **模型加载**：首次运行需要下载DeepSeek模型
4. **结果解释**：低PPL表示模型对文本更确定，但过低可能表示过拟合

## 与complete_token_analysis的区别

- **专注点**：本脚本专注于PPL分布的可视化分析
- **排序展示**：按base PPL排序，更直观地展示趋势
- **简化输出**：只关注PPL指标，不包含entropy等其他指标
- **快速分析**：适合快速了解模型改进的整体模式