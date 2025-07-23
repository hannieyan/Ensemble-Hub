# Single Sample Token Analysis Script

## 概述

`single_sample_token_analysis.py` 是基于 `complete_token_analysis.py` 的精细化分析工具，允许用户选择特定的样本进行详细的token级别分析和可视化。该脚本特别适用于深入研究特定improvement或regression案例的token模式。

## 主要特点

### 1. 灵活的样本选择
- **样本数量控制**：通过`--sample-count`控制分析的样本数量
- **起始位置控制**：通过`--start-index`选择从第几个样本开始分析
- **智能分配**：自动在improvement和regression样本间分配

### 2. 详细的Token级别可视化
- **PPL趋势图**：每个token位置的困惑度变化（对数坐标）
- **Entropy趋势图**：每个token位置的信息熵变化
- **Base vs Enhanced对比**：直观展示模型改进效果
- **样本标识**：每条线都有清晰的样本标签

**重要**：分析的token位置对应回答部分的前N个token，问题部分用作上下文但不显示在图表中。

### 3. 智能样本分配逻辑
- 样本按照 improvement → regression 的顺序编号
- 自动处理跨类别的样本选择
- 如果起始索引超出improvement范围，自动显示regression样本

## 使用方法

### 基本用法
```bash
# 分析第1个样本（索引0）
python scripts/single_sample_token_analysis.py --sample-count 1 --start-index 0

# 分析第5个样本（索引4）
python scripts/single_sample_token_analysis.py --sample-count 1 --start-index 4

# 分析从第10个样本开始的3个样本
python scripts/single_sample_token_analysis.py --sample-count 3 --start-index 9
```

### 高级用法
```bash
# 只分析回答部分（不包含问题）
python scripts/single_sample_token_analysis.py --sample-count 1 --start-index 0 --response-only

# 分析更多token
python scripts/single_sample_token_analysis.py --sample-count 1 --start-index 0 --n-tokens 200

# 使用top-k entropy计算方法
python scripts/single_sample_token_analysis.py --sample-count 1 --start-index 0 --entropy-method top-k
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--base` | `saves/base.jsonl` | Base模型结果文件路径 |
| `--enhanced` | `saves/enhanced.jsonl` | Enhanced模型结果文件路径 |
| `--output-dir` | `saves/single_sample_analysis_output` | 输出目录 |
| `--model` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 用于token分析的模型 |
| `--n-tokens` | `100` | 分析的token数量 |
| `--sample-count` | `1` | **要分析的样本数量** |
| `--start-index` | `0` | **起始样本索引（0-based）** |
| `--include-problem` | `True` | 是否在分析中包含问题文本 |
| `--response-only` | `False` | 只分析回答，排除问题 |
| `--entropy-method` | `full` | Entropy计算方法：`full` 或 `top-k` |

## 样本索引逻辑

### 索引分配规则
假设有56个improvement样本和28个regression样本：

- **索引 0-27**：两种类型都有样本，会同时显示improvement和regression的图
- **索引 28-55**：只有improvement有样本，只显示improvement的图
- **索引 ≥56**：两种类型都没有样本，显示错误

### 使用示例

```bash
# 示例1：分析第20个样本（两种类型都有），会显示improvement[20]和regression[20]
python scripts/single_sample_token_analysis.py --sample-count 1 --start-index 20

# 示例2：分析第40个样本（只有improvement有），只显示improvement[40]
python scripts/single_sample_token_analysis.py --sample-count 1 --start-index 40

# 示例3：分析第1个样本（两种类型都有），会显示improvement[0]和regression[0]
python scripts/single_sample_token_analysis.py --sample-count 1 --start-index 0

# 示例4：分析从第25个样本开始的5个样本
# - improvement[25-27] 和 regression[25-27]：两种都显示
# - improvement[28-29]：只显示improvement
python scripts/single_sample_token_analysis.py --sample-count 5 --start-index 25

# 示例5：超出范围的索引会报错
python scripts/single_sample_token_analysis.py --sample-count 1 --start-index 60  # 错误
```

## 输出文件

### 1. 可视化图表
- `single_sample_analysis_start{索引}_count{数量}.png`：主要分析图表
  - 如果只有improvement：显示1行2列（PPL + Entropy）
  - 如果只有regression：显示1行2列（PPL + Entropy）
  - 如果同时有两种：显示2行2列（上排improvement，下排regression）

### 2. 详细数据
- `sample_details_start{索引}_count{数量}.json`：包含分析样本的详细统计信息

## 典型应用场景

### 1. 深入分析特定案例
```bash
# 分析最有代表性的improvement案例
python scripts/single_sample_token_analysis.py --sample-count 1 --start-index 10
```

### 2. 对比分析
```bash
# 分析相邻的几个样本，观察模式差异
python scripts/single_sample_token_analysis.py --sample-count 3 --start-index 20
```

### 3. 边界案例研究
```bash
# 分析improvement和regression的边界案例
python scripts/single_sample_token_analysis.py --sample-count 2 --start-index 55
# 这会显示最后一个improvement（索引55）和第一个regression（索引56）
```

### 4. 错误模式分析
```bash
# 专门分析某个problematic的regression案例
python scripts/single_sample_token_analysis.py --sample-count 1 --start-index 60
```

## 输出解读

### 图表解读
1. **PPL曲线**：
   - **蓝线（Base）**：原始模型的困惑度
   - **绿线（Enhanced）**：改进模型的困惑度（improvement案例）
   - **红线（Enhanced）**：改进模型的困惑度（regression案例）
   - **对数坐标**：更好地展示PPL的变化幅度

2. **Entropy曲线**：
   - **线性坐标**：熵值通常在较小范围内
   - **趋势观察**：观察模型在每个token位置的确定性变化

### 统计信息
```json
{
  "parameters": {
    "sample_count": 1,
    "start_index": 40,
    "total_improvements": 56,
    "total_regressions": 28
  },
  "analyzed_samples": {
    "improvements": [{
      "sample_index": 40,
      "global_index": 40,
      "base_stats": {
        "avg_ppl": 95.23,
        "avg_entropy": 2.34,
        "token_count": 95
      },
      "enhanced_stats": {
        "avg_ppl": 87.45,
        "avg_entropy": 2.12,
        "token_count": 98
      },
      "ppl_change": -7.78,
      "entropy_change": -0.22
    }]
  }
}
```

## 与其他脚本的关系

### 分析流程建议
1. **complete_token_analysis.py**：获得整体概览和统计
2. **ppl_distribution_analysis.py**：观察PPL分布模式
3. **single_sample_token_analysis.py**：深入分析特定样本

### 数据一致性
- 使用相同的token分析方法
- 相同的样本匹配逻辑
- 相同的特征提取算法

## 注意事项

### 1. 索引范围
- 确保`start_index`在有效范围内（0 到 总样本数-1）
- 超出范围会显示错误信息

### 2. 样本数量
- `sample_count`必须至少为1
- 如果请求的样本数量超过可用样本，会自动调整

### 3. 可视化限制
- 当样本数量较多时（>5），图例可能会重叠
- 建议单次分析不超过3个样本以保持可读性

### 4. 计算资源
- 每个样本都需要运行7B模型进行token分析
- 建议在有GPU的环境下运行

## 扩展建议

### 1. 交互式选择
可以结合其他脚本的输出来选择感兴趣的样本索引

### 2. 批量分析
可以写wrapper脚本来批量分析多个单独的样本

### 3. 对比模式
可以扩展为直接对比两个特定样本的功能