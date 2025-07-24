# 多长度辅助分析脚本

## 概述

`multi_length_assistance_analysis.py` 是一个用于分析和优化不同辅助长度（100、500、1000 tokens）对7B模型性能影响的高级工具。该脚本实现了渐进式辅助策略，能够训练预测模型来决定在什么时候使用多长的辅助。

## 核心思想

### 问题背景
在Ensemble-Hub中，我们有多种增强选项：
- `enhanced100.jsonl`：使用32B模型生成前100个token作为辅助
- `enhanced500.jsonl`：使用32B模型生成前500个token作为辅助  
- `enhanced1000.jsonl`：使用32B模型生成前1000个token作为辅助

不同长度的辅助对不同问题的影响各不相同，有时短辅助就足够了，有时需要更长的辅助才有帮助。

### 渐进式策略
理想的策略是：
1. **先尝试短辅助（100 tokens）**：如果预测有帮助就使用，否则继续
2. **再尝试中等辅助（500 tokens）**：如果预测有帮助就使用，否则继续  
3. **最后尝试长辅助（1000 tokens）**：如果预测有帮助就使用，否则用base
4. **如果都没帮助**：直接使用7B模型的base结果

这样可以在保证性能的同时，最小化计算成本（优先使用短辅助）。

## 主要功能

1. **多长度影响分析**：识别每个问题的最优辅助策略
2. **渐进式预测器训练**：为每个长度训练二分类器
3. **策略模拟**：模拟渐进式决策过程
4. **性能对比**：与各种基线策略对比

## 使用方法

### 基本用法

```bash
# 使用默认文件进行分析
python scripts/multi_length_assistance_analysis.py

# 指定特定文件
python scripts/multi_length_assistance_analysis.py \
    --base saves/base.jsonl \
    --enhanced100 saves/enhanced100.jsonl \
    --enhanced500 saves/enhanced500.jsonl \
    --enhanced1000 saves/enhanced1000.jsonl \
    --output-dir saves/multi_length_analysis
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--base` | `saves/base.jsonl` | 7B模型单独推理的结果 |
| `--enhanced100` | `saves/enhanced100.jsonl` | 100token辅助的结果 |
| `--enhanced500` | `saves/enhanced500.jsonl` | 500token辅助的结果 |
| `--enhanced1000` | `saves/enhanced1000.jsonl` | 1000token辅助的结果 |
| `--output-dir` | `saves/multi_length_output` | 输出目录 |
| `--model` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 特征提取模型 |
| `--threshold` | `0.5` | 预测概率阈值 |

## 工作流程

### 1. 数据加载和匹配
- 自动评估所有文件的正确性
- 按问题文本匹配不同长度的结果
- 只保留在所有长度中都存在的问题

### 2. 最优策略识别
对每个问题，确定最优策略：
- `use_base`：不使用任何辅助
- `use_100`：使用100token辅助
- `use_500`：使用500token辅助  
- `use_1000`：使用1000token辅助

策略选择原则：使用能带来帮助的最短辅助。

### 3. 特征提取
为每个辅助长度提取16个特征：
- PPL特征（7个）：平均值、标准差、分位数等
- Entropy特征（7个）：平均值、标准差、分位数等
- 趋势特征（2个）：PPL和熵的变化趋势

### 4. 预测器训练
为每个长度训练二分类器：
- **输入**：该长度的token特征
- **输出**：是否应该使用这个长度的辅助
- **模型**：随机森林（class_weight='balanced'）

### 5. 渐进式策略模拟
模拟实际使用场景：
1. 分析100token特征，预测是否有帮助
2. 如果有帮助就停止，否则继续到500token
3. 分析500token特征，预测是否有帮助
4. 如果有帮助就停止，否则继续到1000token
5. 最后决定使用1000token辅助还是base结果

## 输出文件

### 1. 可视化图表
- `multi_length_overview.png`：策略分布和精度对比
- `predictor_performance.png`：各长度预测器的性能对比

### 2. 策略样本文件
- `use_base_samples.jsonl`：最优策略是不使用辅助的样本
- `use_100_samples.jsonl`：最优策略是使用100token辅助的样本
- `use_500_samples.jsonl`：最优策略是使用500token辅助的样本
- `use_1000_samples.jsonl`：最优策略是使用1000token辅助的样本

### 3. 分析结果
- `multi_length_summary.json`：完整分析结果摘要
- 各种基线策略的精度对比
- 预测器性能统计

## 结果解释

### 典型输出示例

```
=== Multi-Length Impact Analysis ===
Optimal strategy 'use_base': 120 (24.0%)
Optimal strategy 'use_100': 180 (36.0%)
Optimal strategy 'use_500': 120 (24.0%)
Optimal strategy 'use_1000': 80 (16.0%)

=== Baseline Accuracies ===
base_only: 0.720 (72.0%)
always_100: 0.756 (75.6%)
always_500: 0.768 (76.8%)
always_1000: 0.770 (77.0%)
optimal: 0.825 (82.5%)

=== Training Predictor for Length 100 ===
Length 100 - Test accuracy: 0.843
Length 100 - F1 score: 0.789 (+/- 0.045)

=== Simulating Progressive Strategy ===
Strategy Usage:
  base_only: 95 (19.0%)
  enhanced100: 190 (38.0%)
  enhanced500: 135 (27.0%)
  enhanced1000: 80 (16.0%)

Final Accuracy: 0.812 (406/500)

=== FINAL RESULTS ===
Base accuracy: 0.720
Progressive strategy accuracy: 0.812
Optimal strategy accuracy: 0.825
Improvement over base: 0.092 (9.2%)
Efficiency vs optimal: 98.4%
```

### 关键指标说明

1. **策略分布**：显示理论上最优的策略分布
   - 36%的问题用100token辅助就足够了
   - 24%的问题不需要任何辅助
   - 只有16%的问题需要完整的1000token辅助

2. **基线对比**：
   - `base_only`：从不使用辅助的精度
   - `always_X`：总是使用X长度辅助的精度
   - `optimal`：理论最优策略的精度上限

3. **预测器性能**：
   - F1分数衡量预测质量（0.8+表示良好）
   - 准确率表示分类精度

4. **最终效果**：
   - 渐进式策略接近最优策略（98.4%效率）
   - 相比base提升9.2%精度
   - 平均使用的辅助长度更短，计算成本更低

## 应用价值

### 1. 计算效率
- **平均辅助长度**：通过优先使用短辅助，大幅降低计算成本
- **动态停止**：不需要的情况下避免生成长辅助
- **资源优化**：32B模型的使用更加高效

### 2. 性能提升
- **接近最优**：98%+的最优策略效率
- **鲁棒性**：对不同类型问题都能做出合适决策
- **可解释性**：每个决策都基于明确的特征

### 3. 实际部署
```python
# 伪代码：实际部署逻辑
def progressive_assistance(problem):
    # 生成100token辅助
    assistance_100 = generate_assistance(problem, length=100)
    features_100 = extract_features(assistance_100)
    
    if predictor_100.predict_proba(features_100) > threshold:
        return use_assistance(assistance_100)
    
    # 继续生成到500token
    assistance_500 = extend_assistance(assistance_100, length=500)
    features_500 = extract_features(assistance_500)
    
    if predictor_500.predict_proba(features_500) > threshold:
        return use_assistance(assistance_500)
    
    # 继续生成到1000token
    assistance_1000 = extend_assistance(assistance_500, length=1000)
    features_1000 = extract_features(assistance_1000)
    
    if predictor_1000.predict_proba(features_1000) > threshold:
        return use_assistance(assistance_1000)
    else:
        return use_base_only()
```

## 技术创新

### 1. 渐进式决策
- 传统方法：固定长度辅助
- 本方法：动态长度选择，最小化不必要的计算

### 2. 多级预测
- 每个长度都有专门的预测器
- 考虑了不同长度的特征差异
- 避免了信息泄露（短长度预测时不使用长长度信息）

### 3. 实用性导向
- 模拟真实部署场景
- 考虑计算成本和性能的平衡
- 提供可直接应用的决策逻辑

## 局限性

### 1. 数据要求
- 需要所有长度的辅助结果文件
- 样本必须在所有长度中都存在
- 需要足够的训练样本

### 2. 模型假设
- 假设短辅助的特征能预测长辅助的效果
- 基于当前的特征工程（PPL、熵）
- 可能存在领域特异性

### 3. 计算复杂度
- 训练时需要为每个长度提取特征
- 推理时需要逐步生成和评估
- 仍然需要7B模型进行特征提取

## 扩展方向

### 1. 更多长度
- 支持更细粒度的长度选择（如50、200、750等）
- 动态长度选择而不是固定档位

### 2. 更好的特征
- 使用更深层的语义特征
- 结合问题类型信息
- 引入上下文相关性特征

### 3. 在线学习
- 根据实际使用效果调整预测器
- 适应不同领域和任务
- 个性化的辅助策略