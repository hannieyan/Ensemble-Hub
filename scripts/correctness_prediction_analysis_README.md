# 正确性预测分析脚本

## 概述

`correctness_prediction_analysis.py` 是一个用于分析32B大模型辅助对7B小模型性能影响的机器学习工具。该脚本能够：

1. **自动评估答案正确性**：集成 `eval_acc_lm_eval.py` 进行自动化评估
2. **识别辅助效果的4种情况**：
   - ✅ 对→对：原本正确，辅助后仍正确（保持良好）
   - ❌ **对→错：原本正确，辅助后变错误（需要避免）**
   - 🔄 错→错：原本错误，辅助后仍错误（无改善）
   - 🎯 错→对：原本错误，辅助后变正确（显著改善）
3. **训练预测模型**：专门识别可能导致"对→错"的有害辅助
4. **选择性应用辅助**：通过预测过滤掉有害的辅助，实现最优性能

## 核心思想

### 问题背景
在Ensemble-Hub的improve模式中：
- 前1000个token由32B大模型生成（高质量辅助）
- 后续token由7B小模型继续生成
- 但有时32B的辅助反而会误导7B模型，导致原本正确的答案变错

### 解决方案
通过分析32B辅助token的统计特征（PPL、熵等），我们可以预测这种辅助是否会产生负面影响，从而选择性地应用辅助：

- **低PPL（困惑度）**：模型对文本更确定，通常表示答案更流畅
- **适中Entropy（熵）**：既不过于确定也不过于不确定
- **趋势特征**：token序列中PPL/熵的变化模式

## 使用方法

### 基本用法

```bash
# 使用默认的base.jsonl和enhanced1000.jsonl进行分析
python scripts/correctness_prediction_analysis.py

# 指定特定的文件
python scripts/correctness_prediction_analysis.py \
    --base saves/base.jsonl \
    --enhanced saves/enhanced1000.jsonl \
    --output-dir saves/impact_analysis

# 调整分析参数
python scripts/correctness_prediction_analysis.py \
    --n-tokens 1000 \          # 分析前1000个辅助token
    --threshold 0.3 \          # 更严格的过滤阈值
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--base` | `saves/base.jsonl` | 7B模型单独推理的结果文件 |
| `--enhanced` | `saves/enhanced1000.jsonl` | 32B+7B混合推理的结果文件 |
| `--output-dir` | `saves/assistance_impact_output` | 输出目录 |
| `--model` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 用于特征提取的模型 |
| `--n-tokens` | `1000` | 分析的辅助token数量 |
| `--threshold` | `0.5` | 过滤有害辅助的概率阈值 |

## 工作流程

### 1. 自动评估
- 检查是否存在评估结果文件
- 如不存在，自动调用 `eval_acc_lm_eval.py` 进行评估
- 保存评估结果避免重复计算

### 2. 数据匹配
- 按问题文本匹配base和enhanced结果
- 添加正确性标签到每个样本

### 3. 影响分类
- 将每个样本分类到4种影响类别
- 统计各类别的数量和比例

### 4. 特征提取
- 从32B辅助token中提取16个特征
- 包括PPL、熵的统计量和趋势特征

### 5. 模型训练
- 训练逻辑回归和随机森林模型
- 专门预测"对→错"的有害情况

### 6. 性能计算
- 计算base、enhanced、filtered三种策略的精度
- 分析过滤效果和改进幅度

## 特征工程

脚本从前N个辅助token（默认1000）中提取16个特征：

### PPL特征（7个）
- `assistance_ppl_mean`: 平均困惑度
- `assistance_ppl_std/median/max/min`: 标准差、中位数、最值
- `assistance_ppl_q25/q75`: 25%/75%分位数

### Entropy特征（7个）
- `assistance_entropy_mean`: 平均熵
- `assistance_entropy_std/median/max/min`: 标准差、中位数、最值
- `assistance_entropy_q25/q75`: 25%/75%分位数

### 趋势特征（2个）
- `assistance_ppl_trend`: 前20 vs 后20个token的PPL变化
- `assistance_entropy_trend`: 前20 vs 后20个token的Entropy变化

### 其他特征（1个）
- `assistance_token_count`: 实际分析的token数量

## 输出文件

### 1. 可视化图表
- `impact_overview.png`：影响分布饼图和精度对比
- `harm_predictors.png`：有害辅助的特征重要性分析
- `harm_analysis.png`：PPL/熵分布和预测性能曲线

### 2. 数据文件
- `correct_to_correct_samples.jsonl`：保持正确的样本
- `correct_to_incorrect_samples.jsonl`：**变差的样本（重点关注）**
- `incorrect_to_correct_samples.jsonl`：改善的样本
- `incorrect_to_incorrect_samples.jsonl`：无变化的样本
- `analysis_summary.json`：完整分析结果摘要

### 3. 自动生成的评估文件
- `base_eval_results.jsonl`：base结果的正确性评估
- `enhanced1000_eval_results.jsonl`：enhanced结果的正确性评估

## 结果解释

### 典型输出示例

```
=== Impact Analysis ===
correct_to_correct: 320 (64.0%)
correct_to_incorrect: 40 (8.0%)     # 这是我们要避免的
incorrect_to_correct: 65 (13.0%)    # 这是我们想要的
incorrect_to_incorrect: 75 (15.0%)

=== Training Harm Predictor ===
Logistic Regression F1: 0.815 (+/- 0.038)
Random Forest F1: 0.847 (+/- 0.042)

=== Accuracy Analysis ===
Base (7B only) accuracy: 0.720 (360/500)
Enhanced (always use 32B) accuracy: 0.770 (385/500)
Filtered (selective 32B) accuracy: 0.825 (412/500)

Filtering Statistics:
  Kept assistance: 380 (76.0%)
  Rejected assistance: 120 (24.0%)
  Avoided harmful cases: 35
  Missed helpful cases: 8

=== RECOMMENDATION ===
Using selective 32B assistance improves accuracy by 10.5%
From 72.0% → 82.5%
```

### 关键指标说明

1. **影响分布**：
   - 高比例的"对→对"说明辅助总体是有益的
   - 关注"对→错"的比例，这是需要过滤的重点

2. **预测器性能**：
   - F1分数衡量识别有害辅助的能力
   - 0.8+的F1分数表示良好的预测性能

3. **精度提升**：
   - **Base**: 仅使用7B模型的基准精度
   - **Enhanced**: 总是使用32B辅助的精度
   - **Filtered**: 选择性使用32B辅助的最终精度

4. **过滤统计**：
   - **Avoided harmful cases**: 成功避免的"对→错"情况
   - **Missed helpful cases**: 误过滤的"错→对"情况
   - 理想情况是前者多、后者少

## 应用场景

### 1. 推理时选择性辅助
```python
# 伪代码示例
assistance_features = extract_features(assistance_tokens)
if harm_predictor.predict_proba(assistance_features) < threshold:
    use_32b_assistance()
else:
    use_7b_only()
```

### 2. 训练数据筛选
- 过滤掉可能产生负面影响的训练样本
- 专注于真正有帮助的辅助案例

### 3. 模型改进方向
- 分析"对→错"案例的共同特征
- 针对性地改进32B模型的辅助策略

## 最佳实践

### 1. 阈值调优
- **较低阈值（0.3）**：更激进地过滤，精度优先
- **较高阈值（0.7）**：更多保留辅助，召回优先
- **建议0.5**：平衡精度和召回

### 2. 样本要求
- 确保有足够的"对→错"样本训练预测器
- 至少需要几十个有害案例才能有效训练

### 3. 定期更新
- 随着模型更新，重新分析辅助影响
- 调整预测器和阈值参数

## 技术细节

### 计算复杂度
- 特征提取：O(n × m)，n为样本数，m为token数
- 预测器训练：O(n × k²)，k为特征数
- 推理时预测：O(1)，实时可用

### 内存需求
- 需要加载7B模型进行特征提取
- 建议至少16GB GPU内存

### 依赖项
- transformers（模型加载）
- scikit-learn（机器学习）
- matplotlib/seaborn（可视化）
- jsonlines（数据处理）
- subprocess（自动评估）

## 与其他脚本的关系

- **`eval_acc_lm_eval.py`**：被自动调用进行答案正确性评估
- **`complete_token_analysis.py`**：共享TokenAnalyzer类和部分工具函数
- **`grader.py`**：用于手动评估答案正确性（备用方案）

## 局限性和注意事项

### 1. 数据依赖
- **领域特异性**：在数学题上训练，可能不适用其他领域
- **模型特异性**：基于特定7B模型的特征
- **Token限制**：只分析前N个token，长答案的后续部分不被考虑

### 2. 因果关系
- **相关vs因果**：低PPL和正确性可能都由第三因素导致
- **模型偏见**：7B模型的偏见可能影响预测

### 3. 性能上限
- **天花板效应**：基于统计特征的预测有天然上限
- **复杂推理**：无法捕捉深层逻辑错误

### 4. 阈值调整
- **数据集差异**：不同数据集的最优阈值可能不同
- **需重新校准**：在新数据上使用前应重新评估