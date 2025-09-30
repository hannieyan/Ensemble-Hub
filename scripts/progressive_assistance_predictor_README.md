# Progressive Assistance Predictor with RNN Support

## 概述

`progressive_assistance_predictor.py` 是一个避免数据泄露的渐进式辅助预测脚本，支持Random Forest和RNN两种预测模型。该脚本实现了正确的训练/测试分离，确保预测模型的有效性。

## 核心改进

### 1. 数据泄露问题解决
- **原有问题**：之前的脚本在训练时使用了完整的结果，相当于"先知道答案再训练"
- **解决方案**：严格的训练/测试分离
  - 训练集：`saves/train/` 目录下的数据
  - 测试集：`saves/test/` 目录下的数据
  - 预测器只在训练集上训练，在测试集上评估

### 2. RNN模型支持
新增RNN/LSTM模型用于序列化的渐进式决策：

```python
class ProgressiveAssistanceRNN(nn.Module):
    """RNN model for progressive assistance prediction."""
    
    def __init__(self, feature_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
            batch_first=True
        )
        
        # Binary classification output
        self.classifier = nn.Linear(hidden_dim, 2)
```

### 3. 渐进式序列建模
RNN模型将辅助决策建模为序列问题：
1. **时间步1**：基于100token特征，预测是否使用
2. **时间步2**：基于100+500token特征，预测是否使用500token
3. **时间步3**：基于100+500+1000token特征，预测是否使用1000token

## 使用方法

### 数据准备
确保有以下目录结构：
```
saves/
├── train/
│   ├── base.jsonl
│   ├── enhanced100.jsonl
│   ├── enhanced500.jsonl
│   └── enhanced1000.jsonl
└── test/
    ├── base.jsonl
    ├── enhanced100.jsonl
    ├── enhanced500.jsonl
    └── enhanced1000.jsonl
```

### Random Forest模型（默认）
```bash
python scripts/progressive_assistance_predictor.py \
    --data-dir saves \
    --model-type rf \
    --threshold 0.5
```

### RNN模型
```bash
python scripts/progressive_assistance_predictor.py \
    --data-dir saves \
    --model-type rnn \
    --threshold 0.5
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-dir` | `saves` | 包含train/test子目录的根目录 |
| `--model-type` | `rf` | 预测器类型：`rf`(随机森林) 或 `rnn`(RNN/LSTM) |
| `--threshold` | `0.5` | 预测概率阈值 |
| `--model` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 特征提取模型 |
| `--output-dir` | `saves/progressive_predictor_output` | 输出目录 |

## 模型比较

### Random Forest模型
- **优点**：训练快速，可解释性强，特征重要性清晰
- **缺点**：每个长度独立训练，不考虑序列关系
- **适用场景**：快速原型和基线对比

### RNN模型  
- **优点**：考虑序列依赖关系，更符合渐进式决策的本质
- **缺点**：训练时间较长，需要更多数据
- **适用场景**：追求最佳性能的生产环境

## 技术架构

### 1. 训练阶段
```python
# 加载训练数据
train_data = load_dataset_with_lengths(data_dir, 'train', assistance_lengths)

# 提取特征
train_features, train_labels = extract_training_features(analyzer, train_data, assistance_lengths)

# 训练预测器（RF或RNN）
predictors = train_predictors(train_features, train_labels, assistance_lengths, output_dir, model_type)
```

### 2. 测试阶段
```python
# 加载测试数据
test_data = load_dataset_with_lengths(data_dir, 'test', assistance_lengths)

# 在测试集上评估
test_accuracy, strategy_results, true_strategies, pred_strategies = evaluate_on_test_set(
    predictors, test_data, analyzer, assistance_lengths, threshold
)
```

### 3. RNN推理流程
```python
# 渐进式决策
for length in [100, 500, 1000]:
    # 提取当前长度的特征
    features = extract_features(response[:length])
    sequence_features.append(features)
    
    # RNN预测
    logits = rnn_model(sequence_features)
    probability = softmax(logits)[1]  # 使用该长度的概率
    
    if probability > threshold:
        return f'use_{length}'

return 'use_base'
```

## 输出结果

### 控制台输出示例
```
=== PROGRESSIVE ASSISTANCE PREDICTOR RESULTS ===

📊 TRAINING SET:
   Training samples: 800
   Model type: RNN/LSTM

📊 TEST SET BASELINE ACCURACIES:
   7B Only (base):         0.720 (72.0%)
   Always use 100 tokens:  0.756 (75.6%)
   Always use 500 tokens:  0.768 (76.8%)
   Always use 1000 tokens: 0.770 (77.0%)
   Theoretical optimal:    0.825 (82.5%)

🧠 PROGRESSIVE PREDICTOR PERFORMANCE:
   Test accuracy:          0.812 (81.2%)

📈 PERFORMANCE ANALYSIS:
   Improvement over base:   +0.092 (9.2 percentage points)
   Max possible improvement: +0.105 (10.5 percentage points)
   Efficiency vs optimal:   98.4%

🎉 FINAL RESULT:
   ✅ Progressive predictor OUTPERFORMS all fixed strategies!
   📊 Test accuracy: 0.812 vs best fixed: 0.770
   🔧 Average tokens used: 520 (computational efficiency!)
```

### 保存文件
- `trained_predictors.pkl` 或 `rnn_predictor.pkl`：训练好的预测器
- `test_results_summary.json`：测试结果摘要
- `best_rnn_model.pth`：RNN模型权重（仅RNN模式）

## 关键优势

### 1. 避免数据泄露
- 严格分离训练和测试数据
- 预测器训练时完全不接触测试集
- 结果具有真实的泛化能力

### 2. 模型选择灵活性
- Random Forest：快速基线，适合初步验证
- RNN：考虑序列依赖，性能更优

### 3. 渐进式效率
- 优先使用短辅助，降低计算成本
- 只在必要时使用长辅助
- 平均token使用量显著降低

### 4. 性能指标完整
- 测试集准确率
- 策略匹配准确率
- 与基线策略的详细比较
- 计算效率分析

## 实际应用

### 部署建议
```python
# 加载训练好的预测器
with open('saves/progressive_predictor_output/rnn_predictor.pkl', 'rb') as f:
    predictor = pickle.load(f)

# 渐进式推理
def progressive_inference(problem, model_32b, model_7b):
    for length in [100, 500, 1000]:
        # 生成32B辅助
        assistance = model_32b.generate(problem, max_length=length)
        
        # 提取特征
        features = extract_features(assistance)
        
        # RNN预测
        probability = predict_with_rnn(predictor, features)
        
        if probability > threshold:
            # 使用辅助生成
            return model_7b.generate_with_assistance(problem, assistance)
    
    # 使用7B直接生成
    return model_7b.generate(problem)
```

### 监控指标
- 每种策略的使用频率
- 平均辅助长度
- 准确率相对基线的提升
- 计算资源节省量

## 局限性与改进方向

### 当前局限性
1. **数据要求**：需要大量训练数据支持RNN训练
2. **特征工程**：当前基于PPL/熵，可能不是最优特征
3. **固定长度**：辅助长度固定为100/500/1000，不够灵活

### 改进方向
1. **动态长度**：支持任意长度的辅助预测
2. **更好特征**：结合语义相似度、问题类型等特征
3. **在线学习**：根据实际效果调整预测器参数
4. **多模型集成**：结合不同预测器的优势

## 总结

这个改进版的渐进式辅助预测器解决了原有的数据泄露问题，增加了RNN支持，提供了更可靠的性能评估。通过严格的训练/测试分离和先进的序列建模，能够在实际部署中提供可信的渐进式辅助决策。