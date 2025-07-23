# Complete Token Analysis Script

## 概述

`complete_token_analysis.py` 是一个综合的token级别分析工具，用于比较base和enhanced模型的回答质量，识别improvements和regressions，并进行深度的token-level分析。

## 主要功能

### 1. 自动化评估流程
- **自动评估**：调用 `eval_acc_lm_eval.py` 自动生成评估结果
- **问题匹配**：基于问题内容匹配base和enhanced模型的回答
- **改进识别**：自动识别improvements（enhanced正确，base错误）和regressions（base正确，enhanced错误）

### 2. Token级别分析
- **深度学习模型**：使用 `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` 进行token分析
- **Perplexity计算**：计算每个token位置的困惑度
- **Entropy计算**：计算每个token位置的信息熵，支持两种计算方法：
  - `full`：使用全部概率分布（默认）
  - `top-k`：只使用top-1000概率，避免数值问题

### 3. 灵活的输入处理
- **完整上下文分析**：分析回答的前100个token，但使用问题作为上下文来提供更准确的困惑度计算
- **仅回答分析**：只分析回答部分，不使用问题上下文
- **自定义token数量**：支持分析回答的前N个token（默认100，可设置如1000）

**重要**：当启用问题上下文时（`--include-problem`），100个token限制仅适用于回答部分，问题部分用作上下文但不计入token数量。

### 4. 全面的可视化
- **对比分析图**：生成improvements和regressions的scatter plots和bar charts
- **位置趋势图**：显示PPL和Entropy随token位置的变化趋势
- **四种情况对比**：
  - Improvements中的base vs enhanced
  - Regressions中的base vs enhanced
  - 分别展示PPL和Entropy变化

## 使用方法

### 基本用法
```bash
# 使用默认参数（enhanced.jsonl，100个token）
python scripts/complete_token_analysis.py

# 分析1000个token
python scripts/complete_token_analysis.py --n-tokens 1000

# 使用不同的enhanced文件
python scripts/complete_token_analysis.py --enhanced saves/enhanced1.jsonl
```

### 高级选项
```bash
# 只分析回答部分，不包含问题
python scripts/complete_token_analysis.py --response-only

# 使用top-k entropy计算方法
python scripts/complete_token_analysis.py --entropy-method top-k

# 限制样本数量进行快速测试
python scripts/complete_token_analysis.py --sample-size 20

# 只保存样本文件，跳过token分析
python scripts/complete_token_analysis.py --skip-analysis
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--base` | `saves/base.jsonl` | Base模型结果文件路径 |
| `--enhanced` | `saves/enhanced.jsonl` | Enhanced模型结果文件路径 |
| `--output-dir` | `saves/token_analysis_output` | 输出目录 |
| `--model` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 用于token分析的模型 |
| `--n-tokens` | `100` | 分析的token数量 |
| `--sample-size` | 无限制 | 限制分析的样本数量 |
| `--include-problem` | `True` | 是否在分析中包含问题文本 |
| `--response-only` | `False` | 只分析回答，排除问题 |
| `--entropy-method` | `full` | Entropy计算方法：`full` 或 `top-k` |
| `--skip-analysis` | `False` | 跳过token分析，只保存样本 |

## 输出文件

### 1. 样本文件
- `improvements_simple.jsonl`：改进案例（enhanced正确，base错误）
- `regressions_simple.jsonl`：退化案例（base正确，enhanced错误）

### 2. 可视化文件
- `token_analysis_comparison.png`：主要对比图（scatter plots和bar charts）
- `token_metrics_distributions.png`：token位置趋势图
  - 上排：Improvement cases的PPL和Entropy趋势
  - 下排：Regression cases的PPL和Entropy趋势

### 3. 分析结果
- `analysis_results.json`：完整的数值分析结果

## 技术特点

### 1. 路径处理
- **绝对路径**：所有路径基于脚本位置计算，避免工作目录依赖
- **跨平台兼容**：支持本地和服务器环境

### 2. 数值稳定性
- **高精度计算**：使用float64进行entropy计算
- **异常处理**：自动处理NaN和Inf值
- **概率归一化**：确保概率分布和为1

### 3. 内存优化
- **临时文件管理**：自动清理评估过程中的临时文件
- **批量处理**：支持大规模数据集的处理
- **模型量化**：支持8-bit量化以节省显存

## 输出示例

### 统计信息
```
=== Summary Statistics ===

Improvement Cases (n=58):
  Base avg PPL: 98.33
  Enhanced avg PPL: 104.70
  PPL Change: 6.37
  Base avg Entropy: 3.45
  Enhanced avg Entropy: 3.12
  Entropy Change: -0.33

Regression Cases (n=28):
  Base avg PPL: 95.55
  Enhanced avg PPL: 104.85
  PPL Change: 9.30
  Base avg Entropy: 3.78
  Enhanced avg Entropy: 3.45
  Entropy Change: -0.33
```

### 分析模式
- **Problem + Response模式**（推荐）：分析完整输入上下文
- **Response only模式**：只分析模型回答部分

## 应用场景

1. **模型对比**：比较不同模型版本的性能差异
2. **改进分析**：识别模型改进的具体案例
3. **错误分析**：分析模型退化的原因
4. **Token级别诊断**：理解模型在不同位置的确定性
5. **研究分析**：深入理解模型行为差异

## 注意事项

1. **模型加载**：首次运行需要下载DeepSeek模型，需要足够的存储空间和网络
2. **计算资源**：Token分析需要GPU加速，CPU模式会很慢
3. **内存需求**：处理大量样本时需要足够的内存
4. **时间成本**：完整分析可能需要较长时间，建议使用`--sample-size`进行快速测试

## 故障排除

1. **Entropy全为0**：尝试使用`--entropy-method top-k`
2. **路径错误**：确保在项目根目录运行脚本
3. **内存不足**：使用`--sample-size`限制样本数量
4. **模型加载失败**：检查网络连接和存储空间

## 更新日志

- **v1.0**：基础功能实现
- **v1.1**：添加自动评估功能
- **v1.2**：修复路径问题，支持服务器环境
- **v1.3**：优化entropy计算，添加调试功能
- **v1.4**：代码精简，生产环境优化
- **v1.5**：修复token数量限制问题，支持任意token数量分析