在完成一个阶段的任务后，将代码上传到GitHub

# Ensemble-Hub 代码实现思路整理

## 1. 单模型测试支持改进 (2025-01-06)

### 问题背景
- 用户需要一个命令行脚本来测试单个模型
- 原有的ensemble.py在单模型场景下只分配256个max_tokens
- API需要正确处理text completion和chat completion的格式差异

### 核心实现改进

#### 1.1 单模型最大token长度自动检测 (ensemble.py)
```python
# 在Stage 2开始前检测单模型场景
if len(selected_specs) == 1:
    logger.info("⏭️  Skipping output aggregation, using only one model")
    
    # 根据generator类型动态设置max_tokens
    if gen_class_name == "HFGenerator" and hasattr(first_gen, 'tokenizer'):
        model_max_length = first_gen.tokenizer.model_max_length
        if model_max_length and model_max_length < 1000000:
            default_max_tokens = int(model_max_length * 0.75)  # 使用75%的最大长度
    elif gen_class_name == "VLLMGenerator":
        default_max_tokens = 32768  # vLLM默认支持较长上下文
    else:
        default_max_tokens = 16384  # 其他类型的合理默认值
```

#### 1.2 API格式自动选择 (api.py)
```python
# 添加is_chat参数来区分请求类型
def process_single_request(..., is_chat: bool = True):
    if not is_chat:
        # Text completion - 使用alpaca格式
        example = {
            "instruction": "",
            "input": user_content,
            "output": ""
        }
    else:
        # Chat completion - 使用messages格式
        example = {
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "output": ""
        }

# 在处理请求时判断
is_chat = (req.prompt is None)  # prompt字段表示text completion
```

#### 1.3 导入路径修复
- 修复了ensemble.py中的相对导入问题：
  ```python
  # 从相对导入改为绝对导入
  from ensemblehub.conversation import ConversationTemplate
  ```

### 命令行使用示例
```bash
# 基础单模型测试
python -m ensemblehub.api \
    --model_specs "Qwen/Qwen2.5-0.5B-Instruct:hf:cpu" \
    --disable_internal_template \
    --show_attribution \
    --host 0.0.0.0 \
    --port 8000

# 与lm-evaluation-harness配合使用
lm_eval --model openai-completions \
    --model_args model=ensemble,base_url=http://localhost:8000/v1/completions,tokenizer_backend=None \
    --tasks arc_challenge_chat \
    --batch_size 8
```

### 关键参数说明
- `--disable_internal_template`: 禁用内部模板处理，提高与lm-eval的兼容性
- `--show_attribution`: 显示模型归属信息，便于追踪输出来源
- `--model_specs`: 格式为 "model_path:engine:device"

## 2. 格式处理逻辑

### ShareGPT格式要求
- ShareGPT格式期望成对的user/assistant消息
- 单个user消息会导致转换错误
- 解决方案：根据请求类型（text/chat）选择不同的数据格式

### 数据流程
1. API接收请求 → 判断是text还是chat completion
2. 创建相应格式的example
3. Generator根据输入格式自动选择处理方式
4. 单模型跳过聚合步骤，直接生成输出

## 3. 架构要点
- GeneratorPool管理不同类型的生成器（HF、vLLM等）
- EnsembleFramework协调模型选择和输出聚合
- API层提供OpenAI兼容接口
- 支持批量请求和单个请求的自动检测