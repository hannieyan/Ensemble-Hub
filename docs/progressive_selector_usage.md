# ProgressiveSelector Usage Guide

ProgressiveSelector is a new inference method that supports dynamically switching models during generation. It has two modes:

## 1. Length-based Model Switching

Switch models based on cumulative token count, suitable for letting larger models handle initial reasoning and smaller models handle subsequent generation.

```python
from ensemblehub.ensemble_methods.output_aggregation.sentence_level import ProgressiveSelector
from ensemblehub.generator import HFGenerator

# Create generators (ordered from high to low capability)
generators = [
    HFGenerator("Qwen/Qwen2.5-1.5B-Instruct", device="cpu"),  # Larger model used first
    HFGenerator("Qwen/Qwen2.5-0.5B-Instruct", device="cpu")   # Smaller model used later
]

# Create length-based switching selector
selector = ProgressiveSelector(
    switch_mode="length",
    length_thresholds=[1000, 2000, 3000],  # Switch at these token counts
    name="LengthBasedSelector"
)

# Usage:
# - First 1000 tokens use model 0 (Qwen2.5-1.5B)
# - Tokens 1000-2000 use model 1 (Qwen2.5-0.5B) 
# - Tokens 2000-3000 use model 2 (if available)
# - Beyond 3000 tokens, stick with the last model
```

## 2. Special Token-based Model Switching

Switch models based on specific tokens, suitable for structured reasoning.

```python
# Create token-based switching selector
selector = ProgressiveSelector(
    switch_mode="token",
    special_tokens=[r"<think>", r"<analyze>", r"<conclude>"],
    name="TokenBasedSelector"
)

# Usage:
# - Model 0 reasons until first <think>
# - Model 1 reasons until first <analyze> 
# - Model 2 reasons until first <conclude>
# - After that, stick with the last model for remaining inference
```

## 3. Simple Single Switch

```python
# Switch only once when encountering <think>
selector = ProgressiveSelector(
    switch_mode="token",
    special_tokens=[r"<think>"],  # Only one token
    name="SingleSwitchSelector"
)

# Usage:
# - Model 0 reasons until <think>
# - Model 1 continues with remaining inference
```

## Complete Usage Example

```python
from ensemblehub.ensemble_methods.output_aggregation.sentence_level import ProgressiveSelector
from ensemblehub.generator import HFGenerator

# Simple scorer for testing
class ConstantScorer:
    def __init__(self, score=1.0):
        self.score = score
    
    def score(self, prompt, completions):
        return [self.score] * len(completions)

# Create models (ordered by capability in descending order)
generators = [
    HFGenerator("Qwen/Qwen2.5-1.5B-Instruct", device="cpu"),  # More capable model
    HFGenerator("Qwen/Qwen2.5-0.5B-Instruct", device="cpu")   # Smaller model
]

# Create selector
selector = ProgressiveSelector(
    switch_mode="length",
    length_thresholds=[500]  # First 500 tokens use large model, then small model
)

# Create scorer
scorer = ConstantScorer(1.0)

# Test data
example = {
    "instruction": "Explain the principles and applications of artificial intelligence in detail",
    "input": "Please start from basic concepts and go deeper step by step.",
    "output": ""
}

# Run progressive generation
result = selector.aggregate_generation(
    generators=generators,
    scorers=[scorer],
    example=example,
    max_rounds=10,
    max_new_tokens_per_round=100
)

print(f"Generated result: {result}")
```

## Configuration Recommendations

### Length-based Switching Mode
- **Multi-model progression**: `[500, 1500, 3000]` - For complex tasks, large model handles beginning, gradually transitions to smaller models
- **Dual-model switching**: `[1000]` - For general tasks, large model for beginning, small model for continuation
- **Fine-grained control**: `[200, 400, 600]` - For scenarios requiring frequent switching

### Token-based Switching Mode
- **Chain of thought**: `[r"<\think>"]` - Large model reasons until </think>, then small model continues
- **Multi-stage analysis**: `[r"<\think>", r"<\analyze>", r"<\conclude>"]` - For complex analysis tasks
- **Natural language markers**: `[r"then", r"next"]` - Use natural language as switching points

## Important Notes

1. **Model ordering**: Models should be ordered from high to low capability, as typically you want strong models for initial complex reasoning
2. **Threshold setting**: Length thresholds should be set reasonably based on task complexity and model capabilities
3. **Token selection**: Special tokens should be ensured to be generatable by models, recommend explicit instruction in prompts
4. **Performance balance**: Too frequent switching may affect generation fluency, need to balance accuracy and efficiency

## Practical Application Scenarios

1. **Long document generation**: Large model generates beginning and structure, small model fills in details
2. **Code generation**: Large model designs architecture and logic, small model implements specific code
3. **Creative writing**: Large model conceives plot, small model perfects descriptions
4. **Academic writing**: Large model organizes arguments, small model polishes expression

## Command Line Usage

You can now use ProgressiveSelector from the command line:

```bash
# Length-based switching example
python -m ensemblehub.inference \
    --input_path data/test.json \
    --output_path output/progressive_length.jsonl \
    --ensemble_method progressive \
    --progressive_mode length \
    --length_thresholds "500,1500" \
    --max_rounds 10

# Token-based switching example  
python -m ensemblehub.inference \
    --input_path data/test.json \
    --output_path output/progressive_token.jsonl \
    --ensemble_method progressive \
    --progressive_mode token \
    --special_tokens "<\\think>,<\\analyze>" \
    --max_rounds 8
```