# Ensemble-Hub Command Line Usage Guide

This guide covers all command-line options for running ensemble inference with Ensemble-Hub.

## Basic Command Structure

```bash
python -m ensemblehub.inference [OPTIONS]
```

## Required Arguments

- `--input_path`: Path to input JSON file containing examples
- `--output_path`: Path to save output JSONL file (default: ensemble-generated-predictions.jsonl)

## Model Selection Options

### Use All Models (Skip Model Selection)
To use all specified models without any selection:

```bash
python -m ensemblehub.inference \
    --input_path test/example_data.json \
    --output_path output/all_models.jsonl \
    --ensemble_method progressive \
    --progressive_mode length \
    --length_thresholds "500,1000"
```

**Note**: The current default uses z-score selection. To use all models, modify the code to set `model_selection_method="all"` in inference.py line 97.

### Z-Score Model Selection (Default)
The system currently defaults to z-score selection, which may filter out models without statistics:

```bash
# Current default behavior
python -m ensemblehub.inference \
    --input_path test/example_data.json \
    --output_path output/zscore_selected.jsonl \
    --ensemble_method progressive
```

## Ensemble Methods

### 1. Progressive Selector

#### Length-based Switching
Switch models at specific token thresholds:

```bash
python -m ensemblehub.inference \
    --input_path test/example_data.json \
    --output_path output/progressive_length.jsonl \
    --ensemble_method progressive \
    --progressive_mode length \
    --length_thresholds "500,1000,2000" \
    --max_rounds 10
```

**Parameters:**
- `--progressive_mode length`: Enable length-based switching
- `--length_thresholds`: Comma-separated token counts for switching
- Example: "500,1000,2000" means:
  - 0-500 tokens: Use model 1 (Qwen2.5-1.5B)
  - 500-1000 tokens: Use model 2 (Qwen2.5-0.5B)
  - 1000+ tokens: Use model 2 (continues with last model)

#### Token-based Switching
Switch models when encountering special tokens:

```bash
python -m ensemblehub.inference \
    --input_path test/example_data.json \
    --output_path output/progressive_token.jsonl \
    --ensemble_method progressive \
    --progressive_mode token \
    --special_tokens "<\think>,<\analyze>" \
    --max_rounds 8
```

**Parameters:**
- `--progressive_mode token`: Enable token-based switching
- `--special_tokens`: Comma-separated special tokens
- Example: "<\think>,<\analyze>" means:
  - Model 1 until first `<\think>`
  - Model 2 until first `<\analyze>`
  - Model 2 continues afterward

#### Single Token Switching
Simple two-model switching:

```bash
python -m ensemblehub.inference \
    --input_path test/example_data.json \
    --output_path output/progressive_simple.jsonl \
    --ensemble_method progressive \
    --progressive_mode token \
    --special_tokens "<\think>" \
    --max_rounds 6
```

### 2. Other Ensemble Methods

#### Simple Ensemble
```bash
python -m ensemblehub.inference \
    --input_path test/example_data.json \
    --output_path output/simple.jsonl \
    --ensemble_method simple \
    --max_rounds 10
```

#### Random Selector
```bash
python -m ensemblehub.inference \
    --input_path test/example_data.json \
    --output_path output/random.jsonl \
    --ensemble_method random \
    --max_rounds 8
```

#### Round Robin (Loop)
```bash
python -m ensemblehub.inference \
    --input_path test/example_data.json \
    --output_path output/round_robin.jsonl \
    --ensemble_method loop \
    --max_rounds 12
```

## General Options

### Processing Control
- `--max_examples N`: Limit processing to N examples
- `--batch_size N`: Process N examples at a time (default: 1)
- `--max_rounds N`: Maximum generation rounds (default: 500)
- `--score_threshold F`: Score threshold for early stopping (default: -2.0)

### Example with Processing Control
```bash
python -m ensemblehub.inference \
    --input_path test/example_data.json \
    --output_path output/limited.jsonl \
    --ensemble_method progressive \
    --progressive_mode length \
    --length_thresholds "300" \
    --max_examples 3 \
    --max_rounds 5
```

## Model Configuration

The system currently uses these models (configured in inference.py):
- **Qwen/Qwen2.5-1.5B-Instruct**: Larger model (used first in progressive)
- **Qwen/Qwen2.5-0.5B-Instruct**: Smaller model (used later in progressive)

## Input File Format

Input JSON should contain examples with these fields:
```json
[
    {
        "instruction": "You are a helpful assistant.",
        "input": "What is 2+2?",
        "output": "4"
    }
]
```

## Output File Format

Output JSONL contains results with these fields:
```json
{"prompt": "...", "predict": "...", "label": "...", "selected_models": ["..."]}
```

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure model paths are correct and models are available
2. **Z-score selection filtering models**: Models without statistics may be skipped
3. **Memory issues**: Use smaller `--max_rounds` or `--batch_size`

### Debug Tips

1. **Use smaller examples**: Test with `--max_examples 1` first
2. **Check logs**: Look for model loading and selection messages
3. **Verify input format**: Ensure JSON structure matches expected format

## Advanced Usage

### Custom Thresholds for Different Tasks

**Short responses** (100-300 tokens):
```bash
--progressive_mode length --length_thresholds "50,150"
```

**Medium responses** (300-1000 tokens):
```bash
--progressive_mode length --length_thresholds "200,600"
```

**Long responses** (1000+ tokens):
```bash
--progressive_mode length --length_thresholds "500,1500,3000"
```

### Token-based for Structured Reasoning

**Chain of thought**:
```bash
--progressive_mode token --special_tokens "<\think>"
```

**Multi-stage analysis**:
```bash
--progressive_mode token --special_tokens "<\think>,<\analyze>,<\conclude>"
```

## Performance Recommendations

1. **For speed**: Use fewer models, smaller `--max_rounds`
2. **For quality**: Use progressive with appropriate thresholds
3. **For consistency**: Use all models with simple ensemble
4. **For efficiency**: Use length-based progressive with optimal thresholds

## Example Workflows

### Quick Test
```bash
python -m ensemblehub.inference \
    --input_path test/example_data.json \
    --output_path output/quick_test.jsonl \
    --ensemble_method progressive \
    --progressive_mode length \
    --length_thresholds "200" \
    --max_examples 2 \
    --max_rounds 3
```

### Production Run
```bash
python -m ensemblehub.inference \
    --input_path data/production_data.json \
    --output_path output/production_results.jsonl \
    --ensemble_method progressive \
    --progressive_mode token \
    --special_tokens "<\think>" \
    --max_rounds 15 \
    --batch_size 1
```