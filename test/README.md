# Ensemble-Hub Testing

This directory contains test files and examples for the Ensemble-Hub batch inference functionality.

## Files

- `example_data.json` - Sample test data with 5 math problems
- `run_test.py` - Comprehensive test script for all functionality
- `README.md` - This documentation file

## Running Tests

### Quick Test
```bash
cd /path/to/Ensemble-Hub
python test/run_test.py
```

### Manual Testing

#### Basic Inference Test
```bash
python -m ensemblehub.inference \
  --input_path test/example_data.json \
  --output_path test/output_basic.jsonl \
  --max_examples 3 \
  --batch_size 1
```

#### Batch Inference Test
```bash
python -m ensemblehub.inference \
  --input_path test/example_data.json \
  --output_path test/output_batch.jsonl \
  --max_examples 5 \
  --batch_size 2 \
  --ensemble_method simple
```

#### With Different Ensemble Methods
```bash
# Simple ensemble (default)
python -m ensemblehub.inference \
  --input_path test/example_data.json \
  --output_path test/output_simple.jsonl \
  --ensemble_method simple

# Random ensemble
python -m ensemblehub.inference \
  --input_path test/example_data.json \
  --output_path test/output_random.jsonl \
  --ensemble_method random

# Loop ensemble  
python -m ensemblehub.inference \
  --input_path test/example_data.json \
  --output_path test/output_loop.jsonl \
  --ensemble_method loop
```

## What Gets Tested

1. **API Scorer** - Tests the API scorer structure and error handling
2. **Generator Batch** - Tests both single and batch generation with HFGenerator
3. **Basic Inference** - Tests the main inference pipeline with minimal configuration
4. **Batch Inference** - Tests batch processing with larger batch sizes

## Expected Outputs

Test outputs will be saved in the `test/` directory:
- `output_basic.jsonl` - Results from basic inference test
- `output_batch.jsonl` - Results from batch inference test  
- `output_simple.jsonl`, `output_random.jsonl`, `output_loop.jsonl` - Results from different ensemble methods

Each output file contains JSONL format with:
```json
{
  "prompt": "<｜User｜>instruction\nquestion<｜Assistant｜>",
  "predict": "model_generated_answer",
  "label": "expected_answer", 
  "selected_models": ["list", "of", "selected", "model", "paths"]
}
```

## Notes

- Tests use `Qwen/Qwen2.5-0.5B-Instruct` as the default model (small and fast)
- Most models and reward models are commented out in the test configuration
- API tests will show connection failures (expected when no API server is running)
- Batch functionality is tested at both the generator level and inference pipeline level

## Troubleshooting

If tests fail:

1. **Model Loading Issues**: Ensure you have sufficient memory and the model is available
2. **Device Issues**: Tests default to CPU/MPS, adjust device settings in `ensemblehub/inference.py` if needed
3. **Memory Issues**: Reduce batch size or use smaller models
4. **Network Issues**: API scorer tests expect connection failures in test environment

For production use, configure models and devices appropriately in the main inference script.