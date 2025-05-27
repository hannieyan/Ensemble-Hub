# API Tests

This directory contains tests for the Ensemble-Hub API functionality.

## Test Files

- `test_attribution.py` - Tests model attribution tracking feature
- `test_concurrent_fix.py` - Tests concurrent request handling and thread safety
- `test_improved_system_message.py` - Tests improved default system message handling
- `test_instruction_handling.py` - Tests instruction/system message processing
- `test_list_str_input.py` - Tests various input formats (single vs batch, prompt vs messages)

## Running Tests

```bash
# Run individual test
python test/api_tests/test_attribution.py

# Make sure API is running first
python -m ensemblehub.api --show_attribution
```