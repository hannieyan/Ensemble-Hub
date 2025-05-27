# LM-Evaluation-Harness Tests

This directory contains tests and utilities for lm-evaluation-harness compatibility.

## Test Files

- `test_lmeval_compat.py` - Basic compatibility tests
- `test_lmeval_compatibility.py` - Comprehensive compatibility test suite
- `test_lmeval_format.py` - Tests for lm-eval request format analysis
- `capture_lmeval_request.py` - Utility to capture lm-eval requests
- `monitor_lmeval_requests.py` - Monitor and log lm-eval requests

## Log Files

- `lmeval_request_captured.json` - Captured request example
- `lmeval_requests.log` - Request logs
- `lm_eval_output.log` - Output logs

## Usage

```bash
# Capture lm-eval requests
python test/lmeval_tests/capture_lmeval_request.py

# Run compatibility tests
python test/lmeval_tests/test_lmeval_compatibility.py
```