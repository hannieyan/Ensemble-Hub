#!/usr/bin/env python3
"""
Test script for Ensemble-Hub batch inference functionality.
This script tests the improved batch inference with various configurations.
"""

import sys
import os
import subprocess
import logging
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def test_basic_inference():
    """Test basic inference functionality with minimal setup."""
    logger.info("Testing basic inference...")
    
    cmd = [
        sys.executable, "-m", "ensemblehub.inference",
        "--input_path", "test/example_data.json",
        "--output_path", "test/output_basic.jsonl",
        "--max_examples", "3",
        "--batch_size", "1"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            logger.info("✅ Basic inference test passed!")
            logger.info(f"Output: {result.stdout}")
        else:
            logger.error(f"❌ Basic inference test failed!")
            logger.error(f"Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.error("❌ Basic inference test timed out!")
        return False
    except Exception as e:
        logger.error(f"❌ Basic inference test error: {e}")
        return False
    
    return True

def test_batch_inference():
    """Test batch inference with larger batch size."""
    logger.info("Testing batch inference...")
    
    cmd = [
        sys.executable, "-m", "ensemblehub.inference", 
        "--input_path", "test/example_data.json",
        "--output_path", "test/output_batch.jsonl",
        "--max_examples", "5",
        "--batch_size", "2",
        "--ensemble_method", "simple"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            logger.info("✅ Batch inference test passed!")
            return True
        else:
            logger.error(f"❌ Batch inference test failed!")
            logger.error(f"Error: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"❌ Batch inference test error: {e}")
        return False

def test_generator_batch():
    """Test the generator batch functionality directly."""
    logger.info("Testing generator batch functionality...")
    
    try:
        from ensemblehub.generator import HFGenerator
        
        # Test data
        test_dicts = [
            {
                "instruction": "You are a helpful assistant.",
                "input": "What is 2+2?",
                "output": ""
            },
            {
                "instruction": "You are a helpful assistant.", 
                "input": "What is 3+3?",
                "output": ""
            }
        ]
        
        # Use a small model that's likely available
        logger.info("Creating HFGenerator with small model...")
        generator = HFGenerator("Qwen/Qwen2.5-0.5B-Instruct", device="mps")
        
        # Test single generation
        logger.info("Testing single generation...")
        single_result = generator.generate(test_dicts[0], max_tokens=50)
        logger.info(f"Single result: {single_result.text[:100]}...")
        
        # Test batch generation
        logger.info("Testing batch generation...")
        batch_results = generator.batch_generate(test_dicts, max_tokens=50)
        logger.info(f"Batch results count: {len(batch_results)}")
        
        logger.info("✅ Generator batch test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Generator batch test failed: {e}")
        return False

def test_api_scorer():
    """Test API scorer functionality (mock test)."""
    logger.info("Testing API scorer...")
    
    try:
        from ensemblehub.scorer import APIScorer
        
        # Create API scorer (will fail to connect, but we test the structure)
        scorer = APIScorer("http://localhost:8000/v1/score/test", timeout=5)
        
        # Test with dummy data (should return zeros due to connection failure)
        scores = scorer.score("Test prompt", ["completion 1", "completion 2"])
        
        if len(scores) == 2 and all(isinstance(s, (int, float)) for s in scores):
            logger.info("✅ API scorer structure test passed!")
            return True
        else:
            logger.error(f"❌ API scorer returned unexpected format: {scores}")
            return False
            
    except Exception as e:
        logger.error(f"❌ API scorer test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting Ensemble-Hub batch inference tests...")
    
    # Create test output directory
    test_dir = project_root / "test"
    test_dir.mkdir(exist_ok=True)
    
    tests = [
        ("API Scorer", test_api_scorer),
        ("Generator Batch", test_generator_batch),
        ("Basic Inference", test_basic_inference),
        ("Batch Inference", test_batch_inference),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            if test_func():
                passed += 1
            else:
                logger.error(f"Test {test_name} failed!")
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Test Results: {passed}/{total} passed")
    logger.info(f"{'='*50}")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.error(f"❌ {total - passed} test(s) failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())