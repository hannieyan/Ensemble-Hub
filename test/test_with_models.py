#!/usr/bin/env python3
"""
Test script that actually loads models and tests the full pipeline.
This requires models to be available and may take some time.
"""

import sys
import os
import json
import logging
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def test_single_model_generation():
    """Test generation with a single small model."""
    logger.info("Testing single model generation...")
    
    try:
        from ensemblehub.generators import HFGenerator
        from ensemblehub.conversation import ConversationTemplate
        
        # Use the smallest available model
        model_path = "Qwen/Qwen2.5-0.5B-Instruct"
        device = "cpu"  # Use CPU for reliability
        
        logger.info(f"Loading model: {model_path}")
        generator = HFGenerator(model_path, device=device)
        
        # Test data
        example = {
            "instruction": "You are a helpful assistant.",
            "input": "What is 2+2?",
            "output": ""
        }
        
        # Generate response
        logger.info("Generating response...")
        result = generator.generate(example, max_tokens=50)
        
        logger.info(f"Generated: {result.text}")
        assert len(result.text) > 0, "Generated text should not be empty"
        
        logger.info("✅ Single model generation successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Single model generation failed: {e}")
        return False

def test_batch_generation():
    """Test batch generation with a small model."""
    logger.info("Testing batch generation...")
    
    try:
        from ensemblehub.generators import HFGenerator
        
        model_path = "Qwen/Qwen2.5-0.5B-Instruct"
        device = "cpu"
        
        logger.info(f"Loading model: {model_path}")
        generator = HFGenerator(model_path, device=device)
        
        # Test data
        examples = [
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
        
        # Batch generate
        logger.info("Batch generating responses...")
        results = generator.batch_generate(examples, max_tokens=50)
        
        assert len(results) == len(examples), f"Expected {len(examples)} results, got {len(results)}"
        
        for i, result in enumerate(results):
            logger.info(f"Result {i+1}: {result.text[:100]}...")
            assert len(result.text) > 0, f"Result {i+1} should not be empty"
        
        logger.info("✅ Batch generation successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Batch generation failed: {e}")
        return False

def test_zscore_selection():
    """Test Z-score model selection with actual models."""
    logger.info("Testing Z-score model selection...")
    
    try:
        from ensemblehub.utils import run_ensemble
        
        # Test data
        example = {
            "instruction": "Solve this math problem:",
            "input": "What is 15 × 23?",
            "output": ""
        }
        
        # Use smaller models that are more likely to be available
        model_specs = [
            {"path": "Qwen/Qwen2.5-0.5B-Instruct", "engine": "hf", "device": "cpu"},
        ]
        
        reward_spec = []  # No external reward models
        
        logger.info("Running ensemble with Z-score selection...")
        result = run_ensemble(
            example=example,
            model_specs=model_specs,
            reward_spec=reward_spec,
            ensemble_method="simple",
            model_selection_method="zscore",
            max_rounds=2
        )
        
        logger.info(f"Ensemble result keys: {list(result.keys())}")
        logger.info(f"Selected models: {result.get('selected_models', [])}")
        logger.info(f"Output preview: {str(result.get('output', ''))[:100]}...")
        
        assert "output" in result, "Result should contain output"
        assert "selected_models" in result, "Result should contain selected_models"
        
        logger.info("✅ Z-score selection successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Z-score selection failed: {e}")
        return False

def test_simple_ensemble():
    """Test simple ensemble method."""
    logger.info("Testing simple ensemble method...")
    
    try:
        from ensemblehub.utils import run_ensemble
        
        example = {
            "instruction": "You are a helpful assistant.",
            "input": "What is the capital of France?",
            "output": ""
        }
        
        model_specs = [
            {"path": "Qwen/Qwen2.5-0.5B-Instruct", "engine": "hf", "device": "cpu"},
        ]
        
        reward_spec = []
        
        logger.info("Running simple ensemble...")
        result = run_ensemble(
            example=example,
            model_specs=model_specs,
            reward_spec=reward_spec,
            ensemble_method="simple",
            model_selection_method="all",  # Use all models
            max_rounds=3
        )
        
        logger.info(f"Simple ensemble output: {str(result.get('output', ''))[:100]}...")
        
        assert "output" in result
        assert len(str(result["output"])) > 0
        
        logger.info("✅ Simple ensemble successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Simple ensemble failed: {e}")
        return False

def test_progressive_selector():
    """Test ProgressiveSelector with actual models."""
    logger.info("Testing ProgressiveSelector...")
    
    try:
        from ensemblehub.generators import HFGenerator
        from ensemblehub.ensemble_methods.output_aggregation.sentence_level import ProgressiveSelector
        
        # Create a simple constant scorer for testing
        class ConstantScorer:
            def __init__(self, score=1.0):
                self.score = score
            
            def score(self, prompt: str, completions):
                return [self.score] * len(completions)
        
        # Use suggested models
        model_paths = [
            "Qwen/Qwen2.5-0.5B-Instruct",  # Smaller model
            "Qwen/Qwen2.5-1.5B-Instruct"   # Larger model
        ]
        
        # Load generators
        generators = []
        for i, model_path in enumerate(model_paths):
            logger.info(f"Loading model {i+1}: {model_path}")
            try:
                generator = HFGenerator(model_path, device="cpu")
                generators.append(generator)
            except Exception as e:
                logger.warning(f"Failed to load {model_path}: {e}")
                # Use single model as fallback
                if i == 1 and generators:
                    logger.info("Using first model as fallback for second model")
                    generators.append(generators[0])
                elif i == 0:
                    raise
        
        # Test length-based selector
        length_selector = ProgressiveSelector(
            switch_mode="length",
            length_thresholds=[30, 60],  # Small thresholds for testing
            name="TestLengthSelector"
        )
        
        scorer = ConstantScorer(score=1.0)
        
        example = {
            "instruction": "You are a helpful assistant.",
            "input": "Explain artificial intelligence in detail.",
            "output": ""
        }
        
        logger.info("Testing length-based progressive selection...")
        result = length_selector.aggregate_generation(
            generators=generators,
            scorers=[scorer],
            example=example,
            max_rounds=5,
            max_new_tokens_per_round=25
        )
        
        logger.info(f"Length-based result preview: {result[:100]}...")
        assert len(result) > 0, "Generated text should not be empty"
        
        # Test token-based selector
        token_selector = ProgressiveSelector(
            switch_mode="token",
            special_tokens=[r"think"],  # Simple token for testing
            name="TestTokenSelector"
        )
        
        example2 = {
            "instruction": "Think step by step and use the word 'think' in your response.",
            "input": "What are the benefits of renewable energy?",
            "output": ""
        }
        
        logger.info("Testing token-based progressive selection...")
        result2 = token_selector.aggregate_generation(
            generators=generators,
            scorers=[scorer],
            example=example2,
            max_rounds=4,
            max_new_tokens_per_round=30
        )
        
        logger.info(f"Token-based result preview: {result2[:100]}...")
        assert len(result2) > 0, "Generated text should not be empty"
        
        logger.info("ProgressiveSelector test successful")
        return True
        
    except Exception as e:
        logger.error(f"ProgressiveSelector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_legacy_compatibility():
    """Test that legacy run_zscore_ensemble still works."""
    logger.info("Testing legacy compatibility...")
    
    try:
        from ensemblehub.utils import run_zscore_ensemble
        from ensemblehub.statistics.compute_model_stats import ModelStatStore
        
        # Create mock stat store
        class MockStatStore:
            def get_all_stats(self):
                return {}
        
        example = {
            "instruction": "You are a helpful assistant.",
            "input": "What is 1+1?",
            "output": ""
        }
        
        model_specs = [
            {"path": "Qwen/Qwen2.5-0.5B-Instruct", "engine": "hf", "device": "cpu"},
        ]
        
        reward_spec = []
        stat_store = MockStatStore()
        
        logger.info("Running legacy zscore ensemble...")
        result = run_zscore_ensemble(
            example=example,
            dataset_problems=[],  # Not used
            model_specs=model_specs,
            reward_spec=reward_spec,
            stat_store=stat_store,
            ensemble_method="simple",
            max_rounds=2
        )
        
        assert "output" in result
        logger.info("✅ Legacy compatibility successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Legacy compatibility failed: {e}")
        return False

def main():
    """Run all model tests."""
    logger.info("🚀 Starting Ensemble-Hub model tests...")
    logger.info("⚠️  These tests require downloading models and may take time...")
    
    tests = [
        ("Single Model Generation", test_single_model_generation),
        ("Batch Generation", test_batch_generation),
        ("Simple Ensemble", test_simple_ensemble),
        ("Z-score Selection", test_zscore_selection),
        ("Progressive Selector", test_progressive_selector),
        ("Legacy Compatibility", test_legacy_compatibility),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} passed!")
            else:
                logger.error(f"❌ {test_name} failed!")
        except Exception as e:
            logger.error(f"❌ {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Model Test Results: {passed}/{total} passed")
    logger.info(f"{'='*60}")
    
    if passed == total:
        logger.info("🎉 All model tests passed!")
        logger.info("\n📋 Your new architecture is working correctly!")
        logger.info("You can now use:")
        logger.info("- run_ensemble() for the new modular approach")
        logger.info("- run_zscore_ensemble() for backward compatibility")
        logger.info("- Enhanced API scorers with retry logic")
        logger.info("- Both simple selection and sentence-level aggregation")
        return 0
    else:
        logger.error(f"❌ {total - passed} test(s) failed!")
        logger.info("💡 Some failures may be due to missing models or hardware constraints")
        return 1

if __name__ == "__main__":
    sys.exit(main())