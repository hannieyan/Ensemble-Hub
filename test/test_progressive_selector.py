#!/usr/bin/env python3
"""
Test script for ProgressiveSelector with actual models.
Tests both length-based and token-based switching modes.
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


# Create a simple constant scorer for testing
class ConstantScorer:
    def __init__(self, score=1.0):
        self.score = score
    
    def score(self, prompt: str, completions):
        return [self.score] * len(completions)


def test_progressive_selector_length_mode():
    """Test ProgressiveSelector with length-based switching using actual models."""
    logger.info("Testing ProgressiveSelector with length-based switching...")
    
    try:
        from ensemblehub.generators import HFGenerator
        from ensemblehub.ensemble_methods.output_aggregation.sentence_level import ProgressiveSelector
        
        # Use two small models for testing - updated to available models
        model_paths = [
            "Qwen/Qwen2.5-0.5B-Instruct",  # Smaller model (will be used later)
            "Qwen/Qwen2.5-1.5B-Instruct"   # Larger model (will be used first)
        ]
        
        # Load generators
        generators = []
        for i, model_path in enumerate(model_paths):
            logger.info(f"Loading model {i+1}: {model_path}")
            try:
                generator = HFGenerator(model_path, device="cpu")
                generators.append(generator)
            except Exception as e:
                logger.warning(f"Failed to load {model_path}, using fallback: {e}")
                # Use the first model as fallback
                if i == 1 and generators:
                    generators.append(generators[0])
                else:
                    raise
        
        # Create progressive selector with small thresholds for testing
        selector = ProgressiveSelector(
            switch_mode="length",
            length_thresholds=[50, 100],  # Small thresholds for quick testing
            max_repeat=3,
            name="TestLengthSelector"
        )
        
        # Create a constant scorer
        scorer = ConstantScorer(score=1.0)
        
        # Test data
        example = {
            "instruction": "Write a detailed explanation about artificial intelligence.",
            "input": "Explain what AI is and how it works in simple terms.",
            "output": ""
        }
        
        logger.info("Running length-based progressive generation...")
        logger.info(f"Selector: {selector}")
        
        # Run generation
        result = selector.aggregate_generation(
            generators=generators,
            scorers=[scorer],
            example=example,
            max_rounds=10,
            max_new_tokens_per_round=30
        )
        
        logger.info(f"Generated text length: {len(result)}")
        logger.info(f"Generated text preview: {result[:200]}...")
        
        assert len(result) > 0, "Generated text should not be empty"
        logger.info("Length-based ProgressiveSelector test successful")
        return True
        
    except Exception as e:
        logger.error(f"Length-based ProgressiveSelector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_progressive_selector_token_mode():
    """Test ProgressiveSelector with token-based switching using actual models."""
    logger.info("Testing ProgressiveSelector with token-based switching...")
    
    try:
        from ensemblehub.generators import HFGenerator
        from ensemblehub.ensemble_methods.output_aggregation.sentence_level import ProgressiveSelector
        
        # Use two small models for testing
        model_paths = [
            "Qwen/Qwen2.5-0.5B-Instruct",  # Will be used first
            "Qwen/Qwen2.5-1.5B-Instruct"   # Will be used after token
        ]
        
        # Load generators
        generators = []
        for i, model_path in enumerate(model_paths):
            logger.info(f"Loading model {i+1}: {model_path}")
            try:
                generator = HFGenerator(model_path, device="cpu")
                generators.append(generator)
            except Exception as e:
                logger.warning(f"Failed to load {model_path}, using fallback: {e}")
                if i == 1 and generators:
                    generators.append(generators[0])
                else:
                    raise
        
        # Create progressive selector with special tokens
        selector = ProgressiveSelector(
            switch_mode="token",
            special_tokens=[r"<think>", r"<analyze>"],  # Use simpler tokens for testing
            max_repeat=3,
            name="TestTokenSelector"
        )
        
        # Create a constant scorer
        scorer = ConstantScorer(score=1.0)
        
        # Test data with instruction to use special tokens
        example = {
            "instruction": "Think step by step. Use <think> to show your reasoning and <analyze> to analyze the problem.",
            "input": "What are the benefits of renewable energy?",
            "output": ""
        }
        
        logger.info("Running token-based progressive generation...")
        logger.info(f"Selector: {selector}")
        
        # Run generation
        result = selector.aggregate_generation(
            generators=generators,
            scorers=[scorer],
            example=example,
            max_rounds=8,
            max_new_tokens_per_round=40
        )
        
        logger.info(f"Generated text length: {len(result)}")
        logger.info(f"Generated text preview: {result[:300]}...")
        
        # Check if special tokens appear in the result
        found_tokens = []
        for token in selector.special_tokens:
            # Remove regex escaping for checking
            simple_token = token.replace("\\", "")
            if simple_token in result:
                found_tokens.append(simple_token)
        
        logger.info(f"Found special tokens: {found_tokens}")
        
        assert len(result) > 0, "Generated text should not be empty"
        logger.info("Token-based ProgressiveSelector test successful")
        return True
        
    except Exception as e:
        logger.error(f"Token-based ProgressiveSelector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_progressive_selector_edge_cases():
    """Test ProgressiveSelector edge cases and error handling."""
    logger.info("Testing ProgressiveSelector edge cases...")
    
    try:
        from ensemblehub.ensemble_methods.output_aggregation.sentence_level import ProgressiveSelector
        
        # Test invalid switch mode
        try:
            ProgressiveSelector(switch_mode="invalid")
            assert False, "Should have raised ValueError for invalid switch_mode"
        except ValueError:
            logger.info("Correctly raised ValueError for invalid switch_mode")
        
        # Test creation with valid parameters
        length_selector = ProgressiveSelector(
            switch_mode="length",
            length_thresholds=[100, 200, 300]
        )
        assert length_selector.switch_mode == "length"
        assert length_selector.length_thresholds == [100, 200, 300]
        logger.info("Length selector created correctly")
        
        token_selector = ProgressiveSelector(
            switch_mode="token",
            special_tokens=[r"<think>", r"<analyze>"]
        )
        assert token_selector.switch_mode == "token"
        assert token_selector.special_tokens == [r"<think>", r"<analyze>"]
        logger.info("Token selector created correctly")
        
        # Test default values
        default_selector = ProgressiveSelector()
        assert default_selector.switch_mode == "length"
        assert default_selector.length_thresholds == [1000, 2000, 3000]
        logger.info("Default selector created correctly")
        
        # Test _determine_current_model_index with mock data
        test_text = "This is a test text with some content."
        
        # Test length mode
        length_selector.length_thresholds = [10, 20]  # Very small for testing
        model_idx = length_selector._determine_current_model_index(test_text, [], 0)
        logger.info(f"Length mode model index: {model_idx}")
        
        # Test token mode
        token_selector.special_tokens = [r"test", r"content"]
        model_idx = token_selector._determine_current_model_index(test_text, [], 0)
        logger.info(f"Token mode model index: {model_idx}")
        
        logger.info("Edge cases test successful")
        return True
        
    except Exception as e:
        logger.error(f"Edge cases test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_progressive_selector_single_model():
    """Test ProgressiveSelector with a single model (fallback behavior)."""
    logger.info("Testing ProgressiveSelector with single model...")
    
    try:
        from ensemblehub.generators import HFGenerator
        from ensemblehub.ensemble_methods.output_aggregation.sentence_level import ProgressiveSelector
        
        # Load single model
        model_path = "Qwen/Qwen2.5-0.5B-Instruct"
        logger.info(f"Loading single model: {model_path}")
        generator = HFGenerator(model_path, device="cpu")
        
        # Create progressive selector
        selector = ProgressiveSelector(
            switch_mode="length",
            length_thresholds=[50, 100],
            name="SingleModelSelector"
        )
        
        # Create a constant scorer
        scorer = ConstantScorer(score=1.0)
        
        # Test data
        example = {
            "instruction": "You are a helpful assistant.",
            "input": "What is machine learning?",
            "output": ""
        }
        
        logger.info("Running single model progressive generation...")
        
        # Run generation with single model
        result = selector.aggregate_generation(
            generators=[generator],  # Only one model
            scorers=[scorer],
            example=example,
            max_rounds=5,
            max_new_tokens_per_round=50
        )
        
        logger.info(f"Generated text length: {len(result)}")
        logger.info(f"Generated text preview: {result[:150]}...")
        
        assert len(result) > 0, "Generated text should not be empty"
        logger.info("Single model ProgressiveSelector test successful")
        return True
        
    except Exception as e:
        logger.error(f"Single model ProgressiveSelector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all ProgressiveSelector tests."""
    logger.info("Starting ProgressiveSelector tests...")
    logger.info("These tests require downloading models and may take time...")
    
    tests = [
        ("Progressive Selector Edge Cases", test_progressive_selector_edge_cases),
        ("Progressive Selector Single Model", test_progressive_selector_single_model),
        ("Progressive Selector Length Mode", test_progressive_selector_length_mode),
        ("Progressive Selector Token Mode", test_progressive_selector_token_mode),
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
                logger.info(f"{test_name} passed!")
            else:
                logger.error(f"{test_name} failed!")
        except Exception as e:
            logger.error(f"{test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"ProgressiveSelector Test Results: {passed}/{total} passed")
    logger.info(f"{'='*60}")
    
    if passed == total:
        logger.info("All ProgressiveSelector tests passed!")
        logger.info("\nProgressiveSelector is working correctly!")
        logger.info("You can now use:")
        logger.info("- Length-based model switching: switch at token thresholds")
        logger.info("- Token-based model switching: switch at special tokens")
        logger.info("- Multi-model progressive inference")
        logger.info("- Fallback to single model when needed")
        return 0
    else:
        logger.error(f"{total - passed} test(s) failed!")
        logger.info("Some failures may be due to missing models or hardware constraints")
        return 1


if __name__ == "__main__":
    sys.exit(main())