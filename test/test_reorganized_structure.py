#!/usr/bin/env python3
"""
Test script for the reorganized ensemble structure.
Tests the new EnsembleFramework and file organization.
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

def test_new_imports():
    """Test that all imports work with the new structure."""
    logger.info("Testing new import structure...")
    
    try:
        # Test main ensemble framework
        from ensemblehub.ensemble_methods import EnsembleFramework, EnsembleConfig
        from ensemblehub.ensemble_methods import run_simple_ensemble, run_selection_only, run_aggregation_only
        
        # Test model selection
        from ensemblehub.ensemble_methods import ZScoreSelector, AllModelsSelector, RandomSelector
        
        # Test output aggregation
        from ensemblehub.ensemble_methods import RewardBasedSelector, RandomSentenceSelector, RoundRobinSelector
        
        # Test token-level
        from ensemblehub.ensemble_methods import GaCTokenAggregator, DistributionAggregator
        
        logger.info("✅ All new imports successful")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_ensemble_framework():
    """Test the EnsembleFramework class."""
    logger.info("Testing EnsembleFramework...")
    
    try:
        from ensemblehub.ensemble_methods import EnsembleFramework, EnsembleConfig
        
        # Test configuration
        config = EnsembleConfig(
            use_model_selection=True,
            model_selection_method="all",
            use_output_aggregation=True,
            aggregation_method="reward_based",
            aggregation_level="sentence"
        )
        
        # Test framework creation
        framework = EnsembleFramework(config)
        
        assert framework.config.model_selection_method == "all"
        assert framework.config.aggregation_method == "reward_based"
        assert framework.model_selector is not None
        assert framework.output_aggregator is not None
        
        logger.info("✅ EnsembleFramework creation successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ EnsembleFramework test failed: {e}")
        return False

def test_factory_methods():
    """Test factory methods for creating ensemble configurations."""
    logger.info("Testing factory methods...")
    
    try:
        from ensemblehub.ensemble_methods import EnsembleFramework
        
        # Test simple ensemble
        simple_framework = EnsembleFramework.create_simple_ensemble("reward_based", "all")
        assert simple_framework.config.use_model_selection == True
        assert simple_framework.config.use_output_aggregation == True
        
        # Test selection only
        selection_framework = EnsembleFramework.create_selection_only("zscore", model_count=2)
        assert selection_framework.config.use_model_selection == True
        assert selection_framework.config.use_output_aggregation == False
        
        # Test aggregation only
        aggregation_framework = EnsembleFramework.create_aggregation_only("round_robin", "sentence")
        assert aggregation_framework.config.use_model_selection == False
        assert aggregation_framework.config.use_output_aggregation == True
        
        logger.info("✅ Factory methods work correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Factory methods test failed: {e}")
        return False

def test_updated_utils():
    """Test that utils.py works with the new structure."""
    logger.info("Testing updated utils.py...")
    
    try:
        from ensemblehub.utils import run_ensemble
        
        # Test data
        example = {
            "instruction": "You are a helpful assistant.",
            "input": "What is 2+2?",
            "output": ""
        }
        
        model_specs = [
            {"path": "Qwen/Qwen2.5-0.5B-Instruct", "engine": "hf", "device": "cpu"},
        ]
        
        reward_spec = []
        
        # Test with new method names
        logger.info("Testing with new method names...")
        result = run_ensemble(
            example=example,
            model_specs=model_specs,
            reward_spec=reward_spec,
            ensemble_method="reward_based",  # New name
            model_selection_method="all"
        )
        
        assert "output" in result
        assert "selected_models" in result
        
        # Test with legacy method names  
        logger.info("Testing with legacy method names...")
        result2 = run_ensemble(
            example=example,
            model_specs=model_specs,
            reward_spec=reward_spec,
            ensemble_method="simple",  # Legacy name
            model_selection_method="all"
        )
        
        assert "output" in result2
        assert "selected_models" in result2
        
        logger.info("✅ Updated utils.py works correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Utils test failed: {e}")
        return False

def test_convenience_functions():
    """Test convenience functions."""
    logger.info("Testing convenience functions...")
    
    try:
        from ensemblehub.ensemble_methods import run_simple_ensemble, run_selection_only, run_aggregation_only
        
        # These should at least be importable and callable (even if they fail due to missing models)
        assert callable(run_simple_ensemble)
        assert callable(run_selection_only)
        assert callable(run_aggregation_only)
        
        logger.info("✅ Convenience functions are importable")
        return True
        
    except Exception as e:
        logger.error(f"❌ Convenience functions test failed: {e}")
        return False

def test_file_structure():
    """Test that the file structure is correct."""
    logger.info("Testing file structure...")
    
    try:
        ensemble_methods_dir = project_root / "ensemblehub" / "ensemble_methods"
        
        # Check that directories exist
        assert (ensemble_methods_dir / "model_selection").exists()
        assert (ensemble_methods_dir / "output_aggregation").exists()
        assert (ensemble_methods_dir / "ensemble.py").exists()
        
        # Check subdirectories
        assert (ensemble_methods_dir / "output_aggregation" / "sentence_level").exists()
        assert (ensemble_methods_dir / "output_aggregation" / "token_level").exists()
        assert (ensemble_methods_dir / "output_aggregation" / "response_level").exists()
        
        logger.info("✅ File structure is correct")
        return True
        
    except Exception as e:
        logger.error(f"❌ File structure test failed: {e}")
        return False

def main():
    """Run all reorganization tests."""
    logger.info("🚀 Starting reorganized structure tests...")
    
    tests = [
        ("File Structure", test_file_structure),
        ("New Imports", test_new_imports),
        ("EnsembleFramework", test_ensemble_framework),
        ("Factory Methods", test_factory_methods),
        ("Convenience Functions", test_convenience_functions),
        ("Updated Utils", test_updated_utils),
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
                logger.info(f"✅ {test_name} passed!")
            else:
                logger.error(f"❌ {test_name} failed!")
        except Exception as e:
            logger.error(f"❌ {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Reorganization Test Results: {passed}/{total} passed")
    logger.info(f"{'='*50}")
    
    if passed == total:
        logger.info("🎉 All reorganization tests passed!")
        logger.info("\n📋 Your new structure is working correctly!")
        logger.info("✨ Features available:")
        logger.info("- EnsembleFramework for flexible ensemble configuration")
        logger.info("- Model selection and output aggregation in ensemble_methods/")
        logger.info("- Factory methods for easy setup")
        logger.info("- Backward compatibility with existing code")
        logger.info("- Easy extensibility for new methods")
        return 0
    else:
        logger.error(f"❌ {total - passed} test(s) failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())