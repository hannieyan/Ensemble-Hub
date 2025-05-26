#!/usr/bin/env python3
"""
Test script for API structure validation (no server required).
Tests imports, model validation, and endpoint structure.
"""

import sys
import logging
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def test_api_imports():
    """Test that API imports work correctly"""
    logger.info("Testing API imports...")
    
    try:
        from ensemblehub.api import app, APIConfig, EnsembleMethodConfig
        from ensemblehub.api import ChatCompletionRequest, EnsembleRequest, BatchRequest
        
        logger.info("✅ All API imports successful")
        return True
        
    except ImportError as e:
        logger.error(f"❌ API import failed: {e}")
        return False

def test_pydantic_models():
    """Test Pydantic model validation"""
    logger.info("Testing Pydantic models...")
    
    try:
        from ensemblehub.api import EnsembleMethodConfig, ChatCompletionRequest, EnsembleRequest
        
        # Test EnsembleMethodConfig
        config = EnsembleMethodConfig(
            model_selection_method="zscore",
            aggregation_method="reward_based",
            aggregation_level="sentence"
        )
        assert config.model_selection_method == "zscore"
        assert config.use_model_selection == True
        
        # Test ChatCompletionRequest
        chat_req = ChatCompletionRequest(
            model="test",
            prompt="Hello",
            max_tokens=100,
            ensemble_config=config
        )
        assert chat_req.prompt == "Hello"
        assert chat_req.ensemble_config.aggregation_method == "reward_based"
        
        # Test EnsembleRequest
        ensemble_req = EnsembleRequest(
            input="Test input",
            ensemble_config=config
        )
        assert ensemble_req.input == "Test input"
        
        logger.info("✅ Pydantic models work correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pydantic model test failed: {e}")
        return False

def test_api_config():
    """Test API configuration class"""
    logger.info("Testing API configuration...")
    
    try:
        from ensemblehub.api import APIConfig
        
        config = APIConfig()
        
        # Check that config has required attributes
        assert hasattr(config, 'model_specs')
        assert hasattr(config, 'reward_spec')
        assert hasattr(config, 'default_ensemble_config')
        assert hasattr(config, 'generator_pool')
        assert hasattr(config, 'scorer_pool')
        assert hasattr(config, 'model_stats')
        
        # Check model_specs format
        assert isinstance(config.model_specs, list)
        if config.model_specs:
            assert 'path' in config.model_specs[0]
            assert 'engine' in config.model_specs[0]
        
        logger.info("✅ API configuration works")
        return True
        
    except Exception as e:
        logger.error(f"❌ API configuration test failed: {e}")
        return False

def test_fastapi_app():
    """Test FastAPI app structure"""
    logger.info("Testing FastAPI app structure...")
    
    try:
        from ensemblehub.api import app
        
        # Check that app is a FastAPI instance
        assert hasattr(app, 'routes')
        assert hasattr(app, 'title')
        
        # Check app metadata
        assert app.title == "Ensemble-Hub API"
        assert app.version == "2.0.0"
        
        # Check that routes exist
        route_paths = [route.path for route in app.routes]
        
        expected_routes = [
            "/",
            "/status", 
            "/v1/ensemble/methods",
            "/v1/ensemble/config",
            "/v1/chat/completions",
            "/v1/ensemble/inference",
            "/v1/ensemble/batch",
            "/v1/ensemble/presets/simple",
            "/v1/ensemble/presets/selection_only",
            "/v1/ensemble/presets/aggregation_only"
        ]
        
        for expected_route in expected_routes:
            assert expected_route in route_paths, f"Route {expected_route} not found"
        
        logger.info("✅ FastAPI app structure is correct")
        return True
        
    except Exception as e:
        logger.error(f"❌ FastAPI app test failed: {e}")
        return False

def test_ensemble_integration():
    """Test integration with ensemble framework"""
    logger.info("Testing ensemble framework integration...")
    
    try:
        from ensemblehub.api import EnsembleMethodConfig
        from ensemblehub.ensemble_methods import EnsembleConfig, EnsembleFramework
        
        # Test conversion from API config to ensemble config
        api_config = EnsembleMethodConfig(
            model_selection_method="all",
            aggregation_method="reward_based",
            aggregation_level="sentence",
            use_model_selection=True,
            use_output_aggregation=True
        )
        
        # Convert to ensemble config
        ensemble_config = EnsembleConfig(
            use_model_selection=api_config.use_model_selection,
            model_selection_method=api_config.model_selection_method,
            use_output_aggregation=api_config.use_output_aggregation,
            aggregation_method=api_config.aggregation_method,
            aggregation_level=api_config.aggregation_level
        )
        
        # Create framework
        framework = EnsembleFramework(ensemble_config)
        
        assert framework.config.model_selection_method == "all"
        assert framework.config.aggregation_method == "reward_based"
        
        logger.info("✅ Ensemble framework integration works")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ensemble integration test failed: {e}")
        return False

def test_endpoint_functions():
    """Test that endpoint functions are callable"""
    logger.info("Testing endpoint functions...")
    
    try:
        from ensemblehub.api import (
            root, status, list_methods, get_config,
            chat_completions, ensemble_inference,
            simple_ensemble, selection_only, aggregation_only
        )
        
        # Test that functions are callable
        assert callable(root)
        assert callable(status) 
        assert callable(list_methods)
        assert callable(get_config)
        assert callable(chat_completions)
        assert callable(ensemble_inference)
        assert callable(simple_ensemble)
        assert callable(selection_only)
        assert callable(aggregation_only)
        
        # Test functions that don't require requests
        root_result = root()
        assert "name" in root_result
        assert "Ensemble-Hub API" in root_result["name"]
        
        status_result = status()
        assert "status" in status_result
        assert status_result["status"] == "ready"
        
        methods_result = list_methods()
        assert "model_selection_methods" in methods_result
        assert "output_aggregation_methods" in methods_result
        
        config_result = get_config()
        assert "model_specs" in config_result
        assert "default_ensemble_config" in config_result
        
        logger.info("✅ Endpoint functions work correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Endpoint functions test failed: {e}")
        return False

def main():
    """Run all API structure tests"""
    logger.info("🚀 Starting API structure validation tests...")
    logger.info("ℹ️  These tests validate API structure without running a server")
    
    tests = [
        ("API Imports", test_api_imports),
        ("Pydantic Models", test_pydantic_models),
        ("API Configuration", test_api_config),
        ("FastAPI App Structure", test_fastapi_app),
        ("Ensemble Integration", test_ensemble_integration),
        ("Endpoint Functions", test_endpoint_functions),
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
    logger.info(f"API Structure Test Results: {passed}/{total} passed")
    logger.info(f"{'='*50}")
    
    if passed == total:
        logger.info("🎉 All API structure tests passed!")
        logger.info("\n📋 Enhanced API Features:")
        logger.info("🔧 Flexible ensemble method configuration")
        logger.info("📝 Comprehensive Pydantic models for validation")
        logger.info("🎯 Multiple endpoint types (chat, direct, batch, presets)")
        logger.info("⚙️ Runtime configuration management")
        logger.info("🔄 Legacy compatibility with original API")
        logger.info("📊 Rich response metadata")
        logger.info("\n🚀 Ready to start API server:")
        logger.info("   python ensemblehub/api.py")
        return 0
    else:
        logger.error(f"❌ {total - passed} test(s) failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())