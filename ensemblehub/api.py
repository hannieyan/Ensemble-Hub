"""
Ensemble-Hub API entry point

This module provides a simple interface to run the Ensemble-Hub API server.
It handles configuration loading and server initialization.
"""

import os
import sys
import ray
import uvicorn
import logging

# Add project root to Python path when running directly
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ensemblehub.api.app import create_app
from ensemblehub.ensemble_methods.ensemble import EnsembleFramework, EnsembleConfig
from ensemblehub.hparams import get_ensemble_args

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_api() -> None:
    """Run the Ensemble-Hub API server"""
    # Parse arguments using the new configuration system
    ensemble_args, method_args, generator_args = get_ensemble_args()
    
    # Build EnsembleConfig from parsed arguments
    ensemble_config = EnsembleConfig(
        model_specs=ensemble_args.model_specs,
        model_selection_method=method_args.model_selection_method,
        model_selection_params=method_args.model_selection_params,
        output_aggregation_method=method_args.output_aggregation_method,
        output_aggregation_params=method_args.output_aggregation_params,
        max_rounds=ensemble_args.max_rounds,
        stop_strings=generator_args.stop_strings,
        show_output_details=ensemble_args.show_output_details,
        show_input_details=ensemble_args.show_input_details,
        enable_thinking=generator_args.enable_thinking,
        save_results=ensemble_args.save_results
    )
    
    # Initialize Ray
    ray.init()
    
    # Create EnsembleFramework
    ensemble_framework = EnsembleFramework(ensemble_config)
    
    # Create FastAPI app
    app = create_app(ensemble_config, ensemble_framework)
    
    # Get server configuration from environment variables
    api_host = os.getenv("API_HOST", "0.0.0.0")
    api_port = int(os.getenv("API_PORT", "8000"))
    
    # Log startup information
    logger.info("Starting Ensemble-Hub API")
    logger.info(f"Visit http://localhost:{api_port}/docs for API documentation")
    logger.info(f"Configuration: {ensemble_config.model_selection_method}+{ensemble_config.output_aggregation_method}")
    logger.info(f"Models: {len(ensemble_config.model_specs)}")
    logger.info(f"Max rounds: {ensemble_config.max_rounds}")
    if ensemble_config.show_output_details:
        logger.info("Output details: Enabled")
    if ensemble_config.show_input_details:
        logger.info("Input details: Enabled")
    if ensemble_config.enable_thinking:
        logger.info("Thinking mode: Enabled")
    if ensemble_config.save_results:
        logger.info("💾 Result saving: Enabled (saves/logs/)")
    
    # Run server
    uvicorn.run(app, host=api_host, port=api_port)


def main():
    """Main entry point"""
    run_api()


if __name__ == "__main__":
    main()