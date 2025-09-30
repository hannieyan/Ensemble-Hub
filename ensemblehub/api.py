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
        # Generation parameters
        max_tokens=generator_args.max_tokens,
        temperature=generator_args.temperature,
        top_p=generator_args.top_p,
        top_k=generator_args.top_k,
        repetition_penalty=generator_args.repetition_penalty,
        stop_strings=generator_args.stop_strings,
        default_extract_after=generator_args.extract_after,
        seed=generator_args.seed,
        # Debug options
        show_output_details=ensemble_args.show_output_details,
        show_input_details=ensemble_args.show_input_details,
        save_results=ensemble_args.save_results
    )
    
    # Initialize Ray
    ray.init()
    
    # Create EnsembleFramework
    ensemble_framework = EnsembleFramework(ensemble_config)
    
    # Create FastAPI app
    app = create_app(ensemble_config, ensemble_framework)
    
    # Get server configuration from parsed arguments
    api_host = ensemble_args.api_host
    api_port = ensemble_args.api_port
    
    # Log startup information
    logger.info("Starting Ensemble-Hub API")
    logger.info(f"Visit http://localhost:{api_port}/docs for API documentation")
    logger.info(f"Configuration: {ensemble_config.model_selection_method}+{ensemble_config.output_aggregation_method}")
    logger.info(f"Models: {len(ensemble_config.model_specs)}")
    logger.info(f"Max rounds: {ensemble_config.max_rounds}")
    logger.info(f"Default generation params: max_tokens={ensemble_config.max_tokens}, temperature={ensemble_config.temperature}, top_p={ensemble_config.top_p}")
    logger.info(f"🖨️ Show Output: {'Enabled' if ensemble_config.show_output_details else 'Disabled'}")
    logger.info(f"🖨️ Show Input: {'Enabled' if ensemble_config.show_input_details else 'Disabled'}")
    logger.info(f"💾 Result saving: {'Enabled (saves/logs/)' if ensemble_config.save_results else 'Disabled'}")
    
    # Run server
    uvicorn.run(app, host=api_host, port=api_port)


def main():
    """Main entry point"""
    run_api()


if __name__ == "__main__":
    main()
