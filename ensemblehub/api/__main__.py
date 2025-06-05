"""
Allow running the API server with python -m ensemblehub.api
"""

from .app import *
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    import argparse
    import uvicorn
    
    parser = argparse.ArgumentParser(description="Ensemble-Hub API Server")
    
    # Server configuration
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    
    # Ensemble configuration
    parser.add_argument("--model_selection_method", type=str, default="all", 
                       choices=["zscore", "all", "random"],
                       help="Model selection method (default: all)")
    parser.add_argument("--output_aggregation_method", type=str, default="loop",
                       choices=["reward_based", "progressive", "random", "loop"],
                       help="Ensemble method (default: loop)")
    parser.add_argument("--progressive_mode", type=str, default="length",
                       choices=["length", "token"],
                       help="Progressive mode for progressive ensemble (default: length)")
    parser.add_argument("--length_thresholds", type=str, default="1000,2000,3000",
                       help="Length thresholds for progressive mode (comma-separated, default: 1000,2000,3000)")
    parser.add_argument("--special_tokens", type=str, default=r"<\think>",
                       help=r"Special tokens for progressive mode (comma-separated, default: <\think>)")
    parser.add_argument("--max_rounds", type=int, default=500,
                       help="Maximum generation rounds (default: 500)")
    parser.add_argument("--score_threshold", type=float, default=-2.0,
                       help="Score threshold for early stopping (default: -2.0)")
    parser.add_argument("--show_output_details", action="store_true",
                       help="Show detailed output results in logs")
    parser.add_argument("--show_input_details", action="store_true",
                       help="Show raw HTTP request body in logs")
    parser.add_argument("--enable_thinking", action="store_true",
                       help="Enable thinking mode for models that support it")
    parser.add_argument("--model_specs", type=str, default=None,
                       help="Model specifications in format 'model1:engine:device,model2:engine:device'")
    
    # vLLM specific options
    parser.add_argument("--vllm_enforce_eager", action="store_true",
                       help="Disable CUDA graphs in vLLM (fixes memory allocation errors)")
    parser.add_argument("--vllm_disable_chunked_prefill", action="store_true", 
                       help="Disable chunked prefill in vLLM (fixes conflicts)")
    parser.add_argument("--vllm_max_model_len", type=int, default=32768,
                       help="Maximum model length for vLLM (default: 32768, reduces OOM)")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.8,
                       help="GPU memory utilization for vLLM (default: 0.8)")
    parser.add_argument("--vllm_disable_sliding_window", action="store_true",
                       help="Disable sliding window attention (fixes layer name conflicts)")
    
    # HuggingFace specific options
    parser.add_argument("--hf_use_eager_attention", action="store_true", default=True,
                       help="Use eager attention implementation (fixes meta tensor errors)")
    parser.add_argument("--hf_disable_device_map", action="store_true",
                       help="Disable device_map for specific device assignment (fixes meta tensor errors)")
    parser.add_argument("--hf_use_8bit", action="store_true",
                       help="Use 8-bit quantization for large models (saves GPU memory)")
    parser.add_argument("--hf_use_4bit", action="store_true",
                       help="Use 4-bit quantization for large models (saves more GPU memory)")
    parser.add_argument("--hf_low_cpu_mem", action="store_true", default=True,
                       help="Use low CPU memory loading (default: True)")
    
    args = parser.parse_args()
    
    # Create app with configuration
    app_configured = create_app_with_config(
        model_selection_method=args.model_selection_method,
        output_aggregation_method=args.output_aggregation_method,
        progressive_mode=args.progressive_mode,
        length_thresholds=args.length_thresholds,
        special_tokens=args.special_tokens,
        max_rounds=args.max_rounds,
        score_threshold=args.score_threshold,
        show_output_details=args.show_output_details,
        show_input_details=args.show_input_details,
        enable_thinking=args.enable_thinking,
        model_specs=args.model_specs,
        hf_use_8bit=args.hf_use_8bit,
        hf_use_4bit=args.hf_use_4bit
    )
    
    uvicorn.run(app_configured, host=args.host, port=args.port)