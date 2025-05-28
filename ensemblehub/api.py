"""
Unified API Server for Ensemble-Hub

Features:
- Single unified endpoint `/v1/chat/completions`
- Automatic batch detection (str vs list input)
- Support for model attribution tracking
- OpenAI-compatible interface
- Flexible ensemble configuration
"""

from typing import Union, Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import time
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor

# Import ensemble framework
from ensemblehub.utils import run_ensemble, get_default_model_stats
from ensemblehub.generators import GeneratorPool
from ensemblehub.scorer import ScorerPool

logger = logging.getLogger(__name__)

# Pydantic models
class Message(BaseModel):
    role: str
    content: str

class EnsembleConfig(BaseModel):
    """Ensemble configuration parameters"""
    # Model selection
    model_selection_method: str = Field(default="all", description="Model selection: zscore, all, random")
    
    # Output aggregation  
    ensemble_method: str = Field(default="simple", description="Ensemble method: simple, progressive, random, loop")
    progressive_mode: Optional[str] = Field(default="length", description="Progressive mode: length, token")
    length_thresholds: Optional[List[int]] = Field(default=None, description="Length thresholds for progressive mode")
    special_tokens: Optional[List[str]] = Field(default=None, description="Special tokens for progressive mode")
    
    # Generation parameters
    max_rounds: int = Field(default=500, description="Maximum generation rounds")
    score_threshold: float = Field(default=-2.0, description="Score threshold for early stopping")
    
    # Attribution
    show_attribution: bool = Field(default=False, description="Include model attribution information")
    
    # Thinking mode
    enable_thinking: bool = Field(default=False, description="Enable thinking mode for models that support it")

class ChatCompletionRequest(BaseModel):
    """Unified chat completion request - handles both single and batch"""
    
    # Core OpenAI-compatible fields
    model: str = Field(default="ensemble", description="Model identifier")
    messages: Optional[Union[List[Message], List[List[Message]]]] = Field(default=None, description="Single conversation or list of conversations")
    
    # Generation parameters
    max_tokens: int = Field(default=2048, description="Maximum tokens to generate")
    temperature: float = Field(default=1.0, description="Sampling temperature")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")
    stream: bool = Field(default=False, description="Stream responses")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    
    # Ensemble configuration
    ensemble_config: Optional[EnsembleConfig] = Field(default=None, description="Ensemble configuration")
    
    # Legacy support (for backward compatibility)
    prompt: Optional[Union[str, List[str]]] = Field(default=None, description="Legacy prompt field")

class ChatCompletionChoice(BaseModel):
    index: int
    message: Optional[Message] = None  # For chat completions
    text: Optional[str] = None  # For text completions (lm-eval compatibility)
    finish_reason: str
    metadata: Optional[Dict[str, Any]] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int]

# API Application
app = FastAPI(
    title="Ensemble-Hub API",
    description="Unified ensemble inference API with automatic batch detection",
    version="3.0.0"
)

# Global configuration
class APIConfig:
    def __init__(self):
        # Debug settings
        self.show_input_details = False
        self.enable_thinking = False
        self.use_internal_template = True  # Use llamafactory templates by default
        
        # Default model specifications
        self.model_specs = [
            {"path": "Qwen/Qwen2.5-1.5B-Instruct",                "engine": "hf", "device": "cpu"},  # Larger model
            {"path": "Qwen/Qwen2.5-0.5B-Instruct",                "engine": "hf", "device": "cpu"},  # Smaller model
            # {"path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "engine": "hf",   "device": "cuda:0"},
            # {"path": "Qwen/Qwen3-4B",                             "engine": "hf",   "device": "cuda:2"},
            # {"path": "Qwen/Qwen2.5-Math-7B-Instruct",             "engine": "hf",   "device": "cuda:6"},
            # {"path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",   "engine": "hf",   "device": "cuda:1"},
            # {"path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",  "engine": "hf",   "device": "cuda:2"},
            # {"path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",  "engine": "hf",   "device": "cuda:3"},
        ]
        
        # Default reward specifications
        self.reward_spec = [
            # {"path": "Qwen/Qwen2.5-Math-PRM-7B",                  "engine": "hf_rm",  "device": "cuda:0", "weight": 0.2},
            # {"path": "http://localhost:8000/v1/score/evaluation", "engine": "api",                        "weight": 0.4},
            # {"path": "Qwen/Qwen2.5-Math-7B-Instruct",             "engine": "hf_gen", "device": "cuda:0", "weight": 1.0},
        ]
        
        # Default ensemble configuration
        self.default_ensemble_config = EnsembleConfig(
            ensemble_method="loop",  # Use round-robin for cycling
            model_selection_method="all",  # No model selection, use all models
            show_attribution=True
        )
        
        # Initialize pools
        self.generator_pool = GeneratorPool()
        self.scorer_pool = ScorerPool()
        self.model_stats = get_default_model_stats()

# Global API configuration
api_config = APIConfig()

def extract_user_content(messages: List[Message]) -> str:
    """Extract user content from messages list"""
    # Find the last user message
    user_messages = [msg for msg in messages if msg.role == "user"]
    if not user_messages:
        raise ValueError("No user message found in conversation")
    
    return user_messages[-1].content

def process_batch_conversations(
    conversations: List[List[Message]],
    ensemble_config: EnsembleConfig,
    max_tokens: int,
    temperature: float,
    stop: Optional[List[str]] = None,
    seed: Optional[int] = None,
    use_internal_template: bool = True
) -> List[Dict[str, Any]]:
    """Process multiple conversations using batch inference when possible"""
    
    # For now, we'll just process sequentially
    # In the future, this could be enhanced to use actual batch inference
    results = []
    for conversation in conversations:
        result = process_single_request(
            conversation,
            ensemble_config,
            max_tokens,
            temperature,
            stop,
            seed,
            use_internal_template
        )
        results.append(result)
    
    return results

def process_single_request(
    messages: List[Message], 
    ensemble_config: EnsembleConfig,
    max_tokens: int,
    temperature: float,
    stop: Optional[List[str]] = None,
    seed: Optional[int] = None,
    use_internal_template: bool = True
) -> Dict[str, Any]:
    """Process a single chat completion request"""
    
    logger.info("🔍 Processing single request")
    logger.info(f"  Messages count: {len(messages)}")
    for i, msg in enumerate(messages):
        logger.info(f"  Message[{i}]: role={msg.role}, content={msg.content[:100]}...")
    
    # Extract content from messages
    user_content = extract_user_content(messages)
    logger.info(f"  Extracted user content: {user_content[:200]}...")
    
    # Find system message if any
    system_messages = [msg for msg in messages if msg.role == "system"]
    
    # Create example - let HFGenerator decide the format based on content
    example = {
        "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
        "output": ""
    }
    logger.info(f"  Created example with messages format: {example}")
    
    # Prepare model specs with enable_thinking and format info
    model_specs_with_thinking = []
    for spec in api_config.model_specs:
        spec_copy = spec.copy()
        spec_copy["enable_thinking"] = api_config.enable_thinking
        spec_copy["use_internal_template"] = use_internal_template
        model_specs_with_thinking.append(spec_copy)
    
    # Prepare ensemble parameters
    ensemble_params = {
        "example": example,
        "model_specs": model_specs_with_thinking,
        "reward_spec": api_config.reward_spec,
        "ensemble_method": ensemble_config.ensemble_method,
        "model_selection_method": ensemble_config.model_selection_method,
        "max_rounds": ensemble_config.max_rounds,
        "score_threshold": ensemble_config.score_threshold,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "show_attribution": ensemble_config.show_attribution
    }
    
    # Add seed if provided
    if seed is not None:
        ensemble_params["seed"] = seed
    
    # Add stop sequences if provided
    if stop is not None:
        ensemble_params["stop_strings"] = stop
    
    # Add progressive-specific parameters
    if ensemble_config.ensemble_method == "progressive":
        ensemble_params.update({
            "progressive_mode": ensemble_config.progressive_mode,
            "length_thresholds": ensemble_config.length_thresholds or [1000, 2000, 3000],
            "special_tokens": ensemble_config.special_tokens or [r"<\think>"]
        })
    
    # Run ensemble inference
    result = run_ensemble(**ensemble_params)
    
    output = result["output"]
    
    # Log attribution information if requested
    if ensemble_config.show_attribution and "attribution" in result:
        attribution_summary = result["attribution"].get("summary", "No summary available")
        
        # Enhanced attribution logging
        selected_models = result.get("selected_models", [])
        method = result.get("method", "unknown")
        
        logger.info(f"🎯 Ensemble Method: {method}")
        logger.info(f"🔧 Available Models: {[m.split('/')[-1] for m in selected_models]}")
        logger.info(f"📝 Model Attribution: {attribution_summary}")
        
        # Also log the generated output for visibility
        if len(output) > 200:
            # Show first 100 chars ... last 100 chars
            output_preview = output[:100] + " ... " + output[-100:]
        else:
            output_preview = output
        logger.info(f"📄 Generated Output: {output_preview}")
    
    # Apply stop sequences
    finish_reason = "length"
    if stop:
        for stop_seq in stop:
            if stop_seq in output:
                output = output.split(stop_seq)[0]
                finish_reason = "stop"
                break
    
    # Check if output ends naturally
    if output.endswith(("<|im_end|>", "</s>", "<|endoftext|>")):
        finish_reason = "stop"
    
    # Create metadata
    metadata = {
        "selected_models": result.get("selected_models", []),
        "method": result.get("method", "unknown")
    }
    
    # Add attribution if available
    if "attribution" in result:
        metadata["attribution"] = result["attribution"]
    
    return {
        "content": output,
        "finish_reason": finish_reason,
        "metadata": metadata,
        "prompt_tokens": len(user_content.split()),
        "completion_tokens": len(output.split())
    }

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "name": "Ensemble-Hub API",
        "version": "3.0.0",
        "description": "Unified ensemble inference with automatic batch detection",
        "endpoint": "/v1/chat/completions",
        "features": [
            "Automatic single/batch detection",
            "Model attribution tracking", 
            "Progressive ensemble methods",
            "OpenAI-compatible interface"
        ]
    }

@app.get("/status")
def status():
    """Health check endpoint"""
    return {
        "status": "ready",
        "version": "3.0.0",
        "available_methods": {
            "model_selection": ["zscore", "all", "random"],
            "ensemble_methods": ["simple", "progressive", "random", "loop"]
        },
        "model_count": len(api_config.model_specs),
        "reward_count": len(api_config.reward_spec)
    }

@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
    """
    Unified chat completion endpoint with automatic batch detection.
    
    Automatically detects:
    - Single request: messages is List[Message] 
    - Batch request: messages is List[List[Message]]
    
    Returns OpenAI-compatible response format.
    """
    # Debug logging for input details (only if enabled)
    if api_config.show_input_details:
        logger.info("="*80)
        logger.info("Received API request:")
        logger.info(f"Model: {req.model}")
        logger.info(f"Messages: {req.messages}")
        logger.info(f"Prompt: {req.prompt}")
        logger.info(f"Temperature: {req.temperature}")
        logger.info(f"Max tokens: {req.max_tokens}")
        logger.info(f"Stop: {req.stop}")
        logger.info(f"Stream: {req.stream}")
        logger.info(f"Seed: {req.seed}")
        logger.info(f"Full request: {req}")
        logger.info("="*80)
    
    try:
        # Use provided config or default
        ensemble_config = req.ensemble_config or api_config.default_ensemble_config
        
        # Validate input: must have either prompt or messages
        if req.prompt is None and req.messages is None:
            raise HTTPException(status_code=422, detail="Either 'prompt' or 'messages' field is required")
        
        # Handle legacy prompt field (for backward compatibility)
        if req.prompt is not None:
            logger.info("🔄 Converting prompt field to messages format")
            if isinstance(req.prompt, str):
                # Single prompt
                messages_input = [Message(role="user", content=req.prompt)]
                logger.info(f"  Single prompt converted to user message: {req.prompt[:100]}...")
            else:
                # List of prompts - treat as batch
                messages_input = [[Message(role="user", content=p)] for p in req.prompt]
                logger.info(f"  Batch of {len(req.prompt)} prompts converted to messages")
        else:
            # Use messages field
            messages_input = req.messages
            logger.info("📥 Using messages field directly")
        
        # Auto-detect batch vs single request
        is_batch = False
        if messages_input and len(messages_input) > 0:
            # Check if first element is a list (batch) or Message (single)
            if isinstance(messages_input[0], list):
                # Batch request: List[List[Message]]
                is_batch = True
                conversations = messages_input
            elif isinstance(messages_input[0], Message):
                # Single request: List[Message]
                conversations = [messages_input]
            else:
                # This shouldn't happen but handle gracefully
                raise HTTPException(status_code=422, detail="Invalid message format")
        else:
            raise HTTPException(status_code=422, detail="Empty messages")
        
        logger.info(f"Processing {'batch' if is_batch else 'single'} request with {len(conversations)} conversation(s)")
        
        # Process conversations
        results = []
        
        # Helper function to process a single conversation with error handling
        def process_with_error_handling(conversation):
            try:
                return process_single_request(
                    conversation,
                    ensemble_config,
                    req.max_tokens,
                    req.temperature,
                    req.stop,
                    req.seed,
                    api_config.use_internal_template
                )
            except Exception as e:
                logger.error(f"Error processing conversation: {e}")
                if len(conversations) == 1:
                    # For single conversation, propagate the error
                    raise HTTPException(status_code=500, detail=str(e))
                else:
                    # For batch, return error result
                    return {
                        "content": f"Error: {str(e)}",
                        "finish_reason": "error",
                        "metadata": {"error": str(e)},
                        "prompt_tokens": 0,
                        "completion_tokens": 0
                    }
        
        # Try batch processing if conditions are met
        use_batch = (
            len(conversations) > 1 and
            ensemble_config.ensemble_method in ["simple", "loop"] and
            ensemble_config.model_selection_method == "all"
        )
        
        if use_batch:
            logger.info("Attempting batch processing for multiple conversations")
            try:
                results = process_batch_conversations(
                    conversations,
                    ensemble_config,
                    req.max_tokens,
                    req.temperature,
                    req.stop,
                    req.seed,
                    api_config.use_internal_template
                )
            except Exception as e:
                logger.error(f"Batch processing failed: {e}, falling back to sequential")
                # Fall back to sequential processing
                results = [process_with_error_handling(conv) for conv in conversations]
        else:
            # Sequential processing
            results = [process_with_error_handling(conv) for conv in conversations]
        
        # Build response
        choices = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        for i, result in enumerate(results):
            # Support both chat completion and text completion formats
            choice = ChatCompletionChoice(
                index=i,
                finish_reason=result["finish_reason"],
                metadata=result.get("metadata")
            )
            
            # For lm-evaluation-harness compatibility (text completions)
            if req.prompt is not None:
                choice.text = result["content"]
                choice.message = None
            else:
                # Standard chat completion format
                choice.message = Message(role="assistant", content=result["content"])
                choice.text = None

            choices.append(choice)
            total_prompt_tokens += result["prompt_tokens"]
            total_completion_tokens += result["completion_tokens"]
        
        # Determine response object type (OpenAI standard)
        object_type = "text.completion" if req.prompt is not None else "chat.completion"
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            object=object_type,
            created=int(time.time()),
            model=req.model,
            choices=choices,
            usage={
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat_completions: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/v1/completions")
def completions(req: ChatCompletionRequest):
    """
    OpenAI Completions API compatible endpoint.
    This is a separate endpoint for standard text completions.
    """
    # Debug logging for lm-eval requests
    logger.info("="*80)
    logger.info("📌 /v1/completions endpoint called (lm-eval request)")
    logger.info(f"Request prompt field: {req.prompt}")
    logger.info(f"Request type: {type(req.prompt)}")
    if isinstance(req.prompt, list):
        logger.info(f"Prompt is a list with {len(req.prompt)} items")
        for i, p in enumerate(req.prompt[:3]):  # Show first 3 prompts
            logger.info(f"Prompt[{i}]: {p[:200]}..." if len(p) > 200 else f"Prompt[{i}]: {p}")
    else:
        logger.info(f"Single prompt: {req.prompt[:200]}..." if req.prompt and len(req.prompt) > 200 else f"Single prompt: {req.prompt}")
    logger.info("="*80)
    
    # Ensure this is a text completion request
    if req.messages is not None:
        raise HTTPException(
            status_code=400,
            detail="This endpoint expects 'prompt' field, not 'messages'. Use /v1/chat/completions for chat format."
        )
    
    if req.prompt is None:
        raise HTTPException(
            status_code=400,
            detail="Missing required 'prompt' field for text completions"
        )
    
    # Process text completion request
    response = chat_completions(req)
    response.object = "text.completion"
    return response

@app.post("/v1/ensemble/config")
def update_config(
    model_specs: Optional[List[Dict[str, Any]]] = None,
    reward_spec: Optional[List[Dict[str, Any]]] = None,
    default_ensemble_config: Optional[EnsembleConfig] = None
):
    """Update API configuration"""
    if model_specs is not None:
        api_config.model_specs = model_specs
        logger.info(f"Updated model_specs: {len(model_specs)} models")
    
    if reward_spec is not None:
        api_config.reward_spec = reward_spec
        logger.info(f"Updated reward_spec: {len(reward_spec)} scorers")
    
    if default_ensemble_config is not None:
        api_config.default_ensemble_config = default_ensemble_config
        logger.info("Updated default ensemble config")
    
    return {
        "status": "updated",
        "model_count": len(api_config.model_specs),
        "reward_count": len(api_config.reward_spec)
    }

@app.get("/v1/ensemble/config")
def get_config():
    """Get current API configuration"""
    return {
        "model_specs": api_config.model_specs,
        "reward_spec": api_config.reward_spec,
        "default_ensemble_config": api_config.default_ensemble_config.model_dump()
    }


def create_app_with_config(
    model_selection_method: str = "all",
    ensemble_method: str = "simple", 
    progressive_mode: str = "length",
    length_thresholds: str = "1000,2000,3000",
    special_tokens: str = r"<\think>",
    max_rounds: int = 500,
    score_threshold: float = -2.0,
    show_attribution: bool = False,
    show_input_details: bool = False,
    enable_thinking: bool = False,
    use_internal_template: bool = True,
    model_specs: str = None,
    hf_use_8bit: bool = False,
    hf_use_4bit: bool = False
) -> FastAPI:
    """Create FastAPI app with custom ensemble configuration"""
    
    # Parse parameters
    length_threshold_list = [int(x.strip()) for x in length_thresholds.split(",")] if length_thresholds else [1000, 2000, 3000]
    special_token_list = [x.strip() for x in special_tokens.split(",")] if special_tokens else [r"<\think>"]
    
    # Update global config
    api_config.default_ensemble_config = EnsembleConfig(
        model_selection_method=model_selection_method,
        ensemble_method=ensemble_method,
        progressive_mode=progressive_mode,
        length_thresholds=length_threshold_list,
        special_tokens=special_token_list,
        max_rounds=max_rounds,
        score_threshold=score_threshold,
        show_attribution=show_attribution,
        enable_thinking=enable_thinking
    )
    
    # Update model specs if provided
    if model_specs:
        # Parse model specs from string format
        # Expected format: "model1:engine:device,model2:engine:device"
        models = []
        for spec in model_specs.split(","):
            parts = spec.strip().split(":")
            if len(parts) >= 2:
                models.append({
                    "path": parts[0],
                    "engine": parts[1],
                    "device": parts[2] if len(parts) > 2 else "cpu"
                })
        if models:
            api_config.model_specs = models
    
    # Apply quantization settings to HF models
    if hf_use_8bit or hf_use_4bit:
        quantization = "8bit" if hf_use_8bit else "4bit"
        for spec in api_config.model_specs:
            if spec.get("engine") == "hf":
                spec["quantization"] = quantization
                logger.info(f"Applying {quantization} quantization to {spec['path']}")
    
    # Set debug flag
    api_config.show_input_details = show_input_details
    api_config.enable_thinking = enable_thinking
    api_config.use_internal_template = use_internal_template
    
    logger.info(f"API initialized with:")
    logger.info(f"  Model selection: {model_selection_method}")
    logger.info(f"  Ensemble method: {ensemble_method}")
    logger.info(f"  Max rounds: {max_rounds}")
    logger.info(f"  Show attribution: {show_attribution}")
    logger.info(f"  Show input details: {show_input_details}")
    logger.info(f"  Enable thinking: {enable_thinking}")
    logger.info(f"  Use internal template: {use_internal_template}")
    logger.info(f"  Models: {len(api_config.model_specs)}")
    
    return app

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
    parser.add_argument("--ensemble_method", type=str, default="simple",
                       choices=["simple", "progressive", "random", "loop"],
                       help="Ensemble method (default: simple)")
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
    parser.add_argument("--show_attribution", action="store_true",
                       help="Show model attribution by default")
    parser.add_argument("--show_input_details", action="store_true",
                       help="Show detailed input parameters in logs")
    parser.add_argument("--enable_thinking", action="store_true",
                       help="Enable thinking mode for models that support it")
    parser.add_argument("--disable_internal_template", action="store_true",
                       help="Disable llamafactory internal templates (default: use templates)")
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
        ensemble_method=args.ensemble_method,
        progressive_mode=args.progressive_mode,
        length_thresholds=args.length_thresholds,
        special_tokens=args.special_tokens,
        max_rounds=args.max_rounds,
        score_threshold=args.score_threshold,
        show_attribution=args.show_attribution,
        show_input_details=args.show_input_details,
        enable_thinking=args.enable_thinking,
        use_internal_template=not args.disable_internal_template,
        model_specs=args.model_specs,
        hf_use_8bit=args.hf_use_8bit,
        hf_use_4bit=args.hf_use_4bit
    )
    
    uvicorn.run(app_configured, host=args.host, port=args.port)