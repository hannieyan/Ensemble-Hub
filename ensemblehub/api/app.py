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
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import time
import uuid
import logging
import json

# Import ensemble framework
from ensemblehub.ensemble_methods.ensemble import run_ensemble, get_default_model_stats
from ensemblehub.generators import GeneratorPool
from ensemblehub.scorers.base import ScorerPool

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
    output_aggregation_method: str = Field(default="loop", description="Output aggregation method: progressive, random, loop, reward_based")
    progressive_mode: Optional[str] = Field(default="length", description="Progressive mode: length, token")
    length_thresholds: Optional[List[int]] = Field(default=None, description="Length thresholds for progressive mode")
    special_tokens: Optional[List[str]] = Field(default=None, description="Special tokens for progressive mode")
    
    # Generation parameters
    max_rounds: int = Field(default=500, description="Maximum generation rounds")
    score_threshold: float = Field(default=-2.0, description="Score threshold for early stopping")
    
    # Output details
    show_output_details: bool = Field(default=False, description="Show detailed output results in logs")
    
    # Thinking mode
    enable_thinking: bool = Field(default=False, description="Enable thinking mode for models that support it")

class ChatCompletionRequest(BaseModel):
    """Unified request model supporting both OpenAI completion formats"""
    
    # Core fields
    model: str = Field(default="ensemble", description="Model identifier")
    
    # Chat completion fields (v1/chat/completions)
    messages: Optional[Union[List[Message], List[List[Message]]]] = Field(default=None, description="Messages for chat format")
    
    # Text completion fields (v1/completions) 
    prompt: Optional[Union[str, List[str]]] = Field(default=None, description="Prompt for completion format")
    suffix: Optional[str] = Field(default=None, description="Suffix after completion")
    echo: bool = Field(default=False, description="Echo prompt in response")
    best_of: Optional[int] = Field(default=1, description="Generate n completions, return best")
    
    # Common generation parameters
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    temperature: float = Field(default=1.0, description="Sampling temperature")
    top_p: float = Field(default=1.0, description="Top-p sampling")
    n: int = Field(default=1, description="Number of completions")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="Stop sequences")
    presence_penalty: float = Field(default=0.0, description="Presence penalty")
    frequency_penalty: float = Field(default=0.0, description="Frequency penalty")
    logit_bias: Optional[Dict[int, float]] = Field(default=None, description="Token bias")
    user: Optional[str] = Field(default=None, description="User identifier")
    
    # Output options
    stream: bool = Field(default=False, description="Stream responses")
    logprobs: Optional[int] = Field(default=None, description="Include log probabilities")
    
    # Additional parameters
    seed: Optional[int] = Field(default=None, description="Random seed")
    
    # Custom ensemble configuration
    ensemble_config: Optional[EnsembleConfig] = Field(default=None, description="Ensemble configuration")
    
    class Config:
        extra = "allow"  # Allow additional fields not defined here

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
        
        # Default model specifications
        self.model_specs = [
            {"path": "Qwen/Qwen2.5-1.5B-Instruct",                "engine": "hf", "device": "cpu"},  # Larger model
            {"path": "Qwen/Qwen2.5-0.5B-Instruct",                "engine": "hf", "device": "cpu"},  # Smaller model
            # {"path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "engine": "hf",   "device": "cuda:0"},
            # {"path": "Qwen/Qwen3-4B",                             "engine": "hf",   "device": "cuda:0"},
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
            output_aggregation_method="loop",  # Use round-robin for cycling
            model_selection_method="all",  # No model selection, use all models
            show_output_details=False
        )
        
        # Initialize pools
        self.generator_pool = GeneratorPool()
        self.scorer_pool = ScorerPool()
        self.model_stats = get_default_model_stats()

# Global API configuration
api_config = APIConfig()


def process_conversations(
    inputs: List,
    ensemble_config: EnsembleConfig,
    request: ChatCompletionRequest,
    is_chat: bool = True
) -> List[Dict[str, Any]]:
    """Process conversations uniformly using ensemble"""
    
    # Prepare model specs with format info
    model_specs_with_format = []
    for spec in api_config.model_specs:
        spec_copy = spec.copy()
        spec_copy["enable_thinking"] = api_config.enable_thinking
        model_specs_with_format.append(spec_copy)
    
    # Prepare common ensemble parameters
    base_params = {
        "model_specs": model_specs_with_format,
        "reward_spec": api_config.reward_spec,
        "output_aggregation_method": ensemble_config.output_aggregation_method,
        "model_selection_method": ensemble_config.model_selection_method,
        "max_rounds": ensemble_config.max_rounds,
        "score_threshold": ensemble_config.score_threshold,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "show_output_details": ensemble_config.show_output_details
    }
    
    # Add optional parameters
    if request.seed is not None:
        base_params["seed"] = request.seed
    
    if request.stop is not None:
        base_params["stop_strings"] = request.stop if isinstance(request.stop, list) else [request.stop]
    
    if request.presence_penalty != 0:
        base_params["presence_penalty"] = request.presence_penalty

    if request.frequency_penalty != 0:
        base_params["frequency_penalty"] = request.frequency_penalty
    
    # Add progressive-specific parameters
    if ensemble_config.output_aggregation_method == "progressive":
        base_params.update({
            "progressive_mode": ensemble_config.progressive_mode,
            "length_thresholds": ensemble_config.length_thresholds or [1000, 2000, 3000],
            "special_tokens": ensemble_config.special_tokens or [r"<\think>"]
        })
    
    # Process all examples uniformly through ensemble framework
    base_params["is_chat"] = is_chat
    
    # Debug log
    logger.info(f"🔍 Debug: show_output_details = {ensemble_config.show_output_details}")

    outputs = run_ensemble(examples=inputs, **base_params)
    
    # Process results
    results = []
    for i, (example, result) in enumerate(zip(inputs, outputs)):
        output = result["output"]

        # Log output details if requested
        if ensemble_config.show_output_details:
            selected_models = result.get("selected_models", [])
            method = result.get("method", "unknown")
            logger.info(f"📋 Example {i+1}: Method={method}, Model={[m.split('/')[-1] for m in selected_models]}")
            logger.info(f"💬 Generated Output: {output}")

        result_dict = create_result_dict(output, result, example, request, is_chat)
        results.append(result_dict)
    
    return results


def create_result_dict(output: str, result: Dict, example: Union[str, List[Dict]], request: ChatCompletionRequest, is_chat: bool) -> Dict[str, Any]:
    """Create a standardized result dictionary"""

    # Apply stop sequences
    finish_reason = "length"
    if request.stop:
        stop_list = request.stop if isinstance(request.stop, list) else [request.stop]
        for stop_seq in stop_list:
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

    # Calculate tokens
    if not is_chat:
        # In completion mode, example is the prompt string directly
        prompt_tokens = len(example.split())
    else:
        # In chat mode, example is a list of message dicts
        prompt_text = " ".join([msg.get("content", "") for msg in example])
        prompt_tokens = len(prompt_text.split())

    return {
        "content": output,
        "finish_reason": finish_reason,
        "metadata": metadata,
        "prompt_tokens": prompt_tokens,
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
            "output_aggregation_methods": ["progressive", "random", "loop", "reward_based"]
        },
        "model_count": len(api_config.model_specs),
        "reward_count": len(api_config.reward_spec)
    }

async def _process_request(request: Request, req: ChatCompletionRequest, is_chat: bool, endpoint_name: str) -> ChatCompletionResponse:
    """
    Common request processing logic for both chat and text completions.
    """
    # Debug logging for raw request only (if enabled)
    if api_config.show_input_details:
        # Print raw request body
        try:
            raw_body = await request.body()
            if raw_body:
                logger.info("="*80)
                logger.info(f"📨 Raw request at {endpoint_name}:")
                try:
                    # Try to parse and pretty print JSON
                    raw_json = json.loads(raw_body.decode('utf-8'))
                    logger.info(json.dumps(raw_json, indent=2, ensure_ascii=False))
                except:
                    # If not valid JSON, print as string
                    logger.info(raw_body.decode('utf-8', errors='ignore'))
                logger.info("="*80)
        except Exception as e:
            logger.info(f"Could not read raw body: {e}")

    # Use provided config or default
    ensemble_config = req.ensemble_config or api_config.default_ensemble_config

    # Validate input based on endpoint type
    if is_chat:
        # Chat completions: expect messages field
        assert req.messages is not None, "'messages' field is required for chat completions"
        inputs = [req.messages,] if isinstance(req.messages[0], dict) else req.messages
    else:
        # Text completions: expect prompt field
        assert req.prompt is not None, "'prompt' field is required for text completions"
        inputs = [req.prompt,] if isinstance(req.prompt, str) else req.prompt

    # Process all conversations
    results = process_conversations(
        inputs,
        ensemble_config,
        req,
        is_chat=is_chat
    )

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

        if is_chat:
            # Standard chat completion format
            choice.message = Message(role="assistant", content=result["content"])
            choice.text = None
        else:
            # Text completion format (lm-evaluation-harness compatibility)
            choice.text = result["content"]
            choice.message = None

        choices.append(choice)
        total_prompt_tokens += result["prompt_tokens"]
        total_completion_tokens += result["completion_tokens"]

    # Determine response object type (OpenAI standard)
    object_type = "chat.completion" if is_chat else "text.completion"

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

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, req: ChatCompletionRequest) -> ChatCompletionResponse:
    """
    Chat completion endpoint for conversational AI.
    
    Automatically detects:
    - Single request: messages is List[Message] 
    - Batch request: messages is List[List[Message]]
    
    Returns OpenAI-compatible response format.
    """
    return await _process_request(request, req, is_chat=True, endpoint_name="/v1/chat/completions")

@app.post("/v1/completions")
async def completions(request: Request, req: ChatCompletionRequest):
    """
    Text completion endpoint for prompt completion.
    Compatible with OpenAI Completions API.
    """
    return await _process_request(request, req, is_chat=False, endpoint_name="/v1/completions")

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
    output_aggregation_method: str = "loop",
    progressive_mode: str = "length",
    length_thresholds: str = "1000,2000,3000",
    special_tokens: str = r"<\think>",
    max_rounds: int = 500,
    score_threshold: float = -2.0,
    show_output_details: bool = False,
    show_input_details: bool = False,
    enable_thinking: bool = False,
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
        output_aggregation_method=output_aggregation_method,
        progressive_mode=progressive_mode,
        length_thresholds=length_threshold_list,
        special_tokens=special_token_list,
        max_rounds=max_rounds,
        score_threshold=score_threshold,
        show_output_details=show_output_details,
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
    
    logger.info(f"API initialized with:")
    logger.info(f"  Model selection: {model_selection_method}")
    logger.info(f"  Output aggregation method: {output_aggregation_method}")
    logger.info(f"  Max rounds: {max_rounds}")
    logger.info(f"  Show output details: {show_output_details}")
    logger.info(f"  Show input details: {show_input_details}")
    logger.info(f"  Enable thinking: {enable_thinking}")
    logger.info(f"  Models: {len(api_config.model_specs)}")
    
    return app
