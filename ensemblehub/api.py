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
from ensemblehub.generator import GeneratorPool
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

class ChatCompletionRequest(BaseModel):
    """Unified chat completion request - handles both single and batch"""
    
    # Core OpenAI-compatible fields
    model: str = Field(default="ensemble", description="Model identifier")
    messages: Union[List[Message], List[List[Message]]] = Field(..., description="Single conversation or list of conversations")
    
    # Generation parameters
    max_tokens: int = Field(default=256, description="Maximum tokens to generate")
    temperature: float = Field(default=1.0, description="Sampling temperature")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")
    
    # Ensemble configuration
    ensemble_config: Optional[EnsembleConfig] = Field(default=None, description="Ensemble configuration")
    
    # Legacy support (for backward compatibility)
    prompt: Optional[Union[str, List[str]]] = Field(default=None, description="Legacy prompt field")

class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
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
        # Default model specifications
        self.model_specs = [
            {"path": "Qwen/Qwen2.5-1.5B-Instruct", "engine": "hf", "device": "cpu"},
            {"path": "Qwen/Qwen2.5-0.5B-Instruct", "engine": "hf", "device": "cpu"},
        ]
        
        # Default reward specifications
        self.reward_spec = [
            # Add your reward models here
        ]
        
        # Default ensemble configuration
        self.default_ensemble_config = EnsembleConfig()
        
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

def process_single_request(
    messages: List[Message], 
    ensemble_config: EnsembleConfig,
    max_tokens: int,
    temperature: float,
    stop: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Process a single chat completion request"""
    
    # Extract content from messages
    user_content = extract_user_content(messages)
    
    # Find system message if any
    system_messages = [msg for msg in messages if msg.role == "system"]
    instruction = system_messages[0].content if system_messages else "You are a helpful assistant."
    
    # Create example for ensemble
    example = {
        "instruction": instruction,
        "input": user_content,
        "output": ""
    }
    
    # Prepare ensemble parameters
    ensemble_params = {
        "example": example,
        "model_specs": api_config.model_specs,
        "reward_spec": api_config.reward_spec,
        "ensemble_method": ensemble_config.ensemble_method,
        "model_selection_method": ensemble_config.model_selection_method,
        "max_rounds": ensemble_config.max_rounds,
        "score_threshold": ensemble_config.score_threshold,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "show_attribution": ensemble_config.show_attribution
    }
    
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
    try:
        # Use provided config or default
        ensemble_config = req.ensemble_config or api_config.default_ensemble_config
        
        # Handle legacy prompt field (for backward compatibility)
        if req.prompt is not None:
            if isinstance(req.prompt, str):
                # Single prompt
                messages_input = [[Message(role="user", content=req.prompt)]]
            else:
                # List of prompts
                messages_input = [[Message(role="user", content=p)] for p in req.prompt]
        else:
            # Use messages field
            messages_input = req.messages
        
        # Auto-detect batch vs single request
        is_batch = False
        if messages_input and isinstance(messages_input[0], list):
            # Batch request: List[List[Message]]
            is_batch = True
            conversations = messages_input
        else:
            # Single request: List[Message]
            conversations = [messages_input]
        
        logger.info(f"Processing {'batch' if is_batch else 'single'} request with {len(conversations)} conversation(s)")
        
        # Process conversations
        if len(conversations) == 1:
            # Single conversation - process directly
            try:
                result = process_single_request(
                    conversations[0], 
                    ensemble_config,
                    req.max_tokens,
                    req.temperature,
                    req.stop
                )
                results = [result]
            except Exception as e:
                logger.error(f"Error processing single request: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        else:
            # Multiple conversations - process in parallel
            def process_conversation(conv_with_index):
                idx, conversation = conv_with_index
                try:
                    return idx, process_single_request(
                        conversation,
                        ensemble_config, 
                        req.max_tokens,
                        req.temperature,
                        req.stop
                    )
                except Exception as e:
                    logger.error(f"Error processing conversation {idx}: {e}")
                    return idx, {
                        "content": f"Error: {str(e)}",
                        "finish_reason": "error",
                        "metadata": {"error": str(e)},
                        "prompt_tokens": 0,
                        "completion_tokens": 0
                    }
            
            with ThreadPoolExecutor(max_workers=min(len(conversations), 4)) as executor:
                indexed_results = list(executor.map(process_conversation, enumerate(conversations)))
            
            # Sort results by original index
            indexed_results.sort(key=lambda x: x[0])
            results = [result for _, result in indexed_results]
        
        # Build response
        choices = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        for i, result in enumerate(results):
            choices.append(ChatCompletionChoice(
                index=i,
                message=Message(role="assistant", content=result["content"]),
                finish_reason=result["finish_reason"],
                metadata=result.get("metadata")
            ))
            total_prompt_tokens += result["prompt_tokens"]
            total_completion_tokens += result["completion_tokens"]
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            object="chat.completion",
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)