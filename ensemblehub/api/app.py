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
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import time
import uuid
import logging
import json

# Import ensemble framework
from ensemblehub.ensemble_methods.ensemble import EnsembleFramework, EnsembleConfig
from ensemblehub.utils.save_results import save_api_result

logger = logging.getLogger(__name__)

# Pydantic models
class Message(BaseModel):
    role: str
    content: str




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
    logit_bias: Optional[Dict[int, float]] = Field(default=None, description="Token bias")
    user: Optional[str] = Field(default=None, description="User identifier")
    
    # Output options
    stream: bool = Field(default=False, description="Stream responses")
    logprobs: Optional[int] = Field(default=None, description="Include log probabilities")
    
    # Additional parameters
    seed: Optional[int] = Field(default=None, description="Random seed")
    
    
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




def process_conversations(
    inputs: List,
    ensemble_framework: EnsembleFramework,
    request: ChatCompletionRequest,
    is_chat: bool = True
) -> List[Dict[str, Any]]:
    """Process conversations uniformly using ensemble"""
    ensemble_config = ensemble_framework.config
    
    # Prepare model specs with format info
    model_specs_with_format = []
    for spec in ensemble_config.model_specs:
        spec_copy = spec.copy()
        spec_copy["enable_thinking"] = spec.get("enable_thinking", ensemble_config.enable_thinking)
        model_specs_with_format.append(spec_copy)
    
    # Prepare common ensemble parameters
    base_params = {
        "model_specs": model_specs_with_format,
        "reward_spec": ensemble_config.output_aggregation_params.get('reward_specs', []),
        "output_aggregation_method": ensemble_config.output_aggregation_method,
        "model_selection_method": ensemble_config.model_selection_method,
        "max_rounds": ensemble_config.max_rounds,
        "score_threshold": ensemble_config.output_aggregation_params.get('score_threshold', -2.0),
        "show_output_details": ensemble_config.show_output_details
    }
    
    # Use request parameters if provided, otherwise use config defaults
    base_params["max_tokens"] = request.max_tokens if request.max_tokens is not None else ensemble_config.max_tokens
    base_params["temperature"] = request.temperature if request.temperature != 1.0 else ensemble_config.temperature
    base_params["top_p"] = request.top_p if request.top_p != 1.0 else ensemble_config.top_p
    
    # Debug logging
    logger.info(f"🔍 Generation params: max_tokens={base_params.get('max_tokens')}, temperature={base_params.get('temperature')}, top_p={base_params.get('top_p')}")
    logger.info(f"🔍 Config max_tokens={ensemble_config.max_tokens}, Request max_tokens={request.max_tokens}")
    
    # Add optional parameters
    if request.seed is not None:
        base_params["seed"] = request.seed
    
    # Merge stop strings from config and request
    stop_strings = list(ensemble_config.stop_strings)  # Start with config defaults
    if request.stop is not None:
        request_stops = request.stop if isinstance(request.stop, list) else [request.stop]
        # Add request stops, avoiding duplicates
        for stop in request_stops:
            if stop not in stop_strings:
                stop_strings.append(stop)
    
    if stop_strings:
        base_params["stop_strings"] = stop_strings
    
    
    # Process all examples uniformly through ensemble framework
    base_params["is_chat"] = is_chat
    
    # Debug log
    logger.info(f"🔍 Debug: show_output_details = {ensemble_config.show_output_details}")

    # Use the framework instance to run ensemble
    outputs = ensemble_framework.ensemble(examples=inputs, **base_params)
    
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

    # Calculate completion tokens from attribution data
    completion_tokens = 0
    
    if "attribution" in result and "detailed" in result["attribution"]:
        # Sum up all token counts from attribution data
        for segment in result["attribution"]["detailed"]:
            completion_tokens += segment.get("length", 0)
    
    return {
        "content": output,
        "finish_reason": finish_reason,
        "metadata": metadata,
        "completion_tokens": completion_tokens
    }

def create_app(ensemble_config: EnsembleConfig, ensemble_framework: EnsembleFramework) -> FastAPI:
    """Create FastAPI app with injected dependencies"""
    app = FastAPI(
        title="Ensemble-Hub API",
        description="Unified ensemble inference API with automatic batch detection",
        version="3.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
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
            ],
            "config": {
                "method": f"{ensemble_config.model_selection_method}+{ensemble_config.output_aggregation_method}",
                "models": len(ensemble_config.model_specs)
            }
        }

    @app.get("/status")
    def status():
        """Health check endpoint"""
        return {
            "status": "ready",
            "version": "3.0.0",
            "available_methods": {
                "model_selection": ["zscore", "all", "random", "model_judgment"],
                "output_aggregation_methods": ["progressive", "random", "loop", "reward_based"]
            },
            "model_count": len(ensemble_config.model_specs),
            "reward_count": len(ensemble_config.output_aggregation_params.get('reward_specs', []))
        }

    async def _process_request(request: Request, req: ChatCompletionRequest, is_chat: bool, endpoint_name: str) -> ChatCompletionResponse:
        """
        Common request processing logic for both chat and text completions.
        """
        # Read request body once for both debug logging and saving
        raw_body = None
        raw_json = None
        if ensemble_config.show_input_details or ensemble_config.save_results:
            raw_body = await request.body()
            if raw_body:
                raw_json = json.loads(raw_body.decode('utf-8'))
        
        # Debug logging for raw request only (if enabled)
        if ensemble_config.show_input_details and raw_json:
            logger.info("="*80)
            logger.info(f"📨 Raw request at {endpoint_name}:")
            logger.info(json.dumps(raw_json, indent=2, ensure_ascii=False))
            logger.info("="*80)

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
            ensemble_framework,
            req,
            is_chat=is_chat
        )

        # Build response
        choices = []
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
            total_completion_tokens += result["completion_tokens"]

        # Determine response object type (OpenAI standard)
        object_type = "chat.completion" if is_chat else "text.completion"

        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            object=object_type,
            created=int(time.time()),
            model=req.model,
            choices=choices,
            usage={
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_completion_tokens
            }
        )
        
        # Save results if enabled
        if ensemble_config.save_results:
            try:
                # Use already read request data, fallback to req.dict() if needed
                request_data = raw_json if raw_json else req.dict()
                
                # Save the request-response pair
                save_api_result(
                    request_data=request_data,
                    response_data=response.dict(),
                    endpoint=endpoint_name,
                    request_id=response.id
                )
            except Exception as e:
                logger.warning(f"Failed to save results: {e}")
        
        return response

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

    @app.get("/v1/ensemble/config")
    def get_config():
        """Get current ensemble configuration"""
        return {
            "model_specs": ensemble_config.model_specs,
            "reward_specs": ensemble_config.output_aggregation_params.get('reward_specs', []),
            "ensemble_config": {
                "model_selection_method": ensemble_config.model_selection_method,
                "model_selection_params": ensemble_config.model_selection_params,
                "output_aggregation_method": ensemble_config.output_aggregation_method,
                "output_aggregation_params": ensemble_config.output_aggregation_params,
                "max_rounds": ensemble_config.max_rounds,
                "score_threshold": ensemble_config.output_aggregation_params.get('score_threshold', -2.0),
                "show_output_details": ensemble_config.show_output_details,
                "enable_thinking": ensemble_config.enable_thinking
            }
        }

    
    return app

