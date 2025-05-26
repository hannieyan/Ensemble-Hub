"""
Enhanced API Server for Ensemble-Hub

Supports flexible ensemble configuration including:
- Model selection methods (zscore, all, random, etc.)
- Output aggregation methods (reward_based, random, round_robin, etc.)
- Custom ensemble configurations
- Multiple endpoint formats for compatibility
"""

from typing import Union, Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import time
import uuid
import logging

# Import new ensemble framework
from ensemblehub.ensemble_methods import EnsembleFramework, EnsembleConfig
from ensemblehub.utils import run_ensemble, get_default_model_stats
from ensemblehub.generator import GeneratorPool
from ensemblehub.scorer import ScorerPool

logger = logging.getLogger(__name__)

# Pydantic models for API requests
class Message(BaseModel):
    role: str
    content: str

class EnsembleMethodConfig(BaseModel):
    """Configuration for ensemble methods"""
    model_selection_method: str = Field(default="all", description="Model selection method: zscore, all, random, llm_blender")
    model_selection_params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for model selection")
    
    aggregation_method: str = Field(default="reward_based", description="Output aggregation method: reward_based, random, round_robin")
    aggregation_level: str = Field(default="sentence", description="Aggregation level: sentence, token, response") 
    aggregation_params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for output aggregation")
    
    use_model_selection: bool = Field(default=True, description="Whether to use model selection")
    use_output_aggregation: bool = Field(default=True, description="Whether to use output aggregation")

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="ensemble", description="Model identifier")
    prompt: Union[str, List[str]] = Field(..., description="Input prompt(s)")
    max_tokens: int = Field(default=256, description="Maximum tokens to generate")
    temperature: float = Field(default=1.0, description="Sampling temperature")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")
    
    # Ensemble configuration
    ensemble_config: Optional[EnsembleMethodConfig] = Field(default=None, description="Ensemble method configuration")
    
    # Legacy support
    ensemble_method: Optional[str] = Field(default=None, description="Legacy: ensemble method (simple, random, loop)")
    model_selection_method: Optional[str] = Field(default=None, description="Legacy: model selection method")

class EnsembleRequest(BaseModel):
    """Direct ensemble inference request"""
    instruction: str = Field(default="", description="System instruction")
    input: str = Field(..., description="User input")
    output: str = Field(default="", description="Expected output (for training)")
    
    ensemble_config: EnsembleMethodConfig = Field(default_factory=EnsembleMethodConfig, description="Ensemble configuration")
    
    max_rounds: int = Field(default=500, description="Maximum generation rounds")
    score_threshold: float = Field(default=-2.0, description="Score threshold for early stopping")
    max_tokens: int = Field(default=256, description="Maximum tokens per generation")

class BatchRequest(BaseModel):
    """Batch inference request"""
    examples: List[EnsembleRequest] = Field(..., description="List of examples to process")
    batch_size: int = Field(default=1, description="Batch size for processing")

# API Application
app = FastAPI(
    title="Ensemble-Hub API",
    description="Enhanced ensemble inference API with flexible method selection",
    version="2.0.0"
)

# Global configuration
class APIConfig:
    def __init__(self):
        # Default model specifications
        self.model_specs = [
            {"path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "engine": "hf", "device": "mps"},
            # {"path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "engine": "hf", "device": "cpu"},
            # {"path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "engine": "hf", "device": "cpu"},
        ]
        
        # Default reward specifications  
        self.reward_spec = [
            # Add your reward models here
            # {"path": "Qwen/Qwen2.5-Math-PRM-7B", "engine": "hf_rm", "device": "cuda:0", "weight": 0.5},
            # {"path": "http://localhost:8000/v1/score/evaluation", "engine": "api", "weight": 0.3},
        ]
        
        # Default ensemble configuration
        self.default_ensemble_config = EnsembleMethodConfig(
            model_selection_method="all",
            aggregation_method="reward_based",
            aggregation_level="sentence",
            use_model_selection=True,
            use_output_aggregation=True
        )
        
        # Initialize pools
        self.generator_pool = GeneratorPool()
        self.scorer_pool = ScorerPool()
        self.model_stats = get_default_model_stats()

# Global API configuration
api_config = APIConfig()

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "name": "Ensemble-Hub API",
        "version": "2.0.0",
        "description": "Enhanced ensemble inference with flexible method selection",
        "endpoints": {
            "/status": "Health check",
            "/v1/chat/completions": "Chat completion (OpenAI compatible)",
            "/v1/ensemble/inference": "Direct ensemble inference",
            "/v1/ensemble/batch": "Batch ensemble inference",
            "/v1/ensemble/methods": "List available methods",
            "/v1/ensemble/config": "Get/set ensemble configuration"
        }
    }

@app.get("/status")
def status():
    """Health check endpoint"""
    return {
        "status": "ready",
        "available_methods": {
            "model_selection": ["zscore", "all", "random"],
            "output_aggregation": ["reward_based", "random", "round_robin"],
            "aggregation_levels": ["sentence", "token", "response"]
        },
        "model_count": len(api_config.model_specs),
        "reward_count": len(api_config.reward_spec)
    }

@app.get("/v1/ensemble/methods")
def list_methods():
    """List available ensemble methods"""
    return {
        "model_selection_methods": {
            "zscore": "Z-score based model selection using perplexity and confidence",
            "all": "Use all available models",
            "random": "Randomly select subset of models",
            "llm_blender": "LLM-Blender model selection (if implemented)"
        },
        "output_aggregation_methods": {
            "reward_based": "Select outputs based on reward model scores",
            "random": "Randomly select from generated outputs",
            "round_robin": "Round-robin selection from models"
        },
        "aggregation_levels": {
            "sentence": "Aggregate at sentence/segment level during generation", 
            "token": "Aggregate at token level (e.g., GaC)",
            "response": "Aggregate complete responses (e.g., voting)"
        }
    }

@app.get("/v1/ensemble/config")
def get_config():
    """Get current ensemble configuration"""
    return {
        "model_specs": api_config.model_specs,
        "reward_spec": api_config.reward_spec,
        "default_ensemble_config": api_config.default_ensemble_config.dict()
    }

@app.post("/v1/ensemble/config")
def update_config(
    model_specs: Optional[List[Dict[str, Any]]] = None,
    reward_spec: Optional[List[Dict[str, Any]]] = None,
    default_ensemble_config: Optional[EnsembleMethodConfig] = None
):
    """Update ensemble configuration"""
    if model_specs is not None:
        api_config.model_specs = model_specs
        logger.info(f"Updated model_specs: {len(model_specs)} models")
    
    if reward_spec is not None:
        api_config.reward_spec = reward_spec
        logger.info(f"Updated reward_spec: {len(reward_spec)} scorers")
    
    if default_ensemble_config is not None:
        api_config.default_ensemble_config = default_ensemble_config
        logger.info(f"Updated default ensemble config")
    
    return {"status": "updated", "config": get_config()}

@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    """
    OpenAI-compatible chat completion endpoint with ensemble support.
    
    Supports both legacy parameters and new ensemble configuration.
    """
    try:
        # Determine ensemble configuration
        if req.ensemble_config:
            ensemble_config = req.ensemble_config
        else:
            # Use legacy parameters or defaults
            ensemble_config = EnsembleMethodConfig(
                model_selection_method=req.model_selection_method or api_config.default_ensemble_config.model_selection_method,
                aggregation_method=req.ensemble_method or api_config.default_ensemble_config.aggregation_method,
                use_model_selection=api_config.default_ensemble_config.use_model_selection,
                use_output_aggregation=api_config.default_ensemble_config.use_output_aggregation
            )
        
        # Handle prompt format
        if isinstance(req.prompt, list):
            prompt_text = " ".join(req.prompt)
        else:
            prompt_text = req.prompt
        
        # Create example for ensemble
        example = {
            "instruction": "You are a helpful assistant.",
            "input": prompt_text,
            "output": ""
        }
        
        # Run ensemble inference
        result = run_ensemble(
            example=example,
            model_specs=api_config.model_specs,
            reward_spec=api_config.reward_spec,
            ensemble_method=ensemble_config.aggregation_method,
            model_selection_method=ensemble_config.model_selection_method,
            max_tokens=req.max_tokens,
            temperature=req.temperature
        )
        
        output = result["output"]
        
        # Apply stop sequences
        finish_reason = "length"
        if req.stop:
            for stop_seq in req.stop:
                if stop_seq in output:
                    output = output.split(stop_seq)[0]
                    finish_reason = "stop"
                    break
        
        # Apply max_tokens limit (approximate word count)
        if len(output.split()) > req.max_tokens:
            output = " ".join(output.split()[:req.max_tokens])
            finish_reason = "length"
        
        return {
            "id": f"cmpl-{uuid.uuid4()}",
            "object": "text_completion", 
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "text": output,
                    "finish_reason": finish_reason,
                    "metadata": {
                        "selected_models": result.get("selected_models", []),
                        "method": result.get("method", "unknown"),
                        "ensemble_config": ensemble_config.dict()
                    }
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_text.split()),
                "completion_tokens": len(output.split()),
                "total_tokens": len(prompt_text.split()) + len(output.split())
            }
        }
        
    except Exception as e:
        logger.error(f"Error in chat_completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/ensemble/inference")
def ensemble_inference(req: EnsembleRequest):
    """
    Direct ensemble inference with full configuration control.
    
    This endpoint provides direct access to the ensemble framework
    with complete control over all parameters.
    """
    try:
        # Create ensemble configuration
        config = EnsembleConfig(
            use_model_selection=req.ensemble_config.use_model_selection,
            model_selection_method=req.ensemble_config.model_selection_method,
            model_selection_params=req.ensemble_config.model_selection_params,
            use_output_aggregation=req.ensemble_config.use_output_aggregation,
            aggregation_method=req.ensemble_config.aggregation_method,
            aggregation_level=req.ensemble_config.aggregation_level,
            aggregation_params=req.ensemble_config.aggregation_params
        )
        
        # Create ensemble framework
        framework = EnsembleFramework(config)
        
        # Prepare example
        example = {
            "instruction": req.instruction,
            "input": req.input,
            "output": req.output
        }
        
        # Run ensemble
        result = framework.run_ensemble(
            example=example,
            model_specs=api_config.model_specs,
            generators=api_config.generator_pool,
            scorers=api_config.scorer_pool,
            model_stats=api_config.model_stats,
            max_rounds=req.max_rounds,
            score_threshold=req.score_threshold,
            max_tokens=req.max_tokens
        )
        
        return {
            "id": f"ensemble-{uuid.uuid4()}",
            "created": int(time.time()),
            "result": result,
            "config": config.__dict__
        }
        
    except Exception as e:
        logger.error(f"Error in ensemble_inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/ensemble/batch")
def batch_inference(req: BatchRequest):
    """
    Batch ensemble inference for processing multiple examples.
    
    Supports batch processing with configurable batch size.
    """
    try:
        results = []
        
        # Process examples in batches
        for i in range(0, len(req.examples), req.batch_size):
            batch = req.examples[i:i + req.batch_size]
            batch_results = []
            
            for example_req in batch:
                try:
                    # Use the ensemble_inference logic for each example
                    result = ensemble_inference(example_req)
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing example {i}: {e}")
                    batch_results.append({
                        "error": str(e),
                        "example_index": i
                    })
            
            results.extend(batch_results)
        
        return {
            "id": f"batch-{uuid.uuid4()}",
            "created": int(time.time()),
            "total_examples": len(req.examples),
            "batch_size": req.batch_size,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in batch_inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Factory methods for easy configuration
@app.post("/v1/ensemble/presets/simple")
def simple_ensemble(
    prompt: str,
    ensemble_method: str = "reward_based",
    model_selection_method: str = "all",
    max_tokens: int = 256
):
    """Preset: Simple ensemble with basic configuration"""
    req = ChatCompletionRequest(
        model="simple-ensemble",
        prompt=prompt,
        max_tokens=max_tokens,
        ensemble_config=EnsembleMethodConfig(
            model_selection_method=model_selection_method,
            aggregation_method=ensemble_method,
            use_model_selection=True,
            use_output_aggregation=True
        )
    )
    return chat_completions(req)

@app.post("/v1/ensemble/presets/selection_only")
def selection_only(
    prompt: str,
    model_selection_method: str = "zscore",
    max_tokens: int = 256
):
    """Preset: Model selection only (no output aggregation)"""
    req = ChatCompletionRequest(
        model="selection-only",
        prompt=prompt,
        max_tokens=max_tokens,
        ensemble_config=EnsembleMethodConfig(
            model_selection_method=model_selection_method,
            use_model_selection=True,
            use_output_aggregation=False
        )
    )
    return chat_completions(req)

@app.post("/v1/ensemble/presets/aggregation_only")
def aggregation_only(
    prompt: str,
    aggregation_method: str = "reward_based",
    aggregation_level: str = "sentence",
    max_tokens: int = 256
):
    """Preset: Output aggregation only (use all models)"""
    req = ChatCompletionRequest(
        model="aggregation-only",
        prompt=prompt,
        max_tokens=max_tokens,
        ensemble_config=EnsembleMethodConfig(
            aggregation_method=aggregation_method,
            aggregation_level=aggregation_level,
            use_model_selection=False,
            use_output_aggregation=True
        )
    )
    return chat_completions(req)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)