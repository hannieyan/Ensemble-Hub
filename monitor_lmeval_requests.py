#!/usr/bin/env python3
"""
Monitor and log requests from lm-evaluation-harness to understand its API usage
"""
import json
import time
from datetime import datetime
from typing import Any, Dict
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lmeval_requests.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="LM-Eval Request Monitor")

# Store requests for analysis
request_history = []

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests"""
    start_time = time.time()
    
    # Get request details
    request_data = {
        "timestamp": datetime.now().isoformat(),
        "method": request.method,
        "url": str(request.url),
        "path": request.url.path,
        "query_params": dict(request.query_params),
        "headers": dict(request.headers),
    }
    
    # Get request body if present
    if request.method in ["POST", "PUT", "PATCH"]:
        body = await request.body()
        request._body = body  # Store for later use
        try:
            request_data["body"] = json.loads(body.decode('utf-8'))
        except:
            request_data["body"] = body.decode('utf-8')
    
    # Log the request
    logger.info(f"\n{'='*60}")
    logger.info(f"REQUEST: {request.method} {request.url.path}")
    logger.info(f"Headers: {json.dumps(dict(request.headers), indent=2)}")
    if "body" in request_data:
        logger.info(f"Body: {json.dumps(request_data['body'], indent=2)}")
    
    # Process the request
    response = await call_next(request)
    
    # Log response info
    process_time = time.time() - start_time
    logger.info(f"Response Status: {response.status_code}")
    logger.info(f"Process Time: {process_time:.3f}s")
    
    # Store in history
    request_data["response_status"] = response.status_code
    request_data["process_time"] = process_time
    request_history.append(request_data)
    
    return response

@app.post("/v1/completions")
async def completions(request: Request):
    """OpenAI-compatible completions endpoint that lm-eval uses"""
    body = await request.json()
    
    logger.info("\n--- LM-EVAL COMPLETIONS REQUEST ---")
    logger.info(f"Model: {body.get('model', 'not specified')}")
    logger.info(f"Prompt Type: {type(body.get('prompt', [])).__name__}")
    
    # Analyze prompt structure
    prompt = body.get('prompt', [])
    if isinstance(prompt, list):
        logger.info(f"Batch Size: {len(prompt)}")
        if prompt:
            logger.info(f"First Prompt Sample: {prompt[0][:200]}...")
            logger.info(f"Prompt Lengths: {[len(p) for p in prompt[:5]]}...")
    else:
        logger.info(f"Single Prompt: {prompt[:200]}...")
    
    # Log other parameters
    logger.info(f"Max Tokens: {body.get('max_tokens', 'not specified')}")
    logger.info(f"Temperature: {body.get('temperature', 'not specified')}")
    logger.info(f"Top P: {body.get('top_p', 'not specified')}")
    logger.info(f"Stop: {body.get('stop', 'not specified')}")
    logger.info(f"Echo: {body.get('echo', 'not specified')}")
    logger.info(f"Logprobs: {body.get('logprobs', 'not specified')}")
    
    # Additional parameters
    extra_params = {k: v for k, v in body.items() 
                   if k not in ['model', 'prompt', 'max_tokens', 'temperature', 'top_p', 'stop', 'echo', 'logprobs']}
    if extra_params:
        logger.info(f"Extra Parameters: {json.dumps(extra_params, indent=2)}")
    
    # Mock response for lm-eval
    choices = []
    prompts = prompt if isinstance(prompt, list) else [prompt]
    
    for i, p in enumerate(prompts):
        # Simple mock completion
        completion = " completion text for testing"
        choices.append({
            "text": p + completion if body.get('echo', False) else completion,
            "index": i,
            "logprobs": None,
            "finish_reason": "length"
        })
    
    response = {
        "id": f"cmpl-{int(time.time())}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": body.get('model', 'mock-model'),
        "choices": choices,
        "usage": {
            "prompt_tokens": sum(len(p.split()) for p in prompts),
            "completion_tokens": len(choices) * 5,
            "total_tokens": sum(len(p.split()) for p in prompts) + len(choices) * 5
        }
    }
    
    return JSONResponse(content=response)

@app.get("/v1/models")
async def models():
    """Models endpoint that lm-eval might call"""
    logger.info("\n--- LM-EVAL MODELS REQUEST ---")
    
    return {
        "object": "list",
        "data": [
            {
                "id": "ensemble",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ensemble-hub"
            }
        ]
    }

@app.get("/request_history")
async def get_request_history():
    """Get all logged requests"""
    return {
        "total_requests": len(request_history),
        "requests": request_history
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "LM-Eval Request Monitor",
        "purpose": "Monitor and analyze requests from lm-evaluation-harness",
        "endpoints": {
            "/v1/completions": "OpenAI-compatible completions endpoint",
            "/v1/models": "List available models",
            "/request_history": "View all logged requests"
        },
        "total_requests_logged": len(request_history)
    }

if __name__ == "__main__":
    print("🔍 LM-Eval Request Monitor")
    print("=" * 60)
    print("This server will log all requests from lm-evaluation-harness")
    print("Logs are saved to: lmeval_requests.log")
    print("\nTo test with lm-eval, run:")
    print("lm_eval --model local-completions \\")
    print("    --tasks gsm8k \\")
    print("    --model_args model=ensemble,base_url=http://localhost:8001/v1/completions \\")
    print("    --batch_size 2 \\")
    print("    --limit 5")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8001)