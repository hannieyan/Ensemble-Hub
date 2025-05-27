#!/usr/bin/env python3
"""Capture lm-eval request by running a mock API server"""

from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional, List, Union, Dict, Any
import json
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: Optional[Union[List[Message], List[List[Message]], List[Dict[str, str]]]] = None
    prompt: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stop: Optional[List[str]] = None
    stream: Optional[bool] = None
    # Add any other fields lm-eval might send
    class Config:
        extra = "allow"  # Allow extra fields

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Capture and log the raw request"""
    # Get raw body
    body = await request.body()
    body_str = body.decode('utf-8')
    
    # Parse JSON
    try:
        data = json.loads(body_str)
    except:
        data = {"error": "Failed to parse JSON", "raw": body_str}
    
    # Log everything
    logger.info("="*80)
    logger.info("CAPTURED LM-EVAL REQUEST:")
    logger.info(f"Headers: {dict(request.headers)}")
    logger.info(f"Method: {request.method}")
    logger.info(f"URL: {request.url}")
    logger.info(f"Raw Body: {body_str}")
    logger.info(f"Parsed Data: {json.dumps(data, indent=2)}")
    logger.info("="*80)
    
    # Save to file
    with open("lmeval_request_captured.json", "w") as f:
        json.dump({
            "headers": dict(request.headers),
            "method": request.method,
            "url": str(request.url),
            "body": data
        }, f, indent=2)
    
    # Return a mock response in text completion format
    return {
        "id": "cmpl-mock",
        "object": "text_completion",
        "created": 1234567890,
        "model": "mock-model",
        "choices": [{
            "index": 0,
            "text": "Janet makes $18 at the farmers' market every day.",
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }

if __name__ == "__main__":
    print("Starting capture server on port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)