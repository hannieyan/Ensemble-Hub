#!/usr/bin/env python3
"""Test concurrent model loading fix"""

import requests
import json
from concurrent.futures import ThreadPoolExecutor
import time

def test_concurrent_requests():
    """Test multiple concurrent requests to ensure model loading is thread-safe"""
    
    base_url = "http://localhost:8000"
    endpoint = f"{base_url}/v1/chat/completions"
    
    print("Testing concurrent model loading fix")
    print("="*80)
    
    # Create multiple test requests
    test_prompts = [
        "What is 1+1?",
        "What is 2+2?", 
        "What is 3+3?",
        "What is 4+4?"
    ]
    
    # Test 1: Multiple requests using prompt field (lm-eval style)
    print("\nTest 1: Concurrent requests with prompt field")
    print("-"*40)
    
    def make_request(prompt):
        request = {
            "prompt": prompt,
            "model": "ensemble",
            "max_tokens": 20,
            "temperature": 0.0,
            "ensemble_config": {
                "ensemble_method": "random",
                "model_selection_method": "all"
            }
        }
        
        try:
            start = time.time()
            response = requests.post(endpoint, json=request, timeout=30)
            elapsed = time.time() - start
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('choices', [{}])[0].get('text', 'N/A')
                return f"✅ '{prompt}' -> '{answer.strip()}' ({elapsed:.2f}s)"
            else:
                return f"❌ '{prompt}' -> Error {response.status_code}: {response.text[:100]}"
        except Exception as e:
            return f"❌ '{prompt}' -> Exception: {str(e)}"
    
    # Send requests concurrently
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(make_request, prompt) for prompt in test_prompts]
        results = [f.result() for f in futures]
    
    for result in results:
        print(result)
    
    # Test 2: Batch request (should work without issues)
    print("\n\nTest 2: Batch request")
    print("-"*40)
    
    batch_request = {
        "prompt": test_prompts,
        "model": "ensemble",
        "max_tokens": 20,
        "temperature": 0.0,
        "ensemble_config": {
            "ensemble_method": "random",
            "model_selection_method": "all"
        }
    }
    
    try:
        start = time.time()
        response = requests.post(endpoint, json=batch_request, timeout=30)
        elapsed = time.time() - start
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch request successful ({elapsed:.2f}s)")
            for i, choice in enumerate(result.get('choices', [])):
                answer = choice.get('text', 'N/A')
                print(f"   {test_prompts[i]} -> {answer.strip()}")
        else:
            print(f"❌ Batch request failed: {response.status_code}")
            print(f"   {response.text[:200]}")
    except Exception as e:
        print(f"❌ Batch request exception: {e}")

if __name__ == "__main__":
    test_concurrent_requests()