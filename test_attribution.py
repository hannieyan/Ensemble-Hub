#!/usr/bin/env python3
"""Test script to verify show_attribution functionality"""

import requests
import json
import logging

# Set up logging to see the attribution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_attribution():
    url = "http://localhost:8000/v1/chat/completions"
    
    # Test with show_attribution=True
    request_with_attribution = {
        "model": "ensemble",
        "messages": [
            {"role": "user", "content": "What is 2+2?"}
        ],
        "max_tokens": 50,
        "ensemble_config": {
            "ensemble_method": "random",
            "model_selection_method": "all",
            "show_attribution": True  # This should trigger attribution logging
        }
    }
    
    print("Testing with show_attribution=True:")
    response = requests.post(url, json=request_with_attribution)
    result = response.json()
    
    print(f"Response status: {response.status_code}")
    print(f"Response: {json.dumps(result, indent=2)}")
    
    # Check if attribution is in metadata
    if result.get("choices") and result["choices"][0].get("metadata", {}).get("attribution"):
        print("\n✅ Attribution data found in response metadata!")
        attribution = result["choices"][0]["metadata"]["attribution"]
        print(f"Attribution summary: {attribution.get('summary', 'N/A')}")
    else:
        print("\n❌ No attribution data found in response metadata")
    
    print("\n" + "="*80 + "\n")
    
    # Test with show_attribution=False
    request_without_attribution = {
        "model": "ensemble",
        "messages": [
            {"role": "user", "content": "What is 3+3?"}
        ],
        "max_tokens": 50,
        "ensemble_config": {
            "ensemble_method": "random",
            "model_selection_method": "all",
            "show_attribution": False  # This should NOT trigger attribution logging
        }
    }
    
    print("Testing with show_attribution=False:")
    response = requests.post(url, json=request_without_attribution)
    result = response.json()
    
    print(f"Response status: {response.status_code}")
    print(f"Response: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    test_attribution()