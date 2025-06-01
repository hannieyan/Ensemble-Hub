#!/usr/bin/env python3
"""
Simple test script for single model API usage
"""

import requests
import json
import sys

def test_single_model_api(base_url="http://localhost:8000"):
    """Test single model API with different request types"""
    
    print("🧪 Testing Single Model API")
    print("=" * 60)
    
    # Test 1: Simple chat completion
    print("\n1. Testing simple chat completion...")
    response = requests.post(f"{base_url}/v1/chat/completions", json={
        "model": "ensemble",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 50,
        "temperature": 0.0
    })
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Success!")
        print(f"Response: {result['choices'][0]['message']['content']}")
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"Response: {response.text}")
        return False
    
    # Test 2: Longer generation
    print("\n2. Testing longer generation...")
    response = requests.post(f"{base_url}/v1/chat/completions", json={
        "model": "ensemble",
        "messages": [{"role": "user", "content": "Explain quantum computing in simple terms"}],
        "max_tokens": 500,
        "temperature": 0.7
    })
    
    if response.status_code == 200:
        result = response.json()
        content = result['choices'][0]['message']['content']
        print("✅ Success!")
        print(f"Response length: {len(content)} characters")
        print(f"First 200 chars: {content[:200]}...")
    else:
        print(f"❌ Error: {response.status_code}")
        return False
    
    # Test 3: Text completion format (for lm-eval compatibility)
    print("\n3. Testing text completion format (lm-eval style)...")
    response = requests.post(f"{base_url}/v1/chat/completions", json={
        "model": "ensemble",
        "prompt": "The capital of France is",
        "max_tokens": 20,
        "temperature": 0.0
    })
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Success!")
        print(f"Response: {result['choices'][0]['text']}")
    else:
        print(f"❌ Error: {response.status_code}")
        return False
    
    # Test 4: Check configuration
    print("\n4. Checking current configuration...")
    response = requests.get(f"{base_url}/v1/ensemble/config")
    
    if response.status_code == 200:
        config = response.json()
        print("✅ Configuration retrieved:")
        print(f"Models: {len(config['model_specs'])}")
        for i, spec in enumerate(config['model_specs']):
            print(f"  [{i}] {spec['path']} ({spec['engine']} on {spec.get('device', 'auto')})")
    else:
        print(f"❌ Error: {response.status_code}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 All tests passed!")
    return True

if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    success = test_single_model_api(base_url)
    sys.exit(0 if success else 1)