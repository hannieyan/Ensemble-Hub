#!/usr/bin/env python3
"""Test how the API handles list of strings as input"""

import requests
import json

def test_list_str_input():
    url = "http://localhost:8000/v1/chat/completions"
    
    print("Test 1: List of strings via 'prompt' field (legacy support)")
    print("="*60)
    
    # This should work - converts to batch request
    request1 = {
        "model": "ensemble",
        "prompt": ["What is 2+2?", "What is 3+3?", "What is 4+4?"],
        "max_tokens": 50,
        "ensemble_config": {
            "ensemble_method": "random",
            "model_selection_method": "all"
        }
    }
    
    try:
        response = requests.post(url, json=request1)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Got {len(result.get('choices', []))} responses")
            for i, choice in enumerate(result.get('choices', [])):
                print(f"   Response {i+1}: {choice.get('text', '')[:50]}...")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   {response.text}")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print("\n")
    print("Test 2: List of strings via 'messages' field (should fail)")
    print("="*60)
    
    # This should fail - messages expects Message objects
    request2 = {
        "model": "ensemble",
        "messages": ["What is 2+2?", "What is 3+3?"],  # Wrong format!
        "max_tokens": 50
    }
    
    try:
        response = requests.post(url, json=request2)
        if response.status_code == 200:
            print(f"🤔 Unexpected success: {response.json()}")
        else:
            print(f"✅ Expected error: {response.status_code}")
            print(f"   {response.json().get('detail', response.text)}")
    except Exception as e:
        print(f"✅ Expected exception: {e}")
    
    print("\n")
    print("Test 3: Proper batch format via 'messages' field")
    print("="*60)
    
    # Correct batch format
    request3 = {
        "model": "ensemble",
        "messages": [
            [{"role": "user", "content": "What is 2+2?"}],
            [{"role": "user", "content": "What is 3+3?"}]
        ],
        "max_tokens": 50
    }
    
    try:
        response = requests.post(url, json=request3)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Got {len(result.get('choices', []))} responses")
            for i, choice in enumerate(result.get('choices', [])):
                content = choice.get('message', {}).get('content', '')[:50]
                print(f"   Response {i+1}: {content}...")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   {response.text}")
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_list_str_input()