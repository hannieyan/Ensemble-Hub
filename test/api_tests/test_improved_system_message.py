#!/usr/bin/env python3
"""Test improved system message handling for lm-eval compatibility"""

import requests
import json

def test_system_message_handling():
    """Test how the API handles system messages in different scenarios"""
    
    base_url = "http://localhost:8000"
    endpoint = f"{base_url}/v1/chat/completions"
    
    print("Testing improved system message handling")
    print("="*80)
    
    # Test 1: lm-eval style request (prompt field, no system)
    print("\nTest 1: lm-eval style request")
    print("-"*40)
    lmeval_request = {
        "prompt": "Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\nAnswer:",
        "model": "ensemble",
        "max_tokens": 256,
        "temperature": 0.0,
        "stop": ["Question:", "</s>", "<|im_end|>"],
        "seed": 1234
    }
    
    print("Request (lm-eval format):")
    print(json.dumps(lmeval_request, indent=2))
    print("\nExpected behavior:")
    print("- No system message in request")
    print("- API will use improved default instruction:")
    print("  'You are a helpful and harmless AI assistant. You should follow the user's instructions")
    print("   carefully and precisely. Be accurate in your responses, think step by step when solving")
    print("   problems, and provide clear, direct answers. Focus on completing the specific task requested.'")
    
    # Test 2: Chat format with custom system message
    print("\n\nTest 2: Chat format with custom system message")
    print("-"*40)
    chat_request = {
        "messages": [
            {"role": "system", "content": "You are a math tutor. Explain your reasoning clearly."},
            {"role": "user", "content": "Question: What is 15% of 80?\nAnswer:"}
        ],
        "model": "ensemble",
        "max_tokens": 100,
        "temperature": 0.0
    }
    
    print("Request (chat format with system):")
    print(json.dumps(chat_request, indent=2))
    print("\nExpected behavior:")
    print("- Uses provided system message: 'You are a math tutor. Explain your reasoning clearly.'")
    
    # Test 3: Chat format without system message
    print("\n\nTest 3: Chat format without system message")
    print("-"*40)
    chat_no_system = {
        "messages": [
            {"role": "user", "content": "Question: What is 15% of 80?\nAnswer:"}
        ],
        "model": "ensemble",
        "max_tokens": 100,
        "temperature": 0.0
    }
    
    print("Request (chat format without system):")
    print(json.dumps(chat_no_system, indent=2))
    print("\nExpected behavior:")
    print("- No system message in request")
    print("- API will use the same improved default instruction")
    
    # Test with actual API if running
    print("\n" + "="*80)
    print("Testing with live API (if available)...")
    
    try:
        # Quick test with minimal request
        test_request = {
            "prompt": "2 + 2 =",
            "model": "ensemble",
            "max_tokens": 10,
            "temperature": 0.0,
            "ensemble_config": {
                "show_attribution": True
            }
        }
        
        response = requests.post(endpoint, json=test_request, timeout=5)
        if response.status_code == 200:
            print("✅ API is responding correctly")
            result = response.json()
            print(f"Response: {result.get('choices', [{}])[0].get('text', 'N/A')}")
        else:
            print(f"⚠️  API returned status {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("ℹ️  API is not running (start with: python -m ensemblehub.api)")
    except Exception as e:
        print(f"❌ Error testing API: {e}")

if __name__ == "__main__":
    test_system_message_handling()