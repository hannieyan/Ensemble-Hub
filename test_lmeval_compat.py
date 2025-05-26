#!/usr/bin/env python3
"""
Test lm-evaluation-harness compatibility
"""

import requests
import json

def test_text_completion_format():
    """Test the format that lm-evaluation-harness expects"""
    
    # This is similar to what lm-evaluation-harness sends
    payload = {
        "model": "ensemble",
        "prompt": ["Complete this: The capital of France is"],
        "max_tokens": 50,
        "temperature": 0.0
    }
    
    print("Testing lm-eval compatible format:")
    print(json.dumps(payload, indent=2))
    
    try:
        response = requests.post("http://localhost:9876/v1/chat/completions", json=payload)
        print(f"\nStatus: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"Response format:")
            print(f"  Object: {result['object']}")
            print(f"  Choices[0] keys: {list(result['choices'][0].keys())}")
            
            choice = result['choices'][0]
            if 'text' in choice:
                print(f"  Text: {choice['text'][:100]}...")
            if 'message' in choice and choice['message']:
                print(f"  Message content: {choice['message']['content'][:100]}...")
                
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def test_chat_completion_format():
    """Test the standard chat completion format"""
    
    payload = {
        "model": "ensemble",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "max_tokens": 50,
        "temperature": 0.0
    }
    
    print("\n" + "="*50)
    print("Testing standard chat completion format:")
    print(json.dumps(payload, indent=2))
    
    try:
        response = requests.post("http://localhost:9876/v1/chat/completions", json=payload)
        print(f"\nStatus: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"Response format:")
            print(f"  Object: {result['object']}")
            print(f"  Choices[0] keys: {list(result['choices'][0].keys())}")
            
            choice = result['choices'][0]
            if 'text' in choice and choice['text']:
                print(f"  Text: {choice['text'][:100]}...")
            if 'message' in choice and choice['message']:
                print(f"  Message content: {choice['message']['content'][:100]}...")
                
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing lm-evaluation-harness compatibility")
    print("="*60)
    
    success1 = test_text_completion_format()
    success2 = test_chat_completion_format()
    
    print("\n" + "="*60)
    print("📊 Summary:")
    print(f"Text completion format: {'✅' if success1 else '❌'}")
    print(f"Chat completion format: {'✅' if success2 else '❌'}")
    
    if success1 and success2:
        print("🎉 All tests passed! Ready for lm-evaluation-harness")
    else:
        print("⚠️  Some tests failed")