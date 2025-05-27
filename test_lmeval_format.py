#!/usr/bin/env python3
"""Test lm-eval request format and check for alternative system message fields"""

import json
import requests

def test_lmeval_format():
    """Test how lm-eval sends requests"""
    
    print("Analyzing lm-eval request format from captured data:")
    print("="*60)
    
    # Load the captured request
    with open('lmeval_request_captured.json', 'r') as f:
        captured = json.load(f)
    
    body = captured['body']
    print("Request body keys:", list(body.keys()))
    print()
    
    # Check for various possible fields
    possible_system_fields = [
        'system', 'system_prompt', 'system_message',
        'instruction', 'instructions', 'context',
        'prefix', 'preamble', 'role_prompt'
    ]
    
    print("Checking for alternative system message fields:")
    for field in possible_system_fields:
        if field in body:
            print(f"  ✓ Found '{field}': {body[field][:100]}...")
        else:
            print(f"  ✗ No '{field}' field")
    
    print("\n" + "="*60)
    print("Current behavior:")
    print("- lm-eval uses 'prompt' field (legacy format)")
    print("- No system/instruction field provided")
    print("- API will use default system message")
    
    print("\n" + "="*60)
    print("Testing API response with different formats:")
    
    # Test 1: Legacy format (what lm-eval uses)
    print("\nTest 1: Legacy format with 'prompt' field")
    legacy_request = {
        "prompt": "Question: What is 2+2?\nAnswer:",
        "model": "ensemble",
        "max_tokens": 50,
        "temperature": 0.0
    }
    print(f"Request: {json.dumps(legacy_request, indent=2)}")
    
    # Test 2: Chat format with system message
    print("\nTest 2: Chat format with system message")
    chat_request = {
        "messages": [
            {"role": "system", "content": "You are a math expert. Solve problems step by step."},
            {"role": "user", "content": "Question: What is 2+2?\nAnswer:"}
        ],
        "model": "ensemble",
        "max_tokens": 50,
        "temperature": 0.0
    }
    print(f"Request: {json.dumps(chat_request, indent=2)}")
    
    # Test 3: Chat format without system message
    print("\nTest 3: Chat format without system message")
    chat_no_system = {
        "messages": [
            {"role": "user", "content": "Question: What is 2+2?\nAnswer:"}
        ],
        "model": "ensemble",
        "max_tokens": 50,
        "temperature": 0.0
    }
    print(f"Request: {json.dumps(chat_no_system, indent=2)}")

def check_lmeval_library():
    """Check if we can find how lm-eval constructs requests"""
    print("\n" + "="*60)
    print("Checking for lm-eval patterns in code:")
    
    import subprocess
    
    # Search for how lm-eval might set system prompts
    patterns = [
        "system.*prompt",
        "system.*message", 
        "instruction.*prompt",
        "create_prompt",
        "format_prompt"
    ]
    
    for pattern in patterns:
        print(f"\nSearching for '{pattern}':")
        try:
            result = subprocess.run(
                ["grep", "-r", "-i", pattern, ".", "--include=*.py"],
                capture_output=True,
                text=True,
                timeout=5
            )
            lines = result.stdout.strip().split('\n')[:3]  # First 3 matches
            for line in lines:
                if line and 'test_' not in line:
                    print(f"  {line[:120]}...")
        except:
            print("  (search failed)")

if __name__ == "__main__":
    test_lmeval_format()
    # check_lmeval_library()  # Uncomment to search codebase