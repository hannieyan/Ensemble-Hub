#!/usr/bin/env python3
"""
Test script to verify --disable_internal_template flag works correctly
"""

import subprocess
import requests
import time
import sys
import json

def test_template_flag():
    print("Testing --disable_internal_template flag...")
    
    # Start API server with disabled internal template
    print("Starting API server with --disable_internal_template...")
    api_process = subprocess.Popen([
        sys.executable, "-m", "ensemblehub.api",
        "--host", "127.0.0.1",
        "--port", "8001",
        "--disable_internal_template",
        "--show_input_details"
    ], cwd="/Users/fzkuji/PycharmProjects/Ensemble-Hub")
    
    # Wait for server to start
    time.sleep(10)
    
    try:
        # Test request
        test_prompt = "What is 2+2?"
        response = requests.post(
            "http://127.0.0.1:8001/v1/completions",
            json={
                "prompt": test_prompt,
                "max_tokens": 50,
                "temperature": 0.1
            },
            timeout=30
        )
        
        print(f"Response status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print(f"Error: {response.text}")
    
    except Exception as e:
        print(f"Error making request: {e}")
    
    finally:
        # Stop API server
        print("Stopping API server...")
        api_process.terminate()
        api_process.wait()

if __name__ == "__main__":
    test_template_flag()