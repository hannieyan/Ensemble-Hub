#!/usr/bin/env python3
"""
Comprehensive test suite for lm-evaluation-harness compatibility

This script tests various scenarios that lm-evaluation-harness might encounter,
including different prompt formats, batch processing, and error handling.

Usage:
    python test_lmeval_compatibility.py [--port 9876] [--host localhost]
"""

import requests
import json
import time
import argparse
from typing import List, Dict, Any

class LMEvalCompatibilityTester:
    def __init__(self, base_url: str = "http://localhost:9876"):
        self.base_url = base_url
        self.endpoint = f"{base_url}/v1/chat/completions"
        self.test_results = []
    
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test results"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"    {details}")
        
        self.test_results.append({
            "test": test_name,
            "success": success,
            "details": details
        })
    
    def test_api_health(self) -> bool:
        """Test if API is accessible"""
        try:
            response = requests.get(f"{self.base_url}/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.log_test("API Health Check", True, f"Status: {data.get('status', 'unknown')}")
                return True
            else:
                self.log_test("API Health Check", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("API Health Check", False, f"Connection error: {e}")
            return False
    
    def test_single_prompt_text_completion(self) -> bool:
        """Test single prompt in text completion format (lm-eval standard)"""
        payload = {
            "model": "ensemble",
            "prompt": ["The capital of France is"],
            "max_tokens": 50,
            "temperature": 0.0
        }
        
        try:
            response = requests.post(self.endpoint, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                
                # Validate response format
                if (result.get("object") == "text_completion" and 
                    "choices" in result and 
                    len(result["choices"]) > 0 and
                    "text" in result["choices"][0]):
                    
                    text = result["choices"][0]["text"]
                    self.log_test("Single Prompt Text Completion", True, 
                                f"Generated: '{text[:30]}...'")
                    return True
                else:
                    self.log_test("Single Prompt Text Completion", False, 
                                "Invalid response format")
                    return False
            else:
                self.log_test("Single Prompt Text Completion", False, 
                            f"HTTP {response.status_code}: {response.text[:100]}")
                return False
        except Exception as e:
            self.log_test("Single Prompt Text Completion", False, f"Exception: {e}")
            return False
    
    def test_batch_prompts_text_completion(self) -> bool:
        """Test multiple prompts in batch (what lm-eval uses for efficiency)"""
        payload = {
            "model": "ensemble",
            "prompt": [
                "The capital of France is",
                "2 + 2 equals",
                "The largest planet in our solar system is"
            ],
            "max_tokens": 30,
            "temperature": 0.0
        }
        
        try:
            response = requests.post(self.endpoint, json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
                
                # Validate batch response
                if (result.get("object") == "text_completion" and 
                    "choices" in result and 
                    len(result["choices"]) == 3):
                    
                    all_have_text = all("text" in choice for choice in result["choices"])
                    if all_have_text:
                        texts = [choice["text"][:20] + "..." for choice in result["choices"]]
                        self.log_test("Batch Prompts Text Completion", True, 
                                    f"Generated {len(texts)} responses")
                        for i, text in enumerate(texts):
                            print(f"      [{i}]: {text}")
                        return True
                    else:
                        self.log_test("Batch Prompts Text Completion", False, 
                                    "Some choices missing 'text' field")
                        return False
                else:
                    self.log_test("Batch Prompts Text Completion", False, 
                                f"Expected 3 choices, got {len(result.get('choices', []))}")
                    return False
            else:
                self.log_test("Batch Prompts Text Completion", False, 
                            f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Batch Prompts Text Completion", False, f"Exception: {e}")
            return False
    
    def test_chat_completion_format(self) -> bool:
        """Test standard chat completion format (for comparison)"""
        payload = {
            "model": "ensemble",
            "messages": [
                {"role": "user", "content": "What is 5 + 3?"}
            ],
            "max_tokens": 50,
            "temperature": 0.0
        }
        
        try:
            response = requests.post(self.endpoint, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                
                if (result.get("object") == "chat.completion" and 
                    "choices" in result and 
                    len(result["choices"]) > 0 and
                    "message" in result["choices"][0]):
                    
                    message = result["choices"][0]["message"]["content"]
                    self.log_test("Chat Completion Format", True, 
                                f"Generated: '{message[:30]}...'")
                    return True
                else:
                    self.log_test("Chat Completion Format", False, 
                                "Invalid chat completion format")
                    return False
            else:
                self.log_test("Chat Completion Format", False, 
                            f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Chat Completion Format", False, f"Exception: {e}")
            return False
    
    def test_progressive_ensemble_with_attribution(self) -> bool:
        """Test progressive ensemble with model attribution"""
        payload = {
            "model": "progressive-ensemble",
            "prompt": ["Solve step by step: What is 15 + 27?"],
            "max_tokens": 200,
            "temperature": 0.0,
            "ensemble_config": {
                "ensemble_method": "progressive",
                "progressive_mode": "length",
                "length_thresholds": [50, 100],
                "show_attribution": True
            }
        }
        
        try:
            response = requests.post(self.endpoint, json=payload, timeout=45)
            if response.status_code == 200:
                result = response.json()
                
                if ("choices" in result and 
                    len(result["choices"]) > 0 and
                    "metadata" in result["choices"][0] and
                    "attribution" in result["choices"][0]["metadata"]):
                    
                    attribution = result["choices"][0]["metadata"]["attribution"]
                    self.log_test("Progressive Ensemble with Attribution", True, 
                                f"Attribution: {attribution.get('summary', 'N/A')}")
                    return True
                else:
                    self.log_test("Progressive Ensemble with Attribution", False, 
                                "No attribution data found")
                    return False
            else:
                self.log_test("Progressive Ensemble with Attribution", False, 
                            f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Progressive Ensemble with Attribution", False, f"Exception: {e}")
            return False
    
    def test_various_ensemble_methods(self) -> bool:
        """Test different ensemble methods"""
        methods = ["simple", "progressive", "random", "loop"]
        success_count = 0
        
        for method in methods:
            payload = {
                "model": f"{method}-ensemble",
                "prompt": [f"Test {method} ensemble: 2 + 2 = "],
                "max_tokens": 20,
                "temperature": 0.0,
                "ensemble_config": {
                    "ensemble_method": method,
                    "model_selection_method": "all"
                }
            }
            
            try:
                response = requests.post(self.endpoint, json=payload, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        text = result["choices"][0].get("text", "")
                        print(f"      {method}: '{text[:20]}...'")
                        success_count += 1
                    else:
                        print(f"      {method}: Invalid response format")
                else:
                    print(f"      {method}: HTTP {response.status_code}")
            except Exception as e:
                print(f"      {method}: Exception - {e}")
        
        success = success_count == len(methods)
        self.log_test("Various Ensemble Methods", success, 
                    f"{success_count}/{len(methods)} methods working")
        return success
    
    def test_error_handling(self) -> bool:
        """Test API error handling"""
        test_cases = [
            {
                "name": "Missing prompt and messages",
                "payload": {"model": "test", "max_tokens": 10},
                "expected_status": 422
            },
            {
                "name": "Invalid ensemble method",
                "payload": {
                    "model": "test",
                    "prompt": ["test"],
                    "ensemble_config": {"ensemble_method": "invalid_method"}
                },
                "expected_status": 500  # Should fail gracefully
            }
        ]
        
        passed = 0
        for case in test_cases:
            try:
                response = requests.post(self.endpoint, json=case["payload"], timeout=10)
                if response.status_code == case["expected_status"]:
                    passed += 1
                    print(f"      {case['name']}: ✅ (HTTP {response.status_code})")
                else:
                    print(f"      {case['name']}: ❌ Expected {case['expected_status']}, got {response.status_code}")
            except Exception as e:
                print(f"      {case['name']}: ❌ Exception - {e}")
        
        success = passed == len(test_cases)
        self.log_test("Error Handling", success, f"{passed}/{len(test_cases)} cases handled correctly")
        return success
    
    def test_performance_benchmark(self) -> bool:
        """Basic performance test"""
        payload = {
            "model": "ensemble",
            "prompt": ["Performance test: Count from 1 to 5:"],
            "max_tokens": 50,
            "temperature": 0.0
        }
        
        times = []
        for i in range(3):
            try:
                start_time = time.time()
                response = requests.post(self.endpoint, json=payload, timeout=30)
                end_time = time.time()
                
                if response.status_code == 200:
                    times.append(end_time - start_time)
                else:
                    self.log_test("Performance Benchmark", False, 
                                f"Request {i+1} failed with HTTP {response.status_code}")
                    return False
            except Exception as e:
                self.log_test("Performance Benchmark", False, f"Request {i+1} failed: {e}")
                return False
        
        avg_time = sum(times) / len(times)
        success = avg_time < 30.0  # Should complete within 30 seconds
        self.log_test("Performance Benchmark", success, 
                    f"Average response time: {avg_time:.2f}s")
        return success
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all compatibility tests"""
        print("🧪 LM-Evaluation-Harness Compatibility Test Suite")
        print("=" * 60)
        
        # Check API health first
        if not self.test_api_health():
            print("\n❌ API is not accessible. Please start the server:")
            print("   uvicorn ensemblehub.api:app --host 0.0.0.0 --port 9876")
            return {"success": False, "results": self.test_results}
        
        print()
        
        # Run all tests
        tests = [
            ("Core Functionality", [
                self.test_single_prompt_text_completion,
                self.test_batch_prompts_text_completion,
                self.test_chat_completion_format,
            ]),
            ("Ensemble Features", [
                self.test_progressive_ensemble_with_attribution,
                self.test_various_ensemble_methods,
            ]),
            ("Robustness", [
                self.test_error_handling,
                self.test_performance_benchmark,
            ])
        ]
        
        all_passed = True
        for category, test_functions in tests:
            print(f"\n📋 {category}:")
            for test_func in test_functions:
                success = test_func()
                if not success:
                    all_passed = False
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 Test Summary:")
        
        passed = sum(1 for result in self.test_results if result["success"])
        total = len(self.test_results)
        
        print(f"   Passed: {passed}/{total}")
        print(f"   Overall: {'✅ READY FOR LM-EVAL' if all_passed else '❌ NEEDS FIXING'}")
        
        if all_passed:
            print("\n🎉 All tests passed! You can now use lm-evaluation-harness:")
            print("   lm_eval --model openai-completions \\")
            print("           --tasks gsm8k \\")
            print("           --model_args model=ensemble,base_url=http://localhost:9876/v1/chat/completions \\")
            print("           --batch_size 2 \\")
            print("           --num_fewshot 5")
        else:
            print("\n⚠️  Some tests failed. Check the API server logs for details.")
        
        return {
            "success": all_passed,
            "passed": passed,
            "total": total,
            "results": self.test_results
        }

def main():
    parser = argparse.ArgumentParser(description="Test lm-evaluation-harness compatibility")
    parser.add_argument("--host", default="localhost", help="API host (default: localhost)")
    parser.add_argument("--port", type=int, default=9876, help="API port (default: 9876)")
    parser.add_argument("--output", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    tester = LMEvalCompatibilityTester(base_url)
    
    results = tester.run_all_tests()
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")

if __name__ == "__main__":
    main()