"""
Comprehensive API test script for HuggingFace endpoints.
This script tests all available endpoints with proper error handling.
"""
import requests
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:5000"
HF_API_BASE = f"{BASE_URL}/api/huggingface"

class HuggingFaceAPITester:
    """Test class for HuggingFace API endpoints."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.hf_api_base = f"{base_url}/api/huggingface"
        self.test_results = []
    
    def log_test(self, test_name: str, success: bool, response: Dict[Any, Any], error: str = None):
        """Log test results."""
        result = {
            "test": test_name,
            "success": success,
            "response": response,
            "error": error,
            "timestamp": time.strftime("%H:%M:%S")
        }
        self.test_results.append(result)
        
        status = "✓" if success else "✗"
        print(f"{status} {test_name}")
        if error:
            print(f"  Error: {error}")
        elif success and "message" in response:
            print(f"  {response['message']}")
    
    def test_dependencies(self):
        """Test dependencies check endpoint."""
        try:
            response = requests.get(f"{self.hf_api_base}/dependencies")
            data = response.json()
            self.log_test("Check Dependencies", response.status_code == 200, data)
            return data.get("success", False)
        except Exception as e:
            self.log_test("Check Dependencies", False, {}, str(e))
            return False
    
    def test_add_models(self):
        """Test adding multiple models."""
        models_to_add = [
            {
                "model_id": "gpt2",
                "name": "GPT-2",
                "model_type": "text-generation",
                "description": "OpenAI GPT-2 language model",
                "parameters": {
                    "max_new_tokens": 50,
                    "temperature": 0.8,
                    "do_sample": True
                }
            },
            {
                "model_id": "microsoft/DialoGPT-medium",
                "name": "DialoGPT Medium",
                "model_type": "conversational",
                "description": "Conversational AI model",
                "parameters": {
                    "max_new_tokens": 100,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            },
            {
                "model_id": "t5-small",
                "name": "T5 Small",
                "model_type": "text2text-generation",
                "description": "T5 model for text-to-text generation"
            }
        ]
        
        success_count = 0
        for model_data in models_to_add:
            try:
                response = requests.post(
                    f"{self.hf_api_base}/models",
                    json=model_data
                )
                data = response.json()
                success = response.status_code == 200
                if success:
                    success_count += 1
                self.log_test(f"Add Model: {model_data['name']}", success, data)
            except Exception as e:
                self.log_test(f"Add Model: {model_data['name']}", False, {}, str(e))
        
        return success_count == len(models_to_add)
    
    def test_list_models(self):
        """Test listing models."""
        try:
            # Test basic listing
            response = requests.get(f"{self.hf_api_base}/models")
            data = response.json()
            success = response.status_code == 200 and data.get("success", False)
            self.log_test("List All Models", success, data)
            
            # Test filtering by type
            response = requests.get(f"{self.hf_api_base}/models?type=text-generation")
            data = response.json()
            success = response.status_code == 200 and data.get("success", False)
            self.log_test("List Models by Type", success, data)
            
            return True
        except Exception as e:
            self.log_test("List Models", False, {}, str(e))
            return False
    
    def test_get_specific_model(self):
        """Test getting specific model information."""
        try:
            response = requests.get(f"{self.hf_api_base}/models/gpt2")
            data = response.json()
            success = response.status_code == 200 and data.get("success", False)
            self.log_test("Get Specific Model", success, data)
            return success
        except Exception as e:
            self.log_test("Get Specific Model", False, {}, str(e))
            return False
    
    def test_update_model(self):
        """Test updating model information."""
        update_data = {
            "description": "Updated GPT-2 model description",
            "parameters": {
                "max_new_tokens": 75,
                "temperature": 0.9,
                "top_k": 50
            }
        }
        
        try:
            response = requests.put(
                f"{self.hf_api_base}/models/gpt2",
                json=update_data
            )
            data = response.json()
            success = response.status_code == 200 and data.get("success", False)
            self.log_test("Update Model", success, data)
            return success
        except Exception as e:
            self.log_test("Update Model", False, {}, str(e))
            return False
    
    def test_load_model(self):
        """Test loading model into memory."""
        try:
            response = requests.post(f"{self.hf_api_base}/models/gpt2/load")
            data = response.json()
            success = response.status_code == 200
            self.log_test("Load Model", success, data)
            return success
        except Exception as e:
            self.log_test("Load Model", False, {}, str(e))
            return False
    
    def test_text_generation(self):
        """Test text generation endpoint."""
        generation_data = {
            "model_id": "gpt2",
            "prompt": "The future of artificial intelligence is",
            "max_new_tokens": 30,
            "temperature": 0.7,
            "do_sample": True
        }
        
        try:
            response = requests.post(
                f"{self.hf_api_base}/generate",
                json=generation_data
            )
            data = response.json()
            success = response.status_code == 200
            self.log_test("Text Generation", success, data)
            return success
        except Exception as e:
            self.log_test("Text Generation", False, {}, str(e))
            return False
    
    def test_pipeline_generation(self):
        """Test pipeline generation endpoint."""
        pipeline_data = {
            "model_id": "gpt2",
            "task": "text-generation",
            "inputs": "Once upon a time",
            "max_length": 50,
            "num_return_sequences": 1
        }
        
        try:
            response = requests.post(
                f"{self.hf_api_base}/pipeline",
                json=pipeline_data
            )
            data = response.json()
            success = response.status_code == 200
            self.log_test("Pipeline Generation", success, data)
            return success
        except Exception as e:
            self.log_test("Pipeline Generation", False, {}, str(e))
            return False
    
    def test_cache_operations(self):
        """Test cache-related operations."""
        try:
            # Get cache info
            response = requests.get(f"{self.hf_api_base}/cache")
            data = response.json()
            success = response.status_code == 200
            self.log_test("Get Cache Info", success, data)
            
            # Clear specific model cache
            response = requests.delete(f"{self.hf_api_base}/cache?model_id=gpt2")
            data = response.json()
            success = response.status_code == 200
            self.log_test("Clear Specific Cache", success, data)
            
            return True
        except Exception as e:
            self.log_test("Cache Operations", False, {}, str(e))
            return False
    
    def test_statistics(self):
        """Test statistics endpoint."""
        try:
            response = requests.get(f"{self.hf_api_base}/statistics")
            data = response.json()
            success = response.status_code == 200 and data.get("success", False)
            self.log_test("Get Statistics", success, data)
            return success
        except Exception as e:
            self.log_test("Get Statistics", False, {}, str(e))
            return False
    
    def test_model_types(self):
        """Test model types endpoint."""
        try:
            response = requests.get(f"{self.hf_api_base}/model-types")
            data = response.json()
            success = response.status_code == 200 and data.get("success", False)
            self.log_test("Get Model Types", success, data)
            return success
        except Exception as e:
            self.log_test("Get Model Types", False, {}, str(e))
            return False
    
    def test_error_cases(self):
        """Test error handling."""
        try:
            # Test adding duplicate model
            duplicate_model = {
                "model_id": "gpt2",
                "name": "Duplicate GPT-2",
                "model_type": "text-generation"
            }
            response = requests.post(f"{self.hf_api_base}/models", json=duplicate_model)
            success = response.status_code == 400  # Should fail
            self.log_test("Duplicate Model Error", success, response.json())
            
            # Test getting non-existent model
            response = requests.get(f"{self.hf_api_base}/models/non-existent-model")
            success = response.status_code == 404  # Should fail
            self.log_test("Non-existent Model Error", success, response.json())
            
            # Test generation with invalid model
            invalid_gen = {
                "model_id": "invalid-model",
                "prompt": "Test prompt"
            }
            response = requests.post(f"{self.hf_api_base}/generate", json=invalid_gen)
            success = response.status_code != 200  # Should fail
            self.log_test("Invalid Model Generation Error", success, response.json())
            
            return True
        except Exception as e:
            self.log_test("Error Cases", False, {}, str(e))
            return False
    
    def test_cleanup(self):
        """Clean up test data."""
        models_to_remove = ["gpt2", "microsoft/DialoGPT-medium", "t5-small"]
        
        for model_id in models_to_remove:
            try:
                response = requests.delete(f"{self.hf_api_base}/models/{model_id}")
                data = response.json()
                success = response.status_code == 200
                self.log_test(f"Remove Model: {model_id}", success, data)
            except Exception as e:
                self.log_test(f"Remove Model: {model_id}", False, {}, str(e))
    
    def run_all_tests(self, skip_generation=True):
        """Run all tests."""
        print("HuggingFace API Comprehensive Test Suite")
        print("=" * 50)
        
        # Check server availability
        try:
            response = requests.get(self.base_url)
            if response.status_code != 200:
                print("✗ Server not available. Please start the Flask application.")
                return False
        except Exception:
            print("✗ Cannot connect to server. Please start the Flask application.")
            return False
        
        # Run tests
        print("\n1. System Tests")
        print("-" * 20)
        self.test_dependencies()
        
        print("\n2. Model Management Tests")
        print("-" * 30)
        self.test_add_models()
        self.test_list_models()
        self.test_get_specific_model()
        self.test_update_model()
        
        print("\n3. Model Loading Tests")
        print("-" * 25)
        if not skip_generation:
            self.test_load_model()
        else:
            print("⏭ Skipping model loading (requires dependencies)")
        
        print("\n4. Text Generation Tests")
        print("-" * 28)
        if not skip_generation:
            self.test_text_generation()
            self.test_pipeline_generation()
        else:
            print("⏭ Skipping text generation (requires dependencies)")
        
        print("\n5. Cache and Statistics Tests")
        print("-" * 35)
        self.test_cache_operations()
        self.test_statistics()
        self.test_model_types()
        
        print("\n6. Error Handling Tests")
        print("-" * 25)
        self.test_error_cases()
        
        print("\n7. Cleanup")
        print("-" * 12)
        self.test_cleanup()
        
        # Summary
        print("\n" + "=" * 50)
        print("Test Summary")
        print("=" * 50)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests / total_tests * 100):.1f}%")
        
        if passed_tests == total_tests:
            print("\n🎉 All tests passed! The HuggingFace API is working correctly.")
        else:
            print(f"\n⚠️ {total_tests - passed_tests} tests failed. Check the output above for details.")
        
        # Print failed tests
        failed_tests = [result for result in self.test_results if not result["success"]]
        if failed_tests:
            print("\nFailed Tests:")
            for test in failed_tests:
                print(f"  - {test['test']}: {test['error']}")
        
        return passed_tests == total_tests

def main():
    """Main function to run tests."""
    tester = HuggingFaceAPITester()
    
    print("This script tests the HuggingFace API endpoints.")
    print("Make sure the Flask server is running on http://localhost:5000")
    print()
    
    choice = input("Skip text generation tests? (y/n) [y]: ").lower()
    skip_generation = choice != 'n'
    
    if skip_generation:
        print("Note: Skipping text generation tests (requires torch, transformers)")
        print("To test generation, install dependencies and choose 'n'")
    
    print()
    tester.run_all_tests(skip_generation=skip_generation)

if __name__ == "__main__":
    main()
