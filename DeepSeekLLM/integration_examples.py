"""
DeepSeek API Integration Examples

This file contains practical examples of how to integrate with the DeepSeek API
in different programming languages and scenarios.
"""

import requests
import json
import asyncio
import aiohttp
from typing import Dict, Any, Optional, List
import time

# Configuration
API_BASE_URL = "http://localhost:5000/api/deepseek"
DEFAULT_MODEL = "DeepSeek-R1-q2_k.gguf"

class DeepSeekClient:
    """
    A comprehensive client for the DeepSeek API with error handling,
    retry logic, and various helper methods.
    """
    
    def __init__(self, base_url: str = API_BASE_URL, timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the DeepSeek service is healthy."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Health check failed: {str(e)}"}
    
    def list_models(self) -> Dict[str, Any]:
        """Get list of available models."""
        try:
            response = self.session.get(f"{self.base_url}/models", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Failed to list models: {str(e)}"}
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        try:
            response = self.session.get(f"{self.base_url}/models/{model_name}", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Failed to get model info: {str(e)}"}
    
    def generate_text(
        self,
        prompt: str,
        model_name: str = DEFAULT_MODEL,
        max_tokens: int = 200,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text using the DeepSeek API."""
        data = {
            "model_name": model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/generate",
                json=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Text generation failed: {str(e)}"}
    
    def quick_test(self, prompt: str = None, model_name: str = None) -> Dict[str, Any]:
        """Quick test endpoint for basic functionality."""
        data = {}
        if prompt:
            data["prompt"] = prompt
        if model_name:
            data["model_name"] = model_name
        
        try:
            response = self.session.post(
                f"{self.base_url}/test",
                json=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Quick test failed: {str(e)}"}
    
    def generate_with_retry(
        self,
        prompt: str,
        model_name: str = DEFAULT_MODEL,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text with retry logic for better reliability."""
        for attempt in range(max_retries):
            result = self.generate_text(prompt, model_name, **kwargs)
            
            if result.get("success"):
                return result
            
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
        
        return result
    
    def batch_generate(
        self,
        prompts: List[str],
        model_name: str = DEFAULT_MODEL,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate text for multiple prompts sequentially."""
        results = []
        for i, prompt in enumerate(prompts):
            print(f"Processing prompt {i + 1}/{len(prompts)}")
            result = self.generate_text(prompt, model_name, **kwargs)
            results.append(result)
        return results
    
    def close(self):
        """Close the session."""
        self.session.close()

class AsyncDeepSeekClient:
    """
    Async version of the DeepSeek client for concurrent operations.
    """
    
    def __init__(self, base_url: str = API_BASE_URL, timeout: int = 30):
        self.base_url = base_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)
    
    async def generate_text(
        self,
        prompt: str,
        model_name: str = DEFAULT_MODEL,
        max_tokens: int = 200,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Async text generation."""
        data = {
            "model_name": model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    f"{self.base_url}/generate",
                    json=data,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    response.raise_for_status()
                    return await response.json()
        except Exception as e:
            return {"success": False, "error": f"Async generation failed: {str(e)}"}
    
    async def batch_generate_concurrent(
        self,
        prompts: List[str],
        model_name: str = DEFAULT_MODEL,
        max_concurrent: int = 3,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate text for multiple prompts concurrently with rate limiting."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_with_semaphore(prompt):
            async with semaphore:
                return await self.generate_text(prompt, model_name, **kwargs)
        
        tasks = [generate_with_semaphore(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)

# Example usage scenarios
def example_basic_usage():
    """Basic usage example."""
    print("=== Basic DeepSeek API Usage ===")
    
    client = DeepSeekClient()
    
    # Health check
    health = client.health_check()
    print(f"Service health: {health.get('status', 'unknown')}")
    
    # List models
    models = client.list_models()
    if models.get("success"):
        print(f"Available models: {len(models.get('models', []))}")
        for model in models.get("models", []):
            print(f"  - {model['name']} ({model['size_mb']} MB)")
    
    # Generate text
    result = client.generate_text(
        prompt="Explain quantum computing in simple terms.",
        max_tokens=150,
        temperature=0.5
    )
    
    if result.get("success"):
        print(f"\nGenerated text:\n{result['generated_text']}")
    else:
        print(f"Generation failed: {result.get('error')}")
    
    client.close()

def example_creative_writing():
    """Example for creative writing tasks."""
    print("\n=== Creative Writing Example ===")
    
    client = DeepSeekClient()
    
    creative_prompts = [
        "Write a haiku about programming.",
        "Create a short story about a time-traveling cat.",
        "Describe a futuristic city in 50 words."
    ]
    
    for prompt in creative_prompts:
        print(f"\nPrompt: {prompt}")
        result = client.generate_text(
            prompt=prompt,
            max_tokens=100,
            temperature=0.9,  # Higher temperature for creativity
            model_name=DEFAULT_MODEL
        )
        
        if result.get("success"):
            print(f"Response: {result['generated_text'][:200]}...")
        else:
            print(f"Error: {result.get('error')}")
    
    client.close()

def example_technical_qa():
    """Example for technical Q&A tasks."""
    print("\n=== Technical Q&A Example ===")
    
    client = DeepSeekClient()
    
    technical_questions = [
        "What is the difference between machine learning and deep learning?",
        "Explain the concept of REST APIs.",
        "What are the benefits of microservices architecture?"
    ]
    
    for question in technical_questions:
        print(f"\nQuestion: {question}")
        result = client.generate_text(
            prompt=question,
            max_tokens=200,
            temperature=0.3,  # Lower temperature for factual responses
            model_name=DEFAULT_MODEL
        )
        
        if result.get("success"):
            print(f"Answer: {result['generated_text']}")
        else:
            print(f"Error: {result.get('error')}")
    
    client.close()

def example_batch_processing():
    """Example of batch processing with retry logic."""
    print("\n=== Batch Processing Example ===")
    
    client = DeepSeekClient()
    
    prompts = [
        "What is artificial intelligence?",
        "Explain blockchain technology.",
        "What are the benefits of cloud computing?",
        "Describe the Internet of Things (IoT)."
    ]
    
    print(f"Processing {len(prompts)} prompts with retry logic...")
    
    results = []
    for i, prompt in enumerate(prompts):
        print(f"\nProcessing prompt {i + 1}: {prompt[:50]}...")
        result = client.generate_with_retry(
            prompt=prompt,
            max_tokens=100,
            temperature=0.5,
            max_retries=3
        )
        results.append(result)
        
        if result.get("success"):
            print("✓ Success")
        else:
            print(f"✗ Failed: {result.get('error')}")
    
    # Summary
    successful = sum(1 for r in results if r.get("success"))
    print(f"\nBatch processing complete: {successful}/{len(results)} successful")
    
    client.close()

async def example_async_processing():
    """Example of async processing for better performance."""
    print("\n=== Async Processing Example ===")
    
    client = AsyncDeepSeekClient()
    
    prompts = [
        "What is machine learning?",
        "Explain neural networks.",
        "What is natural language processing?",
        "Describe computer vision."
    ]
    
    print(f"Processing {len(prompts)} prompts concurrently...")
    
    start_time = time.time()
    results = await client.batch_generate_concurrent(
        prompts=prompts,
        max_tokens=100,
        temperature=0.5,
        max_concurrent=2  # Limit concurrent requests
    )
    end_time = time.time()
    
    successful = sum(1 for r in results if r.get("success"))
    print(f"Async processing complete: {successful}/{len(results)} successful")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    
    for i, result in enumerate(results):
        if result.get("success"):
            print(f"\nPrompt {i + 1}: {prompts[i][:50]}...")
            print(f"Response: {result['generated_text'][:100]}...")

def example_model_comparison():
    """Compare different models for the same prompt."""
    print("\n=== Model Comparison Example ===")
    
    client = DeepSeekClient()
    
    # Get available models
    models_response = client.list_models()
    if not models_response.get("success"):
        print("Failed to get models list")
        return
    
    available_models = [m["name"] for m in models_response.get("models", [])]
    
    if len(available_models) < 2:
        print("Need at least 2 models for comparison")
        return
    
    test_prompt = "Explain the concept of artificial intelligence."
    
    print(f"Comparing models for prompt: {test_prompt}")
    
    for model_name in available_models[:2]:  # Compare first 2 models
        print(f"\n--- Testing {model_name} ---")
        result = client.generate_text(
            prompt=test_prompt,
            model_name=model_name,
            max_tokens=150,
            temperature=0.7
        )
        
        if result.get("success"):
            print(f"Response: {result['generated_text'][:200]}...")
        else:
            print(f"Error: {result.get('error')}")
    
    client.close()

def example_error_handling():
    """Demonstrate error handling scenarios."""
    print("\n=== Error Handling Example ===")
    
    client = DeepSeekClient()
    
    # Test various error scenarios
    error_tests = [
        {
            "name": "Invalid model name",
            "params": {"prompt": "Hello", "model_name": "nonexistent-model.gguf"}
        },
        {
            "name": "Empty prompt",
            "params": {"prompt": "", "model_name": DEFAULT_MODEL}
        },
        {
            "name": "Invalid parameters",
            "params": {
                "prompt": "Hello",
                "model_name": DEFAULT_MODEL,
                "max_tokens": -1,
                "temperature": -0.5
            }
        }
    ]
    
    for test in error_tests:
        print(f"\nTesting: {test['name']}")
        result = client.generate_text(**test["params"])
        
        if result.get("success"):
            print("✓ Unexpected success")
        else:
            print(f"✗ Expected error: {result.get('error')}")
    
    client.close()

if __name__ == "__main__":
    """Run all examples."""
    print("DeepSeek API Integration Examples")
    print("=" * 50)
    
    # Check if service is available
    client = DeepSeekClient()
    health = client.health_check()
    
    if not health.get("success"):
        print(f"❌ DeepSeek service is not available: {health.get('error')}")
        print("Please make sure the API server is running on http://localhost:5000")
        exit(1)
    
    print(f"✅ DeepSeek service is {health.get('status')}")
    client.close()
    
    # Run examples
    try:
        example_basic_usage()
        example_creative_writing()
        example_technical_qa()
        example_batch_processing()
        example_model_comparison()
        example_error_handling()
        
        # Run async example
        print("\n" + "=" * 50)
        asyncio.run(example_async_processing())
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user")
    except Exception as e:
        print(f"\n\nError running examples: {e}")
    
    print("\n" + "=" * 50)
    print("Examples completed!")
