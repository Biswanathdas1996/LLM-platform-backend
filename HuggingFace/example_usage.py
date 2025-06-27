"""
Example script demonstrating how to use the HuggingFace model management system.
"""
import requests
import json

# Base URL for your local API
BASE_URL = "http://localhost:5000"

def check_dependencies():
    """Check if HuggingFace dependencies are available."""
    response = requests.get(f"{BASE_URL}/api/huggingface/dependencies")
    print("Dependencies check:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)

def add_model():
    """Add a new HuggingFace model."""
    model_data = {
        "model_id": "microsoft/DialoGPT-medium",
        "name": "DialoGPT Medium",
        "model_type": "conversational",
        "description": "A conversational AI model trained on Reddit conversations",
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/api/huggingface/models",
        json=model_data
    )
    print("Add model result:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)

def add_another_model():
    """Add another HuggingFace model for text generation."""
    model_data = {
        "model_id": "gpt2",
        "name": "GPT-2",
        "model_type": "text-generation",
        "description": "OpenAI GPT-2 language model",
        "parameters": {
            "max_new_tokens": 50,
            "temperature": 0.8,
            "do_sample": True
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/api/huggingface/models",
        json=model_data
    )
    print("Add GPT-2 model result:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)

def list_models():
    """List all HuggingFace models."""
    response = requests.get(f"{BASE_URL}/api/huggingface/models")
    print("List models:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)

def load_model():
    """Load a model into memory."""
    response = requests.post(
        f"{BASE_URL}/api/huggingface/models/gpt2/load"
    )
    print("Load model result:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)

def generate_text():
    """Generate text using a HuggingFace model."""
    generation_data = {
        "model_id": "gpt2",
        "prompt": "The future of artificial intelligence is",
        "max_new_tokens": 50,
        "temperature": 0.7
    }
    
    response = requests.post(
        f"{BASE_URL}/api/huggingface/generate",
        json=generation_data
    )
    print("Generate text result:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)

def use_pipeline():
    """Use HuggingFace pipeline for text generation."""
    pipeline_data = {
        "model_id": "gpt2",
        "task": "text-generation",
        "inputs": "Once upon a time in a land far away",
        "max_length": 100,
        "num_return_sequences": 1
    }
    
    response = requests.post(
        f"{BASE_URL}/api/huggingface/pipeline",
        json=pipeline_data
    )
    print("Pipeline result:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)

def get_statistics():
    """Get usage statistics."""
    response = requests.get(f"{BASE_URL}/api/huggingface/statistics")
    print("Statistics:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)

def get_cache_info():
    """Get cache information."""
    response = requests.get(f"{BASE_URL}/api/huggingface/cache")
    print("Cache info:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)

def update_model():
    """Update a model's information."""
    update_data = {
        "description": "Updated: OpenAI GPT-2 language model for text generation",
        "parameters": {
            "max_new_tokens": 75,
            "temperature": 0.9,
            "do_sample": True,
            "top_k": 50
        }
    }
    
    response = requests.put(
        f"{BASE_URL}/api/huggingface/models/gpt2",
        json=update_data
    )
    print("Update model result:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)

def remove_model():
    """Remove a model (uncomment to test)."""
    # response = requests.delete(f"{BASE_URL}/api/huggingface/models/microsoft/DialoGPT-medium")
    # print("Remove model result:")
    # print(json.dumps(response.json(), indent=2))
    # print("-" * 50)
    print("Remove model function is commented out for safety")
    print("-" * 50)

if __name__ == "__main__":
    print("HuggingFace Model Management Demo")
    print("=" * 50)
    
    try:
        # Check dependencies first
        check_dependencies()
        
        # Add models
        add_model()
        add_another_model()
        
        # List models
        list_models()
        
        # Load a model (this might take some time and require dependencies)
        # load_model()
        
        # Generate text (requires model to be loaded)
        # generate_text()
        
        # Use pipeline
        # use_pipeline()
        
        # Update model
        update_model()
        
        # Get statistics
        get_statistics()
        
        # Get cache info
        get_cache_info()
        
        # Remove model (commented for safety)
        remove_model()
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running.")
    except Exception as e:
        print(f"Error: {e}")

    print("\nNote: Text generation functions are commented out because they require:")
    print("1. The HuggingFace dependencies to be installed (torch, transformers)")
    print("2. Internet connection to download models")
    print("3. Sufficient memory and possibly GPU for model loading")
    print("\nTo enable text generation:")
    print("1. Install dependencies: pip install torch transformers tokenizers")
    print("2. Uncomment the generation function calls above")
    print("3. Run the script again")
