"""
Test script to verify HuggingFace integration is working.
"""
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from HuggingFace.hf_model_manager import HuggingFaceModelManager
from HuggingFace.hf_service import HuggingFaceService

class MockConfig:
    """Mock configuration for testing."""
    HF_FOLDER = './HuggingFace'
    HF_MODELS_JSON_FILE = './HuggingFace/hf_models_list.json'
    HF_CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')

def test_model_manager():
    """Test the HuggingFace model manager."""
    print("Testing HuggingFace Model Manager...")
    
    # Create model manager
    hf_manager = HuggingFaceModelManager(
        hf_folder='./HuggingFace',
        hf_models_json_file='./HuggingFace/hf_models_list.json'
    )
    
    # Test adding a model
    model_info = {
        "model_id": "gpt2",
        "name": "GPT-2 Test",
        "model_type": "text-generation",
        "description": "Test model for validation"
    }
    
    try:
        added_model = hf_manager.add_hf_model(model_info)
        print(f"✓ Successfully added model: {added_model['name']}")
    except Exception as e:
        print(f"✗ Error adding model: {e}")
        return False
    
    # Test listing models
    try:
        models = hf_manager.list_hf_models()
        print(f"✓ Successfully listed {len(models)} models")
    except Exception as e:
        print(f"✗ Error listing models: {e}")
        return False
    
    # Test getting model
    try:
        model = hf_manager.get_hf_model("gpt2")
        if model:
            print(f"✓ Successfully retrieved model: {model['name']}")
        else:
            print("✗ Model not found")
            return False
    except Exception as e:
        print(f"✗ Error getting model: {e}")
        return False
    
    # Test statistics
    try:
        stats = hf_manager.get_statistics()
        print(f"✓ Statistics: {stats['total_models']} models total")
    except Exception as e:
        print(f"✗ Error getting statistics: {e}")
        return False
    
    # Clean up test model
    try:
        hf_manager.remove_hf_model("gpt2")
        print("✓ Successfully removed test model")
    except Exception as e:
        print(f"✗ Error removing test model: {e}")
        return False
    
    return True

def test_service():
    """Test the HuggingFace service."""
    print("\nTesting HuggingFace Service...")
    
    config = MockConfig()
    hf_manager = HuggingFaceModelManager(
        hf_folder=config.HF_FOLDER,
        hf_models_json_file=config.HF_MODELS_JSON_FILE
    )
    
    hf_service = HuggingFaceService(config, hf_manager)
    
    # Test dependency check
    try:
        deps = hf_service.check_dependencies()
        print(f"✓ Dependencies check: transformers={deps['transformers']}, torch={deps['torch']}")
    except Exception as e:
        print(f"✗ Error checking dependencies: {e}")
        return False
    
    # Test cache info
    try:
        cache_info = hf_service.get_cache_info()
        print(f"✓ Cache info: {cache_info['total_cached']} models cached")
    except Exception as e:
        print(f"✗ Error getting cache info: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("HuggingFace Integration Test")
    print("=" * 40)
    
    success = True
    
    # Test model manager
    if not test_model_manager():
        success = False
    
    # Test service
    if not test_service():
        success = False
    
    print("\n" + "=" * 40)
    if success:
        print("✓ All tests passed! HuggingFace integration is working.")
    else:
        print("✗ Some tests failed. Check the errors above.")
    
    print("\nNext steps:")
    print("1. Start the Flask application: python main.py")
    print("2. Test the API endpoints using the example_usage.py script")
    print("3. Add your favorite HuggingFace models via the API")
