# HuggingFace Model Management

This module provides comprehensive management and usage of HuggingFace models within the Local LLM application.

## Features

- **Model Registry**: Store and manage HuggingFace model metadata in JSON format
- **Model Loading**: Load models into memory with caching support
- **Text Generation**: Generate text using both direct model calls and pipelines
- **Usage Tracking**: Track model usage statistics and performance
- **RESTful API**: Complete API endpoints for all operations
- **Dependency Management**: Check and manage required dependencies

## Installation

### Required Dependencies

```bash
pip install torch transformers tokenizers safetensors accelerate datasets
```

### Optional Dependencies (for better performance)

```bash
# For GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For faster inference
pip install optimum
```

## Quick Start

### 1. Add a Model

```bash
curl -X POST http://localhost:5000/api/huggingface/models \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "gpt2",
    "name": "GPT-2",
    "model_type": "text-generation",
    "description": "OpenAI GPT-2 language model",
    "parameters": {
      "max_new_tokens": 50,
      "temperature": 0.8
    }
  }'
```

### 2. List Models

```bash
curl http://localhost:5000/api/huggingface/models
```

### 3. Generate Text

```bash
curl -X POST http://localhost:5000/api/huggingface/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "gpt2",
    "prompt": "The future of AI is",
    "max_new_tokens": 50
  }'
```

## API Endpoints

### Model Management

| Method | Endpoint                                  | Description            |
| ------ | ----------------------------------------- | ---------------------- |
| GET    | `/api/huggingface/models`                 | List all models        |
| POST   | `/api/huggingface/models`                 | Add new model          |
| GET    | `/api/huggingface/models/<model_id>`      | Get specific model     |
| PUT    | `/api/huggingface/models/<model_id>`      | Update model           |
| DELETE | `/api/huggingface/models/<model_id>`      | Remove model           |
| POST   | `/api/huggingface/models/<model_id>/load` | Load model into memory |

### Text Generation

| Method | Endpoint                    | Description               |
| ------ | --------------------------- | ------------------------- |
| POST   | `/api/huggingface/generate` | Generate text using model |
| POST   | `/api/huggingface/pipeline` | Generate using pipeline   |

### System

| Method | Endpoint                        | Description               |
| ------ | ------------------------------- | ------------------------- |
| GET    | `/api/huggingface/cache`        | Get cache information     |
| DELETE | `/api/huggingface/cache`        | Clear model cache         |
| GET    | `/api/huggingface/statistics`   | Get usage statistics      |
| GET    | `/api/huggingface/model-types`  | Get available model types |
| GET    | `/api/huggingface/dependencies` | Check dependencies        |

## Model Types Supported

- **text-generation**: General text generation models (GPT-2, GPT-3, etc.)
- **conversational**: Dialogue models (DialoGPT, BlenderBot, etc.)
- **text2text-generation**: Sequence-to-sequence models (T5, BART, etc.)
- **summarization**: Text summarization models
- **translation**: Language translation models
- **question-answering**: QA models
- **fill-mask**: Masked language models (BERT, RoBERTa, etc.)

## Configuration

Add to your environment variables or config:

```python
# HuggingFace settings
HF_FOLDER = './HuggingFace'
HF_MODELS_JSON_FILE = './HuggingFace/hf_models_list.json'
HF_CACHE_DIR = '~/.cache/huggingface'
```

## Usage Examples

### Python Client

```python
import requests

# Add a model
response = requests.post('http://localhost:5000/api/huggingface/models', json={
    "model_id": "microsoft/DialoGPT-medium",
    "name": "DialoGPT Medium",
    "model_type": "conversational",
    "description": "Conversational AI model"
})

# Generate text
response = requests.post('http://localhost:5000/api/huggingface/generate', json={
    "model_id": "microsoft/DialoGPT-medium",
    "prompt": "Hello, how are you?",
    "max_new_tokens": 50
})

print(response.json()['response'])
```

### Using Pipeline

```python
# Use pipeline for specific tasks
response = requests.post('http://localhost:5000/api/huggingface/pipeline', json={
    "model_id": "facebook/bart-large-cnn",
    "task": "summarization",
    "inputs": "Long text to summarize...",
    "max_length": 130,
    "min_length": 30
})
```

## Model Storage Format

Models are stored in JSON format with the following structure:

```json
{
  "models": [
    {
      "model_id": "gpt2",
      "name": "GPT-2",
      "model_type": "text-generation",
      "description": "OpenAI GPT-2 language model",
      "parameters": {
        "max_new_tokens": 50,
        "temperature": 0.8
      },
      "added_date": "2025-06-26T21:45:00.000000",
      "last_used": "2025-06-26T22:00:00.000000",
      "usage_count": 5,
      "status": "available"
    }
  ],
  "last_updated": "2025-06-26T22:00:00.000000",
  "count": 1
}
```

## Performance Tips

1. **GPU Usage**: Ensure CUDA is available for faster inference
2. **Model Caching**: Models are cached in memory to avoid reloading
3. **Batch Processing**: Use pipelines for batch processing tasks
4. **Memory Management**: Clear cache when not needed to free memory

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Install required dependencies
2. **CUDA Out of Memory**: Reduce model size or clear cache
3. **Model Not Found**: Check if model exists on HuggingFace Hub
4. **Network Issues**: Ensure internet connection for model downloads

### Debug Mode

Enable debug logging in your config:

```python
DEBUG = True
```

## Contributing

1. Follow the existing code structure
2. Add tests for new functionality
3. Update documentation
4. Ensure backward compatibility

## License

Same as the main Local LLM application.
