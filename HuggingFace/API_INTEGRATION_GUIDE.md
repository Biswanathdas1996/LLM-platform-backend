# HuggingFace API Integration Documentation

## Overview

This document provides comprehensive documentation for integrating with the HuggingFace API endpoints in the Local LLM application. The HuggingFace module provides a complete REST API for managing and using HuggingFace models.

## Base URL

```
http://localhost:5000/api/huggingface
```

## Authentication

Currently, no authentication is required for local development. For production deployment, consider adding API key authentication.

## Content Type

All requests should use `Content-Type: application/json` for POST/PUT requests.

---

## API Endpoints Reference

### 1. Model Management

#### 1.1 List All Models

**GET** `/api/huggingface/models`

Lists all registered HuggingFace models.

**Query Parameters:**

- `type` (optional): Filter by model type (e.g., "text-generation", "conversational")

**Example Request:**

```bash
curl -X GET "http://localhost:5000/api/huggingface/models?type=text-generation"
```

**Example Response:**

```json
{
  "success": true,
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
      "last_used": null,
      "usage_count": 0,
      "status": "available"
    }
  ],
  "count": 1
}
```

#### 1.2 Add New Model

**POST** `/api/huggingface/models`

Adds a new HuggingFace model to the registry.

**Request Body:**

```json
{
  "model_id": "microsoft/DialoGPT-medium",
  "name": "DialoGPT Medium",
  "model_type": "conversational",
  "description": "A conversational AI model trained on Reddit conversations",
  "parameters": {
    "max_new_tokens": 100,
    "temperature": 0.7,
    "do_sample": true,
    "top_p": 0.9
  }
}
```

**Required Fields:**

- `model_id`: HuggingFace model identifier
- `name`: Display name for the model
- `model_type`: Type of model (see supported types below)

**Optional Fields:**

- `description`: Model description
- `parameters`: Default generation parameters

**Example Request:**

```bash
curl -X POST "http://localhost:5000/api/huggingface/models" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "gpt2",
    "name": "GPT-2",
    "model_type": "text-generation",
    "description": "OpenAI GPT-2 language model"
  }'
```

**Example Response:**

```json
{
  "success": true,
  "model": {
    "model_id": "gpt2",
    "name": "GPT-2",
    "model_type": "text-generation",
    "description": "OpenAI GPT-2 language model",
    "parameters": {},
    "added_date": "2025-06-26T21:45:00.000000",
    "last_used": null,
    "usage_count": 0,
    "status": "available"
  },
  "message": "Model 'gpt2' added successfully"
}
```

#### 1.3 Get Specific Model

**GET** `/api/huggingface/models/{model_id}`

Retrieves information about a specific model.

**Path Parameters:**

- `model_id`: The HuggingFace model identifier (URL encoded)

**Example Request:**

```bash
curl -X GET "http://localhost:5000/api/huggingface/models/gpt2"
```

**Example Response:**

```json
{
  "success": true,
  "model": {
    "model_id": "gpt2",
    "name": "GPT-2",
    "model_type": "text-generation",
    "description": "OpenAI GPT-2 language model",
    "parameters": {
      "max_new_tokens": 50,
      "temperature": 0.8
    },
    "added_date": "2025-06-26T21:45:00.000000",
    "last_used": "2025-06-26T22:30:00.000000",
    "usage_count": 5,
    "status": "available"
  }
}
```

#### 1.4 Update Model

**PUT** `/api/huggingface/models/{model_id}`

Updates model information (excluding protected fields).

**Path Parameters:**

- `model_id`: The HuggingFace model identifier

**Request Body:**

```json
{
  "description": "Updated description",
  "parameters": {
    "max_new_tokens": 75,
    "temperature": 0.9,
    "top_k": 50
  }
}
```

**Protected Fields (cannot be updated):**

- `model_id`
- `added_date`
- `usage_count`

**Example Request:**

```bash
curl -X PUT "http://localhost:5000/api/huggingface/models/gpt2" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Updated OpenAI GPT-2 language model",
    "parameters": {
      "max_new_tokens": 75,
      "temperature": 0.9
    }
  }'
```

#### 1.5 Remove Model

**DELETE** `/api/huggingface/models/{model_id}`

Removes a model from the registry.

**Path Parameters:**

- `model_id`: The HuggingFace model identifier

**Example Request:**

```bash
curl -X DELETE "http://localhost:5000/api/huggingface/models/gpt2"
```

**Example Response:**

```json
{
  "success": true,
  "message": "Model 'gpt2' removed successfully",
  "models_data": {
    "models": [],
    "last_updated": "2025-06-26T22:45:00.000000",
    "count": 0
  }
}
```

#### 1.6 Load Model into Memory

**POST** `/api/huggingface/models/{model_id}/load`

Loads a model into memory for faster inference.

**Path Parameters:**

- `model_id`: The HuggingFace model identifier

**Request Body (optional):**

```json
{
  "force_reload": false
}
```

**Example Request:**

```bash
curl -X POST "http://localhost:5000/api/huggingface/models/gpt2/load" \
  -H "Content-Type: application/json" \
  -d '{"force_reload": true}'
```

**Example Response:**

```json
{
  "success": true,
  "message": "Model 'gpt2' loaded successfully",
  "model_info": {
    "model_id": "gpt2",
    "name": "GPT-2",
    "model_type": "text-generation"
  },
  "device": "cuda"
}
```

#### 1.7 List GGUF Files

**GET** `/api/huggingface/models/{model_id}/gguf`

Lists available GGUF files for a model on HuggingFace Hub.

**Path Parameters:**

- `model_id`: The HuggingFace model identifier

**Example Request:**

```bash
curl -X GET "http://localhost:5000/api/huggingface/models/microsoft/Phi-3-mini-4k-instruct-gguf/gguf"
```

**Example Response:**

```json
{
  "success": true,
  "model_id": "microsoft/Phi-3-mini-4k-instruct-gguf",
  "gguf_files": [
    "Phi-3-mini-4k-instruct-q4.gguf",
    "Phi-3-mini-4k-instruct-fp16.gguf",
    "Phi-3-mini-4k-instruct-q8_0.gguf"
  ],
  "total_gguf_files": 3
}
```

#### 1.8 Download GGUF Model

**POST** `/api/huggingface/models/{model_id}/gguf/download`

Downloads a GGUF format model from HuggingFace Hub.

**Path Parameters:**

- `model_id`: The HuggingFace model identifier

**Request Body (optional):**

```json
{
  "filename": "Phi-3-mini-4k-instruct-q4.gguf",
  "download_path": "./models"
}
```

**Optional Fields:**

- `filename`: Specific GGUF file to download (if not provided, downloads first available)
- `download_path`: Custom download path (uses config default if not provided)

**Example Request:**

```bash
curl -X POST "http://localhost:5000/api/huggingface/models/microsoft/Phi-3-mini-4k-instruct-gguf/gguf/download" \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "Phi-3-mini-4k-instruct-q4.gguf"
  }'
```

**Example Response:**

```json
{
  "success": true,
  "message": "GGUF model 'Phi-3-mini-4k-instruct-q4.gguf' downloaded successfully",
  "model_info": {
    "model_id": "microsoft/Phi-3-mini-4k-instruct-gguf",
    "name": "Phi-3 Mini GGUF",
    "model_type": "text-generation"
  },
  "download_info": {
    "downloaded_at": "2025-06-26T23:00:00.000000",
    "gguf_file": "Phi-3-mini-4k-instruct-q4.gguf",
    "local_path": "./models/Phi-3-mini-4k-instruct-q4.gguf",
    "available_gguf_files": [
      "Phi-3-mini-4k-instruct-q4.gguf",
      "Phi-3-mini-4k-instruct-fp16.gguf",
      "Phi-3-mini-4k-instruct-q8_0.gguf"
    ]
  }
}
```

---

### 2. Text Generation

#### 2.1 Generate Text

**POST** `/api/huggingface/generate`

Generates text using a HuggingFace model.

**Request Body:**

```json
{
  "model_id": "gpt2",
  "prompt": "The future of artificial intelligence is",
  "max_new_tokens": 50,
  "temperature": 0.7,
  "do_sample": true,
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.1
}
```

**Required Fields:**

- `model_id`: The HuggingFace model identifier
- `prompt`: Input text prompt

**Optional Generation Parameters:**

- `max_new_tokens`: Maximum number of tokens to generate (default: 100)
- `temperature`: Sampling temperature (default: 0.7)
- `do_sample`: Whether to use sampling (default: true)
- `top_p`: Nucleus sampling parameter (default: 0.9)
- `top_k`: Top-k sampling parameter (default: 50)
- `repetition_penalty`: Repetition penalty (default: 1.1)

**Example Request:**

```bash
curl -X POST "http://localhost:5000/api/huggingface/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "gpt2",
    "prompt": "The future of artificial intelligence is",
    "max_new_tokens": 50,
    "temperature": 0.7
  }'
```

**Example Response:**

```json
{
  "success": true,
  "response": "bright and full of possibilities. With advances in machine learning and neural networks, we can expect to see more intelligent systems that can understand and interact with humans in natural ways.",
  "model_used": "gpt2",
  "prompt": "The future of artificial intelligence is",
  "parameters": {
    "max_new_tokens": 50,
    "temperature": 0.7,
    "do_sample": true,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1
  },
  "model_info": {
    "model_id": "gpt2",
    "name": "GPT-2",
    "model_type": "text-generation"
  }
}
```

#### 2.2 Generate with Pipeline

**POST** `/api/huggingface/pipeline`

Generates content using HuggingFace pipelines for specific tasks.

**Request Body:**

```json
{
  "model_id": "facebook/bart-large-cnn",
  "task": "summarization",
  "inputs": "Long article text to summarize...",
  "max_length": 130,
  "min_length": 30,
  "do_sample": false
}
```

**Required Fields:**

- `model_id`: The HuggingFace model identifier
- `task`: Task type (see supported tasks below)
- `inputs`: Input data for the task

**Supported Tasks:**

- `text-generation`: Generate text
- `summarization`: Summarize text
- `translation`: Translate text
- `question-answering`: Answer questions
- `fill-mask`: Fill masked tokens
- `sentiment-analysis`: Analyze sentiment

**Example Request:**

```bash
curl -X POST "http://localhost:5000/api/huggingface/pipeline" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "gpt2",
    "task": "text-generation",
    "inputs": "Once upon a time",
    "max_length": 100,
    "num_return_sequences": 1
  }'
```

**Example Response:**

```json
{
  "success": true,
  "results": [
    {
      "generated_text": "Once upon a time, in a land far away, there lived a wise old wizard who possessed magical powers beyond imagination."
    }
  ],
  "model_used": "gpt2",
  "task": "text-generation",
  "inputs": "Once upon a time",
  "model_info": {
    "model_id": "gpt2",
    "name": "GPT-2",
    "model_type": "text-generation"
  }
}
```

---

### 3. System & Cache Management

#### 3.1 Get Cache Information

**GET** `/api/huggingface/cache`

Retrieves information about cached models.

**Example Request:**

```bash
curl -X GET "http://localhost:5000/api/huggingface/cache"
```

**Example Response:**

```json
{
  "success": true,
  "cached_models": {
    "gpt2": {
      "loaded_at": "2025-06-26T22:30:00.000000",
      "device": "cuda",
      "model_type": "text-generation"
    }
  },
  "cached_pipelines": ["gpt2_text-generation"],
  "total_cached": 1,
  "cache_directories": {
    "hf_cache_dir": "C:\\Users\\username\\.cache\\huggingface",
    "transformers_cache": "C:\\Users\\username\\.cache\\huggingface\\transformers",
    "cache_exists": true
  }
}
```

#### 3.2 Clear Cache

**DELETE** `/api/huggingface/cache`

Clears model cache to free memory.

**Query Parameters:**

- `model_id` (optional): Clear cache for specific model only

**Example Request:**

```bash
# Clear all cache
curl -X DELETE "http://localhost:5000/api/huggingface/cache"

# Clear specific model cache
curl -X DELETE "http://localhost:5000/api/huggingface/cache?model_id=gpt2"
```

**Example Response:**

```json
{
  "success": true,
  "message": "All caches cleared"
}
```

#### 3.3 Get Statistics

**GET** `/api/huggingface/statistics`

Retrieves usage statistics for HuggingFace models.

**Example Request:**

```bash
curl -X GET "http://localhost:5000/api/huggingface/statistics"
```

**Example Response:**

```json
{
  "success": true,
  "statistics": {
    "total_models": 3,
    "total_usage": 15,
    "model_types": {
      "text-generation": 2,
      "conversational": 1
    },
    "last_updated": "2025-06-26T22:30:00.000000",
    "most_used": {
      "model_id": "gpt2",
      "name": "GPT-2",
      "usage_count": 10
    }
  }
}
```

#### 3.4 Get Model Types

**GET** `/api/huggingface/model-types`

Retrieves list of unique model types in the registry.

**Example Request:**

```bash
curl -X GET "http://localhost:5000/api/huggingface/model-types"
```

**Example Response:**

```json
{
  "success": true,
  "model_types": ["conversational", "text-generation", "text2text-generation"]
}
```

#### 3.5 Check Dependencies

**GET** `/api/huggingface/dependencies`

Checks if required dependencies are installed and available.

**Example Request:**

```bash
curl -X GET "http://localhost:5000/api/huggingface/dependencies"
```

**Example Response:**

```json
{
  "success": true,
  "dependencies": {
    "transformers": true,
    "torch": true,
    "cuda_available": true
  }
}
```

---

## Supported Model Types

| Type                   | Description              | Example Models       |
| ---------------------- | ------------------------ | -------------------- |
| `text-generation`      | General text generation  | GPT-2, GPT-3, OPT    |
| `conversational`       | Dialogue/chat models     | DialoGPT, BlenderBot |
| `text2text-generation` | Sequence-to-sequence     | T5, BART, Pegasus    |
| `summarization`        | Text summarization       | BART-large-CNN, T5   |
| `translation`          | Language translation     | MarianMT, T5         |
| `question-answering`   | QA systems               | BERT, DistilBERT     |
| `fill-mask`            | Masked language models   | BERT, RoBERTa        |
| `sentiment-analysis`   | Sentiment classification | DistilBERT, RoBERTa  |

---

## Error Handling

All endpoints return consistent error responses:

### Error Response Format

```json
{
  "success": false,
  "error": "Error description"
}
```

### Common HTTP Status Codes

- `200 OK`: Successful request
- `400 Bad Request`: Invalid request data
- `404 Not Found`: Model not found
- `500 Internal Server Error`: Server error

### Example Error Responses

**Model Not Found (404):**

```json
{
  "success": false,
  "error": "Model 'invalid-model' not found"
}
```

**Missing Required Field (400):**

```json
{
  "success": false,
  "error": "Missing required field: model_id"
}
```

**Dependencies Not Available (400):**

```json
{
  "success": false,
  "error": "Transformers library not available. Please install: pip install transformers torch"
}
```

---

## Rate Limiting & Performance

### Best Practices

1. **Model Caching**: Load models once and reuse for multiple requests
2. **Batch Processing**: Use pipelines for batch operations
3. **Memory Management**: Clear cache when not needed
4. **GPU Usage**: Ensure CUDA is available for better performance

### Performance Tips

```bash
# Pre-load frequently used models
curl -X POST "http://localhost:5000/api/huggingface/models/gpt2/load"

# Use appropriate parameters for your use case
curl -X POST "http://localhost:5000/api/huggingface/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "gpt2",
    "prompt": "Hello",
    "max_new_tokens": 20,
    "do_sample": false
  }'

# Clear cache to free memory when done
curl -X DELETE "http://localhost:5000/api/huggingface/cache"
```

---

## Example Integration Workflows

### 1. Basic Text Generation Workflow

```bash
# 1. Check dependencies
curl -X GET "http://localhost:5000/api/huggingface/dependencies"

# 2. Add a model
curl -X POST "http://localhost:5000/api/huggingface/models" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "gpt2",
    "name": "GPT-2",
    "model_type": "text-generation"
  }'

# 3. Generate text
curl -X POST "http://localhost:5000/api/huggingface/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "gpt2",
    "prompt": "Hello, how are you?",
    "max_new_tokens": 50
  }'
```

### 2. Model Management Workflow

```bash
# 1. List all models
curl -X GET "http://localhost:5000/api/huggingface/models"

# 2. Get specific model info
curl -X GET "http://localhost:5000/api/huggingface/models/gpt2"

# 3. Update model parameters
curl -X PUT "http://localhost:5000/api/huggingface/models/gpt2" \
  -H "Content-Type: application/json" \
  -d '{
    "parameters": {
      "temperature": 0.9,
      "max_new_tokens": 100
    }
  }'

# 4. Get usage statistics
curl -X GET "http://localhost:5000/api/huggingface/statistics"
```

### 3. GGUF Model Download Workflow

```bash
# 1. Add a GGUF model to registry
curl -X POST "http://localhost:5000/api/huggingface/models" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "microsoft/Phi-3-mini-4k-instruct-gguf",
    "name": "Phi-3 Mini GGUF",
    "model_type": "text-generation"
  }'

# 2. List available GGUF files
curl -X GET "http://localhost:5000/api/huggingface/models/microsoft/Phi-3-mini-4k-instruct-gguf/gguf"

# 3. Download specific GGUF file
curl -X POST "http://localhost:5000/api/huggingface/models/microsoft/Phi-3-mini-4k-instruct-gguf/gguf/download" \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "Phi-3-mini-4k-instruct-q4.gguf",
    "download_path": "./models"
  }'
```

---

## SDK Examples

### Python Client Example

```python
import requests
import json

class HuggingFaceClient:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = f"{base_url}/api/huggingface"

    def add_model(self, model_id, name, model_type, **kwargs):
        data = {
            "model_id": model_id,
            "name": name,
            "model_type": model_type,
            **kwargs
        }
        response = requests.post(f"{self.base_url}/models", json=data)
        return response.json()

    def generate_text(self, model_id, prompt, **params):
        data = {
            "model_id": model_id,
            "prompt": prompt,
            **params
        }
        response = requests.post(f"{self.base_url}/generate", json=data)
        return response.json()

    def list_models(self, model_type=None):
        params = {"type": model_type} if model_type else {}
        response = requests.get(f"{self.base_url}/models", params=params)
        return response.json()

# Usage example
client = HuggingFaceClient()

# Add a model
result = client.add_model(
    model_id="gpt2",
    name="GPT-2",
    model_type="text-generation"
)

# Generate text
result = client.generate_text(
    model_id="gpt2",
    prompt="The future is",
    max_new_tokens=50
)

print(result["response"])
```

### JavaScript/Node.js Example

```javascript
class HuggingFaceClient {
  constructor(baseUrl = "http://localhost:5000") {
    this.baseUrl = `${baseUrl}/api/huggingface`;
  }

  async addModel(modelId, name, modelType, options = {}) {
    const response = await fetch(`${this.baseUrl}/models`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_id: modelId,
        name,
        model_type: modelType,
        ...options,
      }),
    });
    return response.json();
  }

  async generateText(modelId, prompt, params = {}) {
    const response = await fetch(`${this.baseUrl}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_id: modelId,
        prompt,
        ...params,
      }),
    });
    return response.json();
  }
}

// Usage
const client = new HuggingFaceClient();

// Generate text
client
  .generateText("gpt2", "Hello world", { max_new_tokens: 30 })
  .then((result) => console.log(result.response));
```

---

## Troubleshooting

### Common Issues

1. **Connection Refused**: Ensure the Flask server is running on port 5000
2. **Model Not Found**: Check if the model is added to the registry
3. **Memory Issues**: Clear cache or use smaller models
4. **Dependencies Missing**: Install transformers and torch packages

### Debug Commands

```bash
# Check server status
curl -X GET "http://localhost:5000/"

# Check dependencies
curl -X GET "http://localhost:5000/api/huggingface/dependencies"

# Check cache usage
curl -X GET "http://localhost:5000/api/huggingface/cache"
```

---

This documentation provides a complete reference for integrating with the HuggingFace APIs. For additional examples and code samples, see the `example_usage.py` script in the HuggingFace module.

---

## Model Storage Locations

When you download and use HuggingFace models, they are stored in the following locations:

### 1. HuggingFace Cache Directory (Model Files)

**Default Location:**

- **Windows**: `C:\Users\{username}\.cache\huggingface\transformers\`
- **Linux/Mac**: `~/.cache/huggingface/transformers/`

**Customizable via Environment Variable:**

```bash
# Set custom cache directory
export HF_CACHE_DIR="/path/to/your/custom/cache"
```

### 2. Application Registry (Model Metadata)

Model registry information is stored in:

```
./HuggingFace/hf_models_list.json
```

### 3. Memory Cache (Runtime Only)

Active models are cached in memory for faster inference during runtime.

### Check Cache Information

Use the cache endpoint to see where models are stored:

```bash
curl -X GET "http://localhost:5000/api/huggingface/cache"
```

This will return information including the actual cache directory paths being used.

---
