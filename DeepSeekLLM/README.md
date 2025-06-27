# DeepSeek LLM API Documentation

## Overview

The DeepSeek LLM API provides endpoints for text generation using DeepSeek models loaded locally via the llama-cpp-python library. This API allows you to interact with DeepSeek models through RESTful HTTP endpoints.

## Base URL

```
http://localhost:5000/api/deepseek
```

## Authentication

Currently, no authentication is required for the DeepSeek API endpoints.

## Content Type

All POST requests must include the following header:

```
Content-Type: application/json
```

## API Endpoints

### 1. Health Check

Check the status and availability of the DeepSeek service.

**Endpoint:** `GET /api/deepseek/health`

**Sample Request:**

```bash
curl -X GET http://localhost:5000/api/deepseek/health
```

**Sample Response:**

```json
{
  "success": true,
  "service": "DeepSeek LLM Service",
  "status": "healthy",
  "models_available": 4,
  "models_directory": "d:\\projects\\Local LLM\\DeepSeekLLM\\model",
  "models_directory_exists": true,
  "models": [
    "DeepSeek-R1-0528-Qwen3-8B-Q3_K_L.gguf",
    "DeepSeek-R1-q2_k.gguf",
    "DeepSeek-R1-q4_1.gguf",
    "DeepSeek-R1-q4_k_m.gguf"
  ]
}
```

**Error Response:**

```json
{
  "success": false,
  "service": "DeepSeek LLM Service",
  "status": "unhealthy",
  "error": "Models directory not found"
}
```

---

### 2. List Available Models

Get a list of all available DeepSeek models.

**Endpoint:** `GET /api/deepseek/models`

**Sample Request:**

```bash
curl -X GET http://localhost:5000/api/deepseek/models
```

**Sample Response:**

```json
{
  "success": true,
  "models": [
    {
      "name": "DeepSeek-R1-q2_k.gguf",
      "path": "d:\\projects\\Local LLM\\DeepSeekLLM\\model\\DeepSeek-R1-q2_k.gguf",
      "size": 2048576000,
      "size_mb": 1953.13,
      "type": "deepseek"
    },
    {
      "name": "DeepSeek-R1-q4_k_m.gguf",
      "path": "d:\\projects\\Local LLM\\DeepSeekLLM\\model\\DeepSeek-R1-q4_k_m.gguf",
      "size": 4096000000,
      "size_mb": 3906.25,
      "type": "deepseek"
    }
  ],
  "count": 2
}
```

**Error Response:**

```json
{
  "success": false,
  "error": "Error listing DeepSeek models: [error_message]"
}
```

---

### 3. Get Model Information

Get detailed information about a specific DeepSeek model.

**Endpoint:** `GET /api/deepseek/models/{model_name}`

**Parameters:**

- `model_name` (string): The name of the model file (e.g., "DeepSeek-R1-q2_k.gguf")

**Sample Request:**

```bash
curl -X GET http://localhost:5000/api/deepseek/models/DeepSeek-R1-q2_k.gguf
```

**Sample Response:**

```json
{
  "success": true,
  "name": "DeepSeek-R1-q2_k.gguf",
  "path": "d:\\projects\\Local LLM\\DeepSeekLLM\\model\\DeepSeek-R1-q2_k.gguf",
  "size": 2048576000,
  "size_mb": 1953.13,
  "type": "deepseek",
  "exists": true
}
```

**Error Response (Model Not Found):**

```json
{
  "success": false,
  "error": "Model 'NonExistentModel.gguf' not found"
}
```

---

### 4. Generate Text

Generate text using a specified DeepSeek model.

**Endpoint:** `POST /api/deepseek/generate`

**Required Parameters:**

- `model_name` (string): Name of the model to use
- `prompt` (string): Input text prompt for generation

**Optional Parameters:**

- `max_tokens` (integer): Maximum number of tokens to generate (default: 200)
- `temperature` (float): Sampling temperature for randomness (default: 0.7)
- `n_ctx` (integer): Context window size (default: 2048)
- `n_threads` (integer): Number of CPU threads to use (default: 8)
- `use_mlock` (boolean): Lock model in RAM for better performance (default: true)
- `stop` (array): List of stop sequences to end generation

**Sample Request (Full Parameters):**

```bash
curl -X POST http://localhost:5000/api/deepseek/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "DeepSeek-R1-q2_k.gguf",
    "prompt": "What is artificial intelligence?",
    "max_tokens": 200,
    "temperature": 0.7,
    "n_ctx": 2048,
    "n_threads": 8,
    "use_mlock": true,
    "stop": ["</s>", "<|end|>", "<|endoftext|>"]
  }'
```

**Sample Request (Minimal Parameters):**

```bash
curl -X POST http://localhost:5000/api/deepseek/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "DeepSeek-R1-q2_k.gguf",
    "prompt": "Hello, how are you today?"
  }'
```

**Sample Response (Success):**

```json
{
  "success": true,
  "model": "DeepSeek-R1-q2_k.gguf",
  "prompt": "What is artificial intelligence?",
  "generated_text": " Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. It involves creating computer systems capable of performing tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation.\n\nAI can be categorized into two main types:\n\n1. **Narrow AI (Weak AI)**: Designed to perform specific tasks, like voice assistants, recommendation systems, or image recognition.\n\n2. **General AI (Strong AI)**: Hypothetical AI that possesses human-level intelligence across all domains - this doesn't exist yet.\n\nAI technologies include machine learning, neural networks, natural language processing, and deep learning, which enable machines to process data, recognize patterns, and make intelligent decisions.",
  "parameters": {
    "max_tokens": 200,
    "temperature": 0.7,
    "n_ctx": 2048,
    "n_threads": 8,
    "stop": ["</s>", "<|end|>", "<|endoftext|>"]
  }
}
```

**Sample Response (Error):**

```json
{
  "success": false,
  "error": "Model 'NonExistentModel.gguf' not found",
  "model": "NonExistentModel.gguf",
  "prompt": "Hello"
}
```

**Validation Errors:**

```json
{
  "success": false,
  "error": "model_name is required"
}
```

```json
{
  "success": false,
  "error": "prompt is required"
}
```

```json
{
  "success": false,
  "error": "max_tokens must be a positive integer"
}
```

---

### 5. Test Generation

Quick test endpoint with default parameters for easy testing.

**Endpoint:** `POST /api/deepseek/test`

**Optional Parameters:**

- `model_name` (string): Model to test with (default: "DeepSeek-R1-q2_k.gguf")
- `prompt` (string): Test prompt (default: "Hello, how are you?")

**Sample Request (Default Test):**

```bash
curl -X POST http://localhost:5000/api/deepseek/test
```

**Sample Request (Custom Parameters):**

```bash
curl -X POST http://localhost:5000/api/deepseek/test \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "DeepSeek-R1-q4_k_m.gguf",
    "prompt": "Write a haiku about programming."
  }'
```

**Sample Response:**

```json
{
  "test": true,
  "default_model": "DeepSeek-R1-q2_k.gguf",
  "default_prompt": "Hello, how are you?",
  "success": true,
  "model": "DeepSeek-R1-q2_k.gguf",
  "prompt": "Hello, how are you?",
  "generated_text": " I'm doing well, thank you for asking! As an AI assistant, I don't have feelings in the traditional sense, but I'm functioning properly and ready to help you with any questions or tasks you might have. How can I assist you today?",
  "parameters": {
    "max_tokens": 50,
    "temperature": 0.7,
    "n_ctx": 2048,
    "n_threads": 8,
    "stop": ["</s>", "<|end|>", "<|endoftext|>"]
  }
}
```

---

## Integration Examples

### Python Integration

```python
import requests
import json

# Base URL for DeepSeek API
BASE_URL = "http://localhost:5000/api/deepseek"

def test_deepseek_health():
    """Test DeepSeek service health."""
    response = requests.get(f"{BASE_URL}/health")
    return response.json()

def list_deepseek_models():
    """Get list of available DeepSeek models."""
    response = requests.get(f"{BASE_URL}/models")
    return response.json()

def generate_text(model_name, prompt, **kwargs):
    """Generate text using DeepSeek model."""
    data = {
        "model_name": model_name,
        "prompt": prompt,
        **kwargs
    }

    response = requests.post(
        f"{BASE_URL}/generate",
        headers={"Content-Type": "application/json"},
        json=data
    )

    return response.json()

# Example usage
if __name__ == "__main__":
    # Check health
    health = test_deepseek_health()
    print("Health:", health)

    # List models
    models = list_deepseek_models()
    print("Available models:", len(models.get("models", [])))

    # Generate text
    result = generate_text(
        model_name="DeepSeek-R1-q2_k.gguf",
        prompt="Explain machine learning in simple terms.",
        max_tokens=150,
        temperature=0.8
    )

    if result.get("success"):
        print("Generated text:", result["generated_text"])
    else:
        print("Error:", result.get("error"))
```

### JavaScript/Node.js Integration

```javascript
const axios = require("axios");

const BASE_URL = "http://localhost:5000/api/deepseek";

class DeepSeekClient {
  constructor(baseUrl = BASE_URL) {
    this.baseUrl = baseUrl;
    this.client = axios.create({
      baseURL: this.baseUrl,
      headers: {
        "Content-Type": "application/json",
      },
    });
  }

  async healthCheck() {
    try {
      const response = await this.client.get("/health");
      return response.data;
    } catch (error) {
      throw new Error(`Health check failed: ${error.message}`);
    }
  }

  async listModels() {
    try {
      const response = await this.client.get("/models");
      return response.data;
    } catch (error) {
      throw new Error(`Failed to list models: ${error.message}`);
    }
  }

  async generateText(modelName, prompt, options = {}) {
    try {
      const data = {
        model_name: modelName,
        prompt: prompt,
        ...options,
      };

      const response = await this.client.post("/generate", data);
      return response.data;
    } catch (error) {
      throw new Error(`Text generation failed: ${error.message}`);
    }
  }

  async quickTest(prompt = null, modelName = null) {
    try {
      const data = {};
      if (prompt) data.prompt = prompt;
      if (modelName) data.model_name = modelName;

      const response = await this.client.post("/test", data);
      return response.data;
    } catch (error) {
      throw new Error(`Quick test failed: ${error.message}`);
    }
  }
}

// Example usage
async function main() {
  const client = new DeepSeekClient();

  try {
    // Health check
    const health = await client.healthCheck();
    console.log("Service health:", health.status);

    // Generate text
    const result = await client.generateText(
      "DeepSeek-R1-q2_k.gguf",
      "What are the benefits of renewable energy?",
      {
        max_tokens: 200,
        temperature: 0.7,
      }
    );

    if (result.success) {
      console.log("Generated text:", result.generated_text);
    } else {
      console.error("Generation failed:", result.error);
    }
  } catch (error) {
    console.error("Error:", error.message);
  }
}

module.exports = DeepSeekClient;
```

### cURL Automation Script (Bash)

```bash
#!/bin/bash

# DeepSeek API Test Script
BASE_URL="http://localhost:5000/api/deepseek"

echo "=== DeepSeek API Testing Script ==="

# Function to make API calls with error handling
api_call() {
    local method=$1
    local endpoint=$2
    local data=$3

    echo "Testing: $method $endpoint"

    if [ "$method" = "GET" ]; then
        curl -s -X GET "$BASE_URL$endpoint" | jq '.'
    elif [ "$method" = "POST" ]; then
        curl -s -X POST "$BASE_URL$endpoint" \
             -H "Content-Type: application/json" \
             -d "$data" | jq '.'
    fi

    echo ""
}

# Test health check
echo "1. Health Check"
api_call "GET" "/health"

# List models
echo "2. List Models"
api_call "GET" "/models"

# Quick test
echo "3. Quick Test"
api_call "POST" "/test" '{}'

# Generate text
echo "4. Generate Text"
api_call "POST" "/generate" '{
    "model_name": "DeepSeek-R1-q2_k.gguf",
    "prompt": "Write a short story about a robot learning to paint.",
    "max_tokens": 150,
    "temperature": 0.8
}'

echo "=== Testing Complete ==="
```

## Error Handling

All API responses include a `success` boolean field. When `success` is `false`, an `error` field will contain the error message.

### Common Error Codes

- **400 Bad Request**: Invalid parameters or missing required fields
- **404 Not Found**: Model not found
- **500 Internal Server Error**: Server error during processing
- **503 Service Unavailable**: Service health check failed

### Best Practices

1. **Always check the `success` field** in responses before processing data
2. **Handle network timeouts** - Model loading and text generation can take time
3. **Validate parameters** before making API calls
4. **Monitor model memory usage** - Large models require significant RAM
5. **Use appropriate thread counts** based on your system's CPU cores
6. **Set reasonable token limits** to prevent excessive generation times

## Performance Considerations

- **Model Loading**: First request to a model may take longer due to loading time
- **Memory Usage**: Models remain in memory after first use for better performance
- **CPU Threads**: Adjust `n_threads` based on your system (typically 4-8 threads)
- **Context Size**: Larger `n_ctx` values use more memory but allow longer conversations
- **Temperature**: Lower values (0.1-0.3) for focused responses, higher (0.7-1.0) for creative text

## Troubleshooting

### Model Not Found

- Ensure model files are in the `DeepSeekLLM/model/` directory
- Check file permissions
- Verify model file names match exactly (case-sensitive)

### Memory Issues

- Reduce `n_ctx` value
- Use smaller model variants (q2_k vs q4_k_m)
- Ensure sufficient system RAM

### Slow Performance

- Increase `n_threads` (but don't exceed CPU core count)
- Enable `use_mlock` for better performance
- Use SSD storage for model files

### Connection Issues

- Verify the API server is running on port 5000
- Check firewall settings
- Ensure correct base URL in client applications

## Model Variants Explanation

- **q2_k**: 2-bit quantization, smallest size, fastest inference, lower quality
- **q4_1**: 4-bit quantization, balanced size and quality
- **q4_k_m**: 4-bit quantization with improved quality, recommended for most use cases
- **Q3_K_L**: 3-bit quantization, large variant with better quality than standard 3-bit

Choose model variants based on your hardware capabilities and quality requirements.
