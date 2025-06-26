# Local LLM API Documentation

## Overview

The Local LLM API provides endpoints for managing and interacting with local language models. This RESTful API allows you to upload models, generate text responses, and manage model caching.

**Base URL:** `http://127.0.0.1:5000/api/v1`

---

## Endpoints

### 1. Health Check

**Summary:** Check if the API service is running and healthy.

**Endpoint:** `GET /api/v1/health`

**CURL:**

```bash
curl -X GET "http://127.0.0.1:5000/api/v1/health"
```

**Expected Response:**

```json
{
  "status": "healthy",
  "service": "Local LLM API",
  "version": "1.0.0"
}
```

**Status Code:** `200 OK`

---

### 2. List Models

**Summary:** Retrieve a list of all available models in the system.

**Endpoint:** `GET /api/v1/models`

**CURL:**

```bash
curl -X GET "http://127.0.0.1:5000/api/v1/models"
```

**Expected Response:**

```json
{
  "models": [
    {
      "name": "gpt4all-falcon-newbpe-q4_0.gguf",
      "path": "./models/gpt4all-falcon-newbpe-q4_0.gguf",
      "size": "4.1 GB",
      "modified": "2024-01-15T10:30:00Z"
    },
    {
      "name": "Llama-3.2-3B-Instruct-Q4_0.gguf",
      "path": "./models/Llama-3.2-3B-Instruct-Q4_0.gguf",
      "size": "2.8 GB",
      "modified": "2024-01-10T14:25:00Z"
    }
  ],
  "count": 2
}
```

**Status Codes:**

- `200 OK` - Success
- `500 Internal Server Error` - Failed to list models

**Error Response:**

```json
{
  "error": "Failed to list models: [error message]"
}
```

---

### 3. Upload Model

**Summary:** Upload a new model file (.gguf format) to the system.

**Endpoint:** `POST /api/v1/models`

**CURL:**

```bash
curl -X POST "http://127.0.0.1:5000/api/v1/models" \
  -F "file=@/path/to/your/model.gguf"
```

**Expected Response (Success):**

```json
{
  "message": "File uploaded successfully",
  "filename": "model.gguf",
  "model_info": {
    "name": "model.gguf",
    "path": "./models/model.gguf",
    "size": "3.2 GB",
    "modified": "2024-01-20T09:15:00Z"
  }
}
```

**Status Codes:**

- `200 OK` - Upload successful
- `400 Bad Request` - No file provided or invalid file type
- `409 Conflict` - File already exists
- `500 Internal Server Error` - Upload failed

**Error Responses:**

```json
{
  "error": "No file part in the request"
}
```

```json
{
  "error": "No file selected"
}
```

```json
{
  "error": "Invalid file type. Only gguf files are allowed"
}
```

```json
{
  "error": "File already exists"
}
```

---

### 4. Delete Model

**Summary:** Delete a specific model from the system.

**Endpoint:** `DELETE /api/v1/models/{model_name}`

**CURL:**

```bash
# Delete with .gguf extension
curl -X DELETE "http://127.0.0.1:5000/api/v1/models/model.gguf"

# Delete without extension (will auto-append .gguf)
curl -X DELETE "http://127.0.0.1:5000/api/v1/models/model"
```

**Expected Response:**

```json
{
  "message": "Model deleted successfully",
  "filename": "model.gguf",
  "remaining_models": 3
}
```

**Status Codes:**

- `200 OK` - Deletion successful
- `404 Not Found` - Model not found
- `500 Internal Server Error` - Deletion failed

**Error Response:**

```json
{
  "error": "Model not found"
}
```

---

### 5. Sync Models

**Summary:** Manually synchronize the models list with the filesystem.

**Endpoint:** `POST /api/v1/models/sync`

**CURL:**

```bash
curl -X POST "http://127.0.0.1:5000/api/v1/models/sync"
```

**Expected Response:**

```json
{
  "message": "Models list synchronized successfully",
  "data": {
    "models": [
      {
        "name": "gpt4all-falcon-newbpe-q4_0.gguf",
        "path": "./models/gpt4all-falcon-newbpe-q4_0.gguf",
        "size": "4.1 GB",
        "modified": "2024-01-15T10:30:00Z"
      }
    ],
    "count": 1
  }
}
```

**Status Codes:**

- `200 OK` - Sync successful
- `500 Internal Server Error` - Sync failed

---

### 6. Generate Response

**Summary:** Generate a text response using a specified model.

**Endpoint:** `POST /api/v1/generate`

**Request Body:**

```json
{
  "question": "What is artificial intelligence?",
  "model_name": "Llama-3.2-3B-Instruct-Q4_0.gguf",
  "template": "optional custom template",
  "n_gpu_layers": 40,
  "n_batch": 512,
  "temperature": 0.7
}
```

**Required Fields:**

- `question` (string) - The input text/question
- `model_name` (string) - Name of the model to use

**Optional Fields:**

- `template` (string) - Custom prompt template
- `n_gpu_layers` (integer) - Number of GPU layers (default: 40)
- `n_batch` (integer) - Batch size (default: 512)
- `temperature` (float) - Sampling temperature (default: 0.6)

**CURL:**

```bash
curl -X POST "http://127.0.0.1:5000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is artificial intelligence?",
    "model_name": "Llama-3.2-3B-Instruct-Q4_0.gguf",
    "temperature": 0.7
  }'
```

**Expected Response (Success):**

```json
{
  "success": true,
  "response": "Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans...",
  "model_used": "Llama-3.2-3B-Instruct-Q4_0.gguf",
  "processing_time": 2.45,
  "token_count": 156
}
```

**Status Codes:**

- `200 OK` - Generation successful
- `400 Bad Request` - Missing required fields
- `404 Not Found` - Model not found
- `500 Internal Server Error` - Generation failed

**Error Responses:**

```json
{
  "error": "Request body is required"
}
```

```json
{
  "error": "Question is required"
}
```

```json
{
  "error": "Model name is required"
}
```

```json
{
  "error": "Model not found"
}
```

```json
{
  "success": false,
  "error": "Failed to generate response: [error message]"
}
```

---

### 7. Clear Cache

**Summary:** Clear the LLM model cache to free up memory.

**Endpoint:** `POST /api/v1/cache/clear`

**CURL:**

```bash
curl -X POST "http://127.0.0.1:5000/api/v1/cache/clear"
```

**Expected Response:**

```json
{
  "message": "Cache cleared successfully"
}
```

**Status Codes:**

- `200 OK` - Cache cleared successfully
- `500 Internal Server Error` - Failed to clear cache

---

### 8. Cache Status

**Summary:** Get information about currently cached models.

**Endpoint:** `GET /api/v1/cache/status`

**CURL:**

```bash
curl -X GET "http://127.0.0.1:5000/api/v1/cache/status"
```

**Expected Response:**

```json
{
  "cached_models": [
    "Llama-3.2-3B-Instruct-Q4_0.gguf",
    "gpt4all-falcon-newbpe-q4_0.gguf"
  ],
  "count": 2
}
```

**Status Codes:**

- `200 OK` - Success
- `500 Internal Server Error` - Failed to get cache status

---

## Configuration

### Environment Variables

- `HOST` - Server host (default: 127.0.0.1)
- `PORT` - Server port (default: 5000)
- `API_PREFIX` - API path prefix (default: /api/v1)
- `MAX_CONTENT_LENGTH` - Maximum upload size (default: 10GB)
- `DEFAULT_N_GPU_LAYERS` - Default GPU layers (default: 40)
- `DEFAULT_N_BATCH` - Default batch size (default: 512)
- `DEFAULT_TEMPERATURE` - Default temperature (default: 0.6)

### File Upload Limits

- **Allowed Extensions:** `.gguf`
- **Maximum File Size:** 10 GB (configurable)
- **Upload Directory:** `./models`

---

## Error Handling

All error responses follow this format:

```json
{
  "error": "Description of the error"
}
```

Common HTTP status codes:

- `200 OK` - Success
- `400 Bad Request` - Invalid request
- `404 Not Found` - Resource not found
- `409 Conflict` - Resource already exists
- `500 Internal Server Error` - Server error

---

## Authentication

Currently, the API does not require authentication. All endpoints are publicly accessible.

---

## Rate Limiting

No rate limiting is currently implemented.

---

## Examples

### Complete Model Management Workflow

1. **Check API Health:**

```bash
curl -X GET "http://127.0.0.1:5000/api/v1/health"
```

2. **List Current Models:**

```bash
curl -X GET "http://127.0.0.1:5000/api/v1/models"
```

3. **Upload a New Model:**

```bash
curl -X POST "http://127.0.0.1:5000/api/v1/models" \
  -F "file=@./my-model.gguf"
```

4. **Generate Text:**

```bash
curl -X POST "http://127.0.0.1:5000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Explain quantum computing",
    "model_name": "my-model.gguf",
    "temperature": 0.8
  }'
```

5. **Check Cache Status:**

```bash
curl -X GET "http://127.0.0.1:5000/api/v1/cache/status"
```

6. **Clear Cache:**

```bash
curl -X POST "http://127.0.0.1:5000/api/v1/cache/clear"
```

7. **Delete Model:**

```bash
curl -X DELETE "http://127.0.0.1:5000/api/v1/models/my-model.gguf"
```

---

## Notes

- Model files must be in GGUF format
- The API automatically appends `.gguf` extension if not provided in model names
- Models are stored in the `./models` directory
- Large model files may take time to upload
- Text generation time depends on model size and hardware capabilities
