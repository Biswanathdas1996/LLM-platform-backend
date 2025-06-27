# DeepSeek API Quick Reference

## Quick Start Commands

### Start the API Server

```bash
cd "d:\projects\Local LLM"
python main.py
```

### Test Basic Functionality

```bash
# Health check
curl http://localhost:5000/api/deepseek/health

# List models
curl http://localhost:5000/api/deepseek/models

# Quick test
curl -X POST http://localhost:5000/api/deepseek/test
```

## Essential Endpoints

| Method | Endpoint                      | Description              |
| ------ | ----------------------------- | ------------------------ |
| GET    | `/api/deepseek/health`        | Service health check     |
| GET    | `/api/deepseek/models`        | List available models    |
| GET    | `/api/deepseek/models/{name}` | Get model info           |
| POST   | `/api/deepseek/generate`      | Generate text            |
| POST   | `/api/deepseek/test`          | Quick test with defaults |

## Common Usage Patterns

### Simple Text Generation

```bash
curl -X POST http://localhost:5000/api/deepseek/generate \
  -H "Content-Type: application/json" \
  -d '{"model_name": "DeepSeek-R1-q2_k.gguf", "prompt": "Your question here"}'
```

### Creative Writing (Higher Temperature)

```bash
curl -X POST http://localhost:5000/api/deepseek/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "DeepSeek-R1-q2_k.gguf",
    "prompt": "Write a creative story about...",
    "temperature": 0.9,
    "max_tokens": 300
  }'
```

### Factual Responses (Lower Temperature)

```bash
curl -X POST http://localhost:5000/api/deepseek/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "DeepSeek-R1-q2_k.gguf",
    "prompt": "Explain the concept of...",
    "temperature": 0.3,
    "max_tokens": 200
  }'
```

## Parameter Quick Reference

### Required Parameters

- `model_name`: Model file name (e.g., "DeepSeek-R1-q2_k.gguf")
- `prompt`: Input text for generation

### Key Optional Parameters

- `max_tokens`: 50-500 (default: 200)
- `temperature`: 0.1-1.0 (default: 0.7)
- `n_ctx`: 512-4096 (default: 2048)
- `n_threads`: 1-16 (default: 8)

### Temperature Guide

- 0.1-0.3: Focused, factual responses
- 0.4-0.7: Balanced creativity and coherence
- 0.8-1.0: Creative, varied responses

## Model Selection Guide

| Model  | Size     | Speed    | Quality | Use Case                       |
| ------ | -------- | -------- | ------- | ------------------------------ |
| q2_k   | Smallest | Fastest  | Lower   | Quick testing, simple tasks    |
| q4_1   | Medium   | Balanced | Good    | General purpose                |
| q4_k_m | Medium+  | Balanced | Better  | Recommended for most use cases |
| Q3_K_L | Large    | Slower   | High    | High-quality responses         |

## Common Response Patterns

### Success Response

```json
{
  "success": true,
  "generated_text": "Generated content here...",
  "model": "model_name",
  "prompt": "original_prompt"
}
```

### Error Response

```json
{
  "success": false,
  "error": "Error description",
  "model": "model_name"
}
```

## Troubleshooting Quick Fixes

| Issue                 | Quick Fix                                  |
| --------------------- | ------------------------------------------ |
| Model not found       | Check model exists in `DeepSeekLLM/model/` |
| Out of memory         | Use smaller model (q2_k) or reduce n_ctx   |
| Slow response         | Reduce max_tokens or increase n_threads    |
| Server not responding | Check if `python main.py` is running       |
| Connection refused    | Verify port 5000 is available              |

## Integration Templates
