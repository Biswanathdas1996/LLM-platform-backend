# HuggingFace API Quick Reference

## Quick Start Commands

### Add Model

```bash
curl -X POST "http://localhost:5000/api/huggingface/models" \
  -H "Content-Type: application/json" \
  -d '{"model_id": "gpt2", "name": "GPT-2", "model_type": "text-generation"}'
```

### List Models

```bash
curl -X GET "http://localhost:5000/api/huggingface/models"
```

### Generate Text

```bash
curl -X POST "http://localhost:5000/api/huggingface/generate" \
  -H "Content-Type: application/json" \
  -d '{"model_id": "gpt2", "prompt": "Hello", "max_new_tokens": 50}'
```

### Load Model

```bash
curl -X POST "http://localhost:5000/api/huggingface/models/gpt2/load"
```

### Check Dependencies

```bash
curl -X GET "http://localhost:5000/api/huggingface/dependencies"
```

### Get Statistics

```bash
curl -X GET "http://localhost:5000/api/huggingface/statistics"
```

## Endpoints Summary

| Method | Endpoint                            | Purpose       |
| ------ | ----------------------------------- | ------------- |
| GET    | `/api/huggingface/models`           | List models   |
| POST   | `/api/huggingface/models`           | Add model     |
| GET    | `/api/huggingface/models/{id}`      | Get model     |
| PUT    | `/api/huggingface/models/{id}`      | Update model  |
| DELETE | `/api/huggingface/models/{id}`      | Remove model  |
| POST   | `/api/huggingface/models/{id}/load` | Load model    |
| POST   | `/api/huggingface/generate`         | Generate text |
| POST   | `/api/huggingface/pipeline`         | Use pipeline  |
| GET    | `/api/huggingface/cache`            | Cache info    |
| DELETE | `/api/huggingface/cache`            | Clear cache   |
| GET    | `/api/huggingface/statistics`       | Usage stats   |
| GET    | `/api/huggingface/dependencies`     | Check deps    |

## Popular Models to Try

### Text Generation

- `gpt2` - OpenAI GPT-2
- `microsoft/DialoGPT-medium` - Conversational AI
- `facebook/opt-350m` - Meta OPT model

### Summarization

- `facebook/bart-large-cnn` - BART for summarization
- `t5-small` - T5 model for text-to-text

### Translation

- `Helsinki-NLP/opus-mt-en-fr` - English to French
- `Helsinki-NLP/opus-mt-fr-en` - French to English

## Generation Parameters

| Parameter            | Type  | Default | Description            |
| -------------------- | ----- | ------- | ---------------------- |
| `max_new_tokens`     | int   | 100     | Max tokens to generate |
| `temperature`        | float | 0.7     | Sampling temperature   |
| `do_sample`          | bool  | true    | Use sampling           |
| `top_p`              | float | 0.9     | Nucleus sampling       |
| `top_k`              | int   | 50      | Top-k sampling         |
| `repetition_penalty` | float | 1.1     | Repetition penalty     |

## Error Codes

| Code | Meaning                            |
| ---- | ---------------------------------- |
| 200  | Success                            |
| 400  | Bad request / Missing dependencies |
| 404  | Model not found                    |
| 500  | Server error                       |

## Python Quick Start

```python
import requests

base_url = "http://localhost:5000/api/huggingface"

# Add model
requests.post(f"{base_url}/models", json={
    "model_id": "gpt2",
    "name": "GPT-2",
    "model_type": "text-generation"
})

# Generate text
response = requests.post(f"{base_url}/generate", json={
    "model_id": "gpt2",
    "prompt": "Hello world",
    "max_new_tokens": 30
})

print(response.json()["response"])
```
