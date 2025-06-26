# Local LLM Application

A scalable Flask application for managing and interacting with local language models using GGUF format.

## Project Structure

```
Local LLM/
├── api/                    # API routes and blueprints
│   ├── __init__.py
│   └── routes.py          # API endpoints
├── models/                # Model management
│   ├── __init__.py
│   ├── model_manager.py   # Model file operations
│   └── ggml-vocab-aquila.gguf
├── services/              # Business logic services
│   ├── __init__.py
│   └── llm_service.py     # LLM operations
├── utils/                 # Utility functions
│   ├── __init__.py
│   ├── exceptions.py      # Custom exceptions
│   └── helpers.py         # Helper functions
├── app.py                 # Legacy app (deprecated)
├── app_factory.py         # Application factory
├── config.py              # Configuration settings
├── main.py                # Main entry point
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## Features

- **Model Management**: Upload, list, and delete GGUF model files
- **Text Generation**: Generate responses using local LLM models
- **Model Caching**: Efficient model loading with caching
- **RESTful API**: Clean API endpoints with proper error handling
- **Modular Architecture**: Scalable and maintainable code structure
- **Configuration Management**: Environment-based configuration
- **Logging**: Comprehensive logging system
- **CORS Support**: Cross-origin resource sharing enabled

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set environment variables (optional):

```bash
set FLASK_ENV=development
set HOST=127.0.0.1
set PORT=5000
```

## Usage

### Running the Application

**New way (recommended):**

```bash
python main.py
```

**Legacy way (deprecated but still works):**

```bash
python app.py
```

The application will start on `http://localhost:5000` by default.

### API Endpoints

#### Model Management

- `GET /api/v1/models` - List all models
- `POST /api/v1/models` - Upload a new model
- `DELETE /api/v1/models/<model_name>` - Delete a model
- `POST /api/v1/models/sync` - Sync models with filesystem

#### Text Generation

- `POST /api/v1/generate` - Generate text using a model

Example request:

```json
{
  "question": "What is artificial intelligence?",
  "model_name": "mistral-7b-openorca.Q4_0",
  "template": "Question: {question}\n\nAnswer: ",
  "n_gpu_layers": 40,
  "n_batch": 512,
  "temperature": 0.7
}
```

#### System

- `GET /api/v1/health` - Health check
- `POST /api/v1/cache/clear` - Clear model cache
- `GET /api/v1/cache/status` - Get cache status

#### Documentation

- `GET /` - API documentation and endpoint reference

## API Design

This application follows RESTful API design principles:

- **Versioned API**: All endpoints under `/api/v1/` for future compatibility
- **Consistent responses**: JSON format with proper HTTP status codes
- **Clear error messages**: Detailed error information for debugging
- **Self-documenting**: Visit `/` for complete API documentation

## Configuration

The application supports multiple configuration environments:

- **Development**: Default configuration with debug enabled
- **Production**: Production-ready configuration

Configuration can be set via environment variables:

- `FLASK_ENV`: Configuration environment (development/production)
- `SECRET_KEY`: Flask secret key (required for production)
- `HOST`: Server host (default: 127.0.0.1)
- `PORT`: Server port (default: 5000)

## Model Requirements

- Models must be in GGUF format
- Place model files in the `models/` directory
- Supported file extensions: `.gguf`

## Development

### Adding New Features

1. **Services**: Add business logic to `services/`
2. **API Routes**: Add new endpoints to `api/routes.py`
3. **Models**: Add data models to `models/`
4. **Configuration**: Update `config.py` for new settings

### Testing

The modular structure makes it easy to unit test individual components:

```python
from models.model_manager import ModelManager
from services.llm_service import LLMService

# Test model manager
manager = ModelManager('./test_models', './test_models.json')

# Test LLM service
service = LLMService(config)
```

## Architecture Benefits

### Scalability

- **Modular Design**: Easy to add new features without affecting existing code
- **Service Layer**: Business logic separated from API routes
- **Configuration Management**: Environment-specific settings
- **Caching**: Model caching for better performance

### Maintainability

- **Separation of Concerns**: Each module has a specific responsibility
- **Error Handling**: Comprehensive error handling with custom exceptions
- **Logging**: Structured logging for debugging and monitoring
- **Type Hints**: Better code documentation and IDE support

### Best Practices

- **Application Factory Pattern**: Clean app initialization
- **Blueprint Registration**: Organized route management
- **Environment Configuration**: Secure and flexible configuration
- **Clean API Design**: Versioned endpoints under `/api/v1/`

## Migration Guide

### From Legacy Code

**BREAKING CHANGE**: Legacy routes have been removed for a cleaner API structure.

**Before (Legacy)**:

```bash
POST /upload         # Upload model
GET  /models         # List models
POST /generate       # Generate text
```

**After (Clean API)**:

```bash
POST /api/v1/models      # Upload model
GET  /api/v1/models      # List models
POST /api/v1/generate    # Generate text
```

### Migration Steps:

1. **Update Application Start**: Use `python main.py` instead of `python app.py`
2. **Update API Calls**: Change all endpoints to use `/api/v1/` prefix
3. **Update Client Code**: Modify your client applications to use the new endpoints
4. **Test**: Visit `http://localhost:5000/` for complete API documentation

### Benefits of Migration:

- ✅ **Cleaner architecture**: No route duplication
- ✅ **Future-proof**: Versioned API for easy updates
- ✅ **Better features**: Access to cache management and health monitoring
- ✅ **Self-documenting**: Built-in API documentation

## License

This project is for educational and development purposes.
