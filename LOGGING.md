# Logging System Documentation

## Overview

The Local LLM API now includes comprehensive logging functionality that tracks all API calls, responses, and errors. This system provides structured logging with JSON format for easy parsing and analysis.

## Features

- **Request/Response Logging**: Automatically logs all incoming requests and outgoing responses
- **Error Tracking**: Detailed error logging with context and stack traces
- **Performance Monitoring**: Response time tracking for all endpoints
- **Structured Data**: JSON-formatted logs for easy parsing and analysis
- **Log Rotation**: Automatic log file rotation to prevent excessive disk usage
- **Custom Events**: Support for logging custom application events

## Configuration

### Environment Variables

You can configure the logging system using the following environment variables:

```bash
# Enable/disable API call logging (default: True)
LOG_API_CALLS=True

# Log directory path (default: ./logs)
LOG_DIR=./logs

# Maximum log file size in bytes (default: 10MB)
LOG_MAX_BYTES=10485760

# Number of backup log files to keep (default: 5)
LOG_BACKUP_COUNT=5
```

### Log Files

The system creates two main log files:

- `api.log` - Contains all API requests, responses, and general events
- `errors.log` - Contains only errors and exceptions

## Log Format

All logs are stored in JSON format with the following structure:

```json
{
  "timestamp": "2025-06-26T10:30:00.123456",
  "level": "INFO",
  "message": "Incoming request",
  "module": "logger",
  "request_id": "req_1719405000123456",
  "method": "POST",
  "url": "http://localhost:5000/api/v1/generate",
  "endpoint": "generate_response",
  "remote_addr": "127.0.0.1",
  "user_agent": "curl/7.68.0",
  "content_type": "application/json",
  "type": "request"
}
```

## Log Types

### Request Logs

- **Type**: `request`
- **Contains**: HTTP method, URL, headers, request body (for JSON), client IP

### Response Logs

- **Type**: `response`
- **Contains**: Status code, response time, response body (for JSON), content type

### Error Logs

- **Type**: `error` or `exception`
- **Contains**: Error message, exception type, stack trace, request context

### Custom Event Logs

- **Type**: `custom_event`
- **Contains**: Event type, custom data, contextual information

## Using the Log Viewer

The system includes a command-line log viewer utility for analyzing logs:

### Basic Usage

```bash
# View recent API and error logs
python utils/log_viewer.py

# View only API logs
python utils/log_viewer.py --type api

# View only error logs
python utils/log_viewer.py --type errors

# Show last 100 log entries
python utils/log_viewer.py --lines 100
```

### Statistics and Analysis

```bash
# Show API usage statistics
python utils/log_viewer.py --stats

# Show error summary only
python utils/log_viewer.py --errors-only

# Analyze last 6 hours of activity
python utils/log_viewer.py --stats --hours 6
```

### Advanced Options

```bash
# Specify custom log directory
python utils/log_viewer.py --log-dir /path/to/logs

# Combine multiple options
python utils/log_viewer.py --stats --hours 12 --log-dir ./custom_logs
```

## Example Log Entries

### Successful API Request

```json
{
  "timestamp": "2025-06-26T10:30:00.123456",
  "level": "INFO",
  "message": "Text generation completed successfully with model: Llama-3.2-3B-Instruct-Q4_0.gguf",
  "module": "route_handlers",
  "event_type": "text_generation_success",
  "model_name": "Llama-3.2-3B-Instruct-Q4_0.gguf",
  "response_length": 156,
  "generation_time": 2.34,
  "type": "custom_event"
}
```

### Error Log Entry

```json
{
  "timestamp": "2025-06-26T10:31:00.789012",
  "level": "ERROR",
  "message": "Text generation error with model mistral-7b: Model not found",
  "module": "route_handlers",
  "request_id": "req_1719405060789012",
  "method": "POST",
  "url": "http://localhost:5000/api/v1/generate",
  "endpoint": "generate_response",
  "model_name": "mistral-7b",
  "error_type": "ModelNotFoundError",
  "error_message": "Model not found",
  "type": "custom_event"
}
```

### Response Time Log

```json
{
  "timestamp": "2025-06-26T10:30:02.456789",
  "level": "INFO",
  "message": "Outgoing response",
  "module": "logger",
  "request_id": "req_1719405000123456",
  "method": "POST",
  "url": "http://localhost:5000/api/v1/generate",
  "endpoint": "generate_response",
  "status_code": 200,
  "content_type": "application/json",
  "duration_ms": 2340.12,
  "type": "response"
}
```

## Monitoring and Alerts

You can set up monitoring and alerts based on the log data:

### Performance Monitoring

- Track average response times
- Monitor slow endpoints (>5 seconds)
- Alert on response time degradation

### Error Monitoring

- Track error rates by endpoint
- Monitor specific error types
- Alert on error spikes

### Usage Analytics

- Track API usage patterns
- Monitor most popular endpoints
- Analyze user behavior

## Log Analysis Examples

### Find Slow Requests

```bash
# Find requests taking longer than 5 seconds
grep '"duration_ms".*[5-9][0-9][0-9][0-9]' logs/api.log
```

### Count Errors by Type

```python
import json

error_types = {}
with open('logs/errors.log', 'r') as f:
    for line in f:
        log = json.loads(line)
        error_type = log.get('error_type', 'Unknown')
        error_types[error_type] = error_types.get(error_type, 0) + 1

for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
    print(f"{error_type}: {count}")
```

### Monitor API Usage

```bash
# Count requests by endpoint in the last hour
python utils/log_viewer.py --stats --hours 1
```

## Best Practices

1. **Regular Log Review**: Check logs regularly for errors and performance issues
2. **Log Rotation**: Ensure log rotation is working to prevent disk space issues
3. **Error Monitoring**: Set up alerts for error spikes or new error types
4. **Performance Tracking**: Monitor response times and track performance trends
5. **Security Monitoring**: Watch for suspicious patterns in request logs

## Troubleshooting

### Common Issues

1. **Logs not appearing**: Check that `LOG_API_CALLS=True` in your environment
2. **Permission errors**: Ensure the application has write access to the log directory
3. **Large log files**: Adjust `LOG_MAX_BYTES` and `LOG_BACKUP_COUNT` settings
4. **Missing logs**: Check the log directory path configuration

### Debug Mode

For additional debugging information, set the Flask app to debug mode:

```bash
DEBUG=True python main.py
```

This will also output logs to the console in addition to files.
