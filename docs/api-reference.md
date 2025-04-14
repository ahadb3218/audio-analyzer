# API Reference

This document provides detailed information about the Audio Analyzer API endpoints, request/response formats, and error handling.

## Base URL

```
http://localhost:5000/api
```

## Authentication

Currently, the API does not require authentication. Future versions may implement API key authentication.

## Endpoints

### Health Check

```http
GET /health
```

Check the health status of the API.

**Response**
```json
{
    "status": "healthy"
}
```

### Analyze Audio

```http
POST /analyze
```

Analyze an audio file to extract text, generate summary, and perform sentiment analysis.

**Request**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body:
  - `file`: Audio file (supported formats: WAV, MP3, M4A)

**Response**
```json
{
    "transcription": "Full transcribed text...",
    "summary": "Summarized text...",
    "sentiment": {
        "label": "positive",
        "score": 0.95
    },
    "metadata": {
        "duration": 120.5,
        "language": "en",
        "model": "whisper-base"
    }
}
```

**Error Responses**
```json
{
    "error": "Invalid file format",
    "message": "Only WAV, MP3, and M4A files are supported"
}
```

```json
{
    "error": "File too large",
    "message": "Maximum file size is 25MB"
}
```

### Get Analysis Status

```http
GET /status/{job_id}
```

Check the status of an ongoing analysis job.

**Parameters**
- `job_id`: Unique identifier for the analysis job

**Response**
```json
{
    "status": "processing",
    "progress": 75,
    "estimated_time": 30
}
```

## Error Codes

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid input |
| 413 | Payload Too Large - File too big |
| 415 | Unsupported Media Type - Invalid file format |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Model loading issues |

## Rate Limiting

- 100 requests per minute per IP
- 1000 requests per hour per IP

## Best Practices

1. **File Upload**
   - Use appropriate file formats (WAV, MP3, M4A)
   - Keep file size under 25MB
   - Use multipart/form-data for file uploads

2. **Error Handling**
   - Always check response status codes
   - Implement proper error handling for network issues
   - Handle rate limiting gracefully

3. **Performance**
   - Use compression for large files
   - Implement retry logic for failed requests
   - Cache responses when appropriate

## Example Usage

### Python

```python
import requests

def analyze_audio(file_path):
    url = "http://localhost:5000/api/analyze"
    files = {"file": open(file_path, "rb")}
    response = requests.post(url, files=files)
    return response.json()

# Usage
result = analyze_audio("audio.wav")
print(result["summary"])
```

### cURL

```bash
curl -X POST \
  http://localhost:5000/api/analyze \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@audio.wav'
```

## WebSocket API

For real-time analysis, a WebSocket endpoint is available:

```javascript
const ws = new WebSocket('ws://localhost:5000/api/ws');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data.progress);
};
```

## Versioning

The API is versioned through the URL path:
- Current version: `/api/v1/`
- Legacy version: `/api/v0/` (deprecated)

## Changelog

### v1.0.0
- Initial release
- Basic audio analysis features
- Health check endpoint
- Status monitoring 