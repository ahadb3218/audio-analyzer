# Media Analysis API

A comprehensive Flask-based API for analyzing media files (video and audio) with various features including quality analysis, streaming readiness checks, and content analysis.

## Documentation

For detailed documentation, please visit our [documentation site](./docs/README.md). The documentation includes:

- [Getting Started Guide](./docs/getting-started.md)
- [Architecture Overview](./docs/architecture.md)
- [API Reference](./docs/api-reference.md)
- [ML Models Documentation](./docs/models.md)
- [Development Guide](./docs/development.md)
- [Deployment Guide](./docs/deployment.md)

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/zahidhasann88/media-analyzer.git
cd media-analyzer
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install ffmpeg (required for video processing):
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

## Features

- **Video Analysis**
  - Quality assessment (resolution, bitrate, compression artifacts)
  - Streaming readiness analysis
  - Archive suitability analysis
  - Duplicate frame detection
  - Video comparison
  - Enhancement recommendations
  - Metadata analysis

- **Audio Analysis**
  - Feature extraction
  - Content type determination
  - Speech density analysis
  - Transcription (using Whisper)

- **Content Analysis**
  - Text summarization
  - Sentiment analysis
  - Content type classification
  - Word count and speech density metrics

- **Batch Processing**
  - Multiple file analysis in a single request
  - Comprehensive error reporting
  - Progress tracking

## Usage

### Starting the Server

```bash
python app.py
```

The API will be available at `http://localhost:5000`

For detailed API documentation and examples, please refer to our [API Reference](./docs/api-reference.md).

## Configuration

The API can be configured through environment variables or the `config.py` file. For detailed configuration options, see the [Getting Started Guide](./docs/getting-started.md#configuration).

## Model Information

This implementation uses:
- OpenAI's Whisper (base model) for speech-to-text transcription
- Facebook's BART-large-CNN for text summarization
- DistilBERT for sentiment analysis
- Librosa for audio feature extraction
- ffmpeg for video-to-audio conversion

For detailed information about the models and their configurations, see our [ML Models Documentation](./docs/models.md).

## Performance Considerations

For production deployment and performance optimization guidelines, please refer to our [Deployment Guide](./docs/deployment.md#performance-considerations).

## Contributing

We welcome contributions! Please see our [Development Guide](./docs/development.md#contributing) for information on how to contribute to the project.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Check our [documentation](./docs/README.md)
- Open an issue on GitHub
- Contact the maintainers