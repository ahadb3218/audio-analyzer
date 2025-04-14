# Getting Started with Audio Analyzer

This guide will help you set up and run the Audio Analyzer project locally.

## Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- Git
- CUDA-capable GPU (recommended for better performance)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/audioAnalyzer.git
   cd audioAnalyzer
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On Unix or MacOS
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Create a `.env` file in the project root:
   ```env
   FLASK_APP=app.py
   FLASK_ENV=development
   PORT=5000
   HOST=0.0.0.0
   DEBUG=True
   WHISPER_MODEL=base
   DEVICE=0  # Use -1 for CPU, 0 or higher for GPU
   ```

2. Adjust the configuration in `config.py` if needed.

## Running the Application

1. Start the Flask application:
   ```bash
   python app.py
   ```

2. The application will be available at `http://localhost:5000`

3. Access the API documentation at `http://localhost:5000/api/docs`

## Testing

Run the test suite:
```bash
pytest
```

## Common Issues

1. **CUDA/GPU Issues**
   - Ensure CUDA is properly installed
   - Check GPU compatibility
   - Set `DEVICE=-1` in `.env` to use CPU

2. **Dependencies Installation**
   - If you encounter issues with torch installation, visit [PyTorch's website](https://pytorch.org/) for specific installation instructions
   - For librosa issues, ensure you have the required system libraries installed

3. **Port Conflicts**
   - Change the `PORT` in `.env` if port 5000 is already in use

## Next Steps

- Read the [Architecture](./architecture.md) guide to understand the project structure
- Check the [API Reference](./api-reference.md) for available endpoints
- Review the [Development Guide](./development.md) for contributing to the project 