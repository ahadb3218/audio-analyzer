from flask import Flask, jsonify
from flask_cors import CORS

from api.routes import api_bp
from config import Config
from utils.json_encoder import NumpyEncoder
from models.whisper_model import transcriber
from models.summarizer import summarizer
from models.sentiment_model import sentiment_analyzer


def create_app(config_name="development"):
    """Create and configure the Flask application"""
    app = Flask(__name__)

    # Load configuration
    app.config.from_object(Config)

    # Enable CORS
    CORS(app)

    # Register blueprints
    app.register_blueprint(api_bp, url_prefix="/api")

    @app.route("/health")
    def health_check():
        """Health check endpoint"""
        return jsonify({"status": "healthy"})

    return app


if __name__ == "__main__":
    app = create_app("development")
    print(f"Starting Media Analyzer API on port {Config.PORT}")
    print(f"Whisper model: {Config.WHISPER_MODEL}")
    print(f"Using GPU: {Config.DEVICE >= 0}")
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
