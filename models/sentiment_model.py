from transformers import pipeline
from config import Config


class SentimentAnalyzer:
    """Handles sentiment analysis using DistilBERT"""

    def __init__(self):
        """Initialize the sentiment analyzer"""
        print("Loading sentiment analysis model...")
        try:
            self.model = pipeline(
                "sentiment-analysis", model=Config.SENTIMENT_MODEL, device=Config.DEVICE
            )
            print("Sentiment analysis model loaded successfully")
        except Exception as e:
            print(f"Error loading sentiment analysis model: {str(e)}")
            self.model = None

    def analyze(self, text):
        """Analyze sentiment of text"""
        if self.model is None:
            print("Model not loaded")
            return None

        try:
            result = self.model(text)[0]
            return {
                "label": result["label"],
                "score": float(
                    result["score"]
                ),  # Convert to Python float for JSON serialization
            }
        except Exception as e:
            print(f"Error during sentiment analysis: {str(e)}")
            return None


# Create a singleton instance
sentiment_analyzer = SentimentAnalyzer()
