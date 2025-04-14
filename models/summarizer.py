import re
import traceback
from transformers import pipeline
from config import Config
from models.sentiment_model import sentiment_analyzer


class TextSummarizer:
    """Handles text summarization using BART"""

    def __init__(self):
        """Initialize the summarizer"""
        print("Loading summarization model...")
        try:
            self.model = pipeline(
                "summarization",
                model=Config.SUMMARIZER_MODEL,
                tokenizer=Config.SUMMARIZER_MODEL,
                device=Config.DEVICE,
            )
            print("Summarization model loaded successfully")
        except Exception as e:
            print(f"Error loading summarization model: {str(e)}")
            self.model = None

    def _summarize_single_chunk(self, text):
        """Generate summary for a single chunk of text"""
        try:
            summary = self.model(text, max_length=130, min_length=30, do_sample=False)[
                0
            ]
            return summary["summary_text"]
        except Exception as e:
            print(f"Error in single chunk summarization: {str(e)}")
            return None

    def generate_summary(self, text, features=None):
        """Generate a summary from text with optional features"""
        if self.model is None:
            print("Model not loaded")
            return None

        try:
            # Split text into chunks if needed
            chunks = [
                text[i : i + Config.MAX_CHUNK_SIZE]
                for i in range(0, len(text), Config.MAX_CHUNK_SIZE)
            ]

            # Generate summaries for each chunk
            summaries = []
            for chunk in chunks:
                summary = self._summarize_single_chunk(chunk)
                if summary:
                    summaries.append(summary)

            # Combine summaries
            final_summary = " ".join(summaries)

            # Add duration information if available
            if features and "duration" in features:
                duration_min = float(features["duration"]) / 60
                final_summary += f" (Duration: {duration_min:.1f} minutes)"

            # Analyze sentiment if available
            if sentiment_analyzer:
                sentiment_result = sentiment_analyzer.analyze(final_summary)
                if sentiment_result:
                    final_summary += f" (Sentiment: {sentiment_result['label']}, Confidence: {sentiment_result['score']:.2f})"

            return final_summary
        except Exception as e:
            print(f"Error in generate_summary: {str(e)}")
            return None


# Create a singleton instance
summarizer = TextSummarizer()
