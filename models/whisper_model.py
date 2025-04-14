import whisper
from config import Config


class WhisperTranscriber:
    """Handles audio transcription using OpenAI's Whisper model"""

    def __init__(self):
        """Initialize the transcriber with the specified model"""
        print(f"Loading Whisper model: {Config.WHISPER_MODEL}...")
        try:
            self.model = whisper.load_model(Config.WHISPER_MODEL)
            print("Whisper model loaded successfully")
        except Exception as e:
            print(f"Error loading Whisper model: {str(e)}")
            self.model = None

    def transcribe(self, audio_path):
        """Transcribe audio file and return the text"""
        if self.model is None:
            print("Model not loaded")
            return None

        try:
            result = self.model.transcribe(audio_path)
            return result["text"]
        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            return None


# Create a singleton instance
transcriber = WhisperTranscriber()
