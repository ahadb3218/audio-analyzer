import subprocess
import librosa
import numpy as np
from pathlib import Path


def extract_audio_from_video(video_path):
    """Extract audio from video file"""
    audio_path = video_path.replace(Path(video_path).suffix, ".wav")
    try:
        # Use ffmpeg to extract audio
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                video_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                audio_path,
            ],
            check=True,
            capture_output=True,
        )
        return audio_path
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
        print(f"ffmpeg stderr: {e.stderr.decode()}")
        raise Exception(f"Failed to extract audio from video: {str(e)}")


def analyze_audio_features(audio_path):
    """Extract audio features using librosa"""
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    energy = np.mean(y**2)

    features = {"duration": duration, "energy": energy, "sample_rate": sr}

    return features, y, sr


def determine_content_type(transcript, duration):
    """Determine content type based on speech density"""
    word_count = len(transcript.split())
    speech_density = word_count / duration if duration > 0 else 0

    if speech_density > 2.0:
        content_type = "Dense speech/dialogue"
    elif speech_density > 0.5:
        content_type = "Normal conversation"
    elif word_count > 10:
        content_type = "Sparse speech"
    else:
        content_type = "Minimal speech, primarily background audio"

    return content_type, word_count, speech_density
