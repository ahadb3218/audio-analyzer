import cv2
import numpy as np
from typing import Dict, Tuple, List
import os


def analyze_video_quality(video_path: str) -> Dict:
    """
    Analyze video quality metrics including resolution, bitrate, frame rate, and compression artifacts.

    Args:
        video_path: Path to the video file

    Returns:
        Dictionary containing video quality metrics
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video file")

    # Get basic video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    # Calculate bitrate (approximate)
    file_size = os.path.getsize(video_path)
    bitrate = (file_size * 8) / duration if duration > 0 else 0

    # Analyze frame quality
    frame_qualities = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate frame quality metrics
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        laplacian_var = cv2.Laplacian(blur, cv2.CV_64F).var()
        frame_qualities.append(laplacian_var)

    cap.release()

    # Calculate average quality metrics
    avg_quality = np.mean(frame_qualities) if frame_qualities else 0

    # Determine quality level
    quality_level = "High"
    if avg_quality < 100:
        quality_level = "Low"
    elif avg_quality < 200:
        quality_level = "Medium"

    return {
        "resolution": f"{width}x{height}",
        "aspect_ratio": f"{width/height:.2f}",
        "fps": round(fps, 2),
        "duration": round(duration, 2),
        "frame_count": frame_count,
        "bitrate": round(bitrate / 1000, 2),  # Convert to kbps
        "quality_score": round(avg_quality, 2),
        "quality_level": quality_level,
        "file_size_mb": round(file_size / (1024 * 1024), 2),
    }


def analyze_video_compression(video_path: str) -> Dict:
    """
    Analyze video compression artifacts and encoding quality.

    Args:
        video_path: Path to the video file

    Returns:
        Dictionary containing compression analysis results
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video file")

    # Get video codec information
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    # Analyze compression artifacts
    block_artifacts = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect block artifacts
        edges = cv2.Canny(gray, 100, 200)
        block_score = np.mean(edges) / 255.0
        block_artifacts.append(block_score)

    cap.release()

    avg_block_score = np.mean(block_artifacts) if block_artifacts else 0

    # Determine compression quality
    compression_quality = "Good"
    if avg_block_score > 0.3:
        compression_quality = "Poor"
    elif avg_block_score > 0.15:
        compression_quality = "Fair"

    return {
        "codec": codec,
        "compression_quality": compression_quality,
        "block_artifact_score": round(avg_block_score, 3),
        "compression_analysis": {
            "blocking_artifacts": round(avg_block_score * 100, 1),
            "compression_ratio": round(avg_block_score * 100, 1),
        },
    }


def analyze_video_content(video_path: str) -> Dict:
    """
    Analyze video content characteristics like motion, scene changes, and content type.

    Args:
        video_path: Path to the video file

    Returns:
        Dictionary containing content analysis results
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video file")

    prev_frame = None
    motion_scores = []
    scene_changes = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            # Calculate motion score
            diff = cv2.absdiff(gray, prev_frame)
            motion_score = np.mean(diff) / 255.0
            motion_scores.append(motion_score)

            # Detect scene changes
            if motion_score > 0.5:  # Threshold for scene change
                scene_changes.append(True)
            else:
                scene_changes.append(False)

        prev_frame = gray

    cap.release()

    # Calculate content characteristics
    avg_motion = np.mean(motion_scores) if motion_scores else 0
    scene_change_count = sum(scene_changes)

    # Determine content type
    content_type = "Static"
    if avg_motion > 0.3:
        content_type = "Dynamic"
    elif avg_motion > 0.1:
        content_type = "Moderate"

    return {
        "content_type": content_type,
        "motion_level": round(avg_motion * 100, 1),
        "scene_changes": scene_change_count,
        "content_analysis": {
            "motion_score": round(avg_motion * 100, 1),
            "scene_change_frequency": (
                round(scene_change_count / len(motion_scores) * 100, 1)
                if motion_scores
                else 0
            ),
        },
    }
