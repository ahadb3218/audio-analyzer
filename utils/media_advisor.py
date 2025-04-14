import cv2
import numpy as np
from typing import Dict, List, Tuple
import os
import hashlib
from pathlib import Path


def calculate_frame_hash(frame: np.ndarray) -> str:
    """Calculate a hash for a frame to detect duplicates"""
    frame_bytes = frame.tobytes()
    return hashlib.md5(frame_bytes).hexdigest()


def analyze_streaming_readiness(video_path: str) -> Dict:
    """
    Analyze video for streaming readiness including bitrate, resolution, and codec compatibility.

    Args:
        video_path: Path to the video file

    Returns:
        Dictionary containing streaming readiness analysis
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video file")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    # Get codec information
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    # Calculate bitrate
    file_size = os.path.getsize(video_path)
    bitrate = (file_size * 8) / duration if duration > 0 else 0

    # Determine streaming readiness
    resolution_category = "Low"
    if width >= 1920:
        resolution_category = "High"
    elif width >= 1280:
        resolution_category = "Medium"

    # Check codec compatibility
    codec_compatible = codec.lower() in ["avc1", "h264", "hevc", "h265"]

    # Check bitrate suitability
    bitrate_suitable = True
    if resolution_category == "High" and bitrate < 5000:  # 5 Mbps for 1080p
        bitrate_suitable = False
    elif resolution_category == "Medium" and bitrate < 2500:  # 2.5 Mbps for 720p
        bitrate_suitable = False
    elif resolution_category == "Low" and bitrate < 1000:  # 1 Mbps for 480p
        bitrate_suitable = False

    cap.release()

    return {
        "streaming_readiness": {
            "is_ready": codec_compatible and bitrate_suitable,
            "resolution_category": resolution_category,
            "codec_compatible": codec_compatible,
            "bitrate_suitable": bitrate_suitable,
        },
        "technical_details": {
            "resolution": f"{width}x{height}",
            "fps": round(fps, 2),
            "bitrate_kbps": round(bitrate / 1000, 2),
            "codec": codec,
            "duration": round(duration, 2),
        },
        "recommendations": {
            "resolution": (
                "Current resolution is suitable"
                if bitrate_suitable
                else "Consider reducing resolution"
            ),
            "codec": (
                "Current codec is suitable"
                if codec_compatible
                else "Consider converting to H.264"
            ),
            "bitrate": (
                "Current bitrate is suitable"
                if bitrate_suitable
                else "Consider adjusting bitrate"
            ),
        },
    }


def analyze_video_archive(video_path: str) -> Dict:
    """
    Analyze video for archival purposes including quality assessment and storage recommendations.

    Args:
        video_path: Path to the video file

    Returns:
        Dictionary containing archival analysis
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video file")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    # Calculate storage requirements
    file_size = os.path.getsize(video_path)
    file_size_mb = file_size / (1024 * 1024)

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

    # Calculate quality metrics
    avg_quality = np.mean(frame_qualities) if frame_qualities else 0

    # Determine archival quality
    archival_quality = "High"
    if avg_quality < 100:
        archival_quality = "Low"
    elif avg_quality < 200:
        archival_quality = "Medium"

    # Calculate storage efficiency
    storage_efficiency = "Good"
    if file_size_mb / duration > 100:  # More than 100MB per minute
        storage_efficiency = "Poor"
    elif file_size_mb / duration > 50:  # More than 50MB per minute
        storage_efficiency = "Fair"

    return {
        "archival_quality": archival_quality,
        "storage_analysis": {
            "file_size_mb": round(file_size_mb, 2),
            "duration_minutes": round(duration / 60, 2),
            "storage_efficiency": storage_efficiency,
            "mb_per_minute": round(file_size_mb / (duration / 60), 2),
        },
        "technical_details": {
            "resolution": f"{width}x{height}",
            "fps": round(fps, 2),
            "frame_count": frame_count,
            "quality_score": round(avg_quality, 2),
        },
        "recommendations": {
            "storage": (
                "Current storage efficiency is good"
                if storage_efficiency == "Good"
                else "Consider optimizing storage"
            ),
            "quality": (
                "Current quality is suitable for archival"
                if archival_quality in ["High", "Medium"]
                else "Consider improving quality"
            ),
        },
    }


def detect_duplicate_frames(video_path: str, threshold: float = 0.95) -> Dict:
    """
    Detect duplicate or similar frames in a video.

    Args:
        video_path: Path to the video file
        threshold: Similarity threshold (0-1)

    Returns:
        Dictionary containing duplicate frame analysis
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video file")

    frame_hashes = []
    duplicate_groups = []
    current_group = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_hash = calculate_frame_hash(frame)
        frame_number = len(frame_hashes)

        if frame_hash in frame_hashes:
            if not current_group:
                current_group = [frame_hashes.index(frame_hash)]
            current_group.append(frame_number)
        else:
            if current_group:
                duplicate_groups.append(current_group)
                current_group = []
            frame_hashes.append(frame_hash)

    if current_group:
        duplicate_groups.append(current_group)

    cap.release()

    # Calculate duplicate statistics
    total_frames = len(frame_hashes)
    duplicate_frames = sum(len(group) - 1 for group in duplicate_groups)
    duplicate_percentage = (
        (duplicate_frames / total_frames) * 100 if total_frames > 0 else 0
    )

    return {
        "duplicate_analysis": {
            "total_frames": total_frames,
            "duplicate_frames": duplicate_frames,
            "duplicate_percentage": round(duplicate_percentage, 2),
            "duplicate_groups": len(duplicate_groups),
        },
        "duplicate_details": [
            {
                "group_id": i,
                "original_frame": group[0],
                "duplicate_frames": group[1:],
                "count": len(group) - 1,
            }
            for i, group in enumerate(duplicate_groups)
        ],
    }


def get_conversion_recommendations(video_path: str) -> Dict:
    """
    Generate recommendations for video conversion based on content and quality analysis.

    Args:
        video_path: Path to the video file

    Returns:
        Dictionary containing conversion recommendations
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video file")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    # Get codec information
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    # Calculate bitrate
    file_size = os.path.getsize(video_path)
    bitrate = (file_size * 8) / duration if duration > 0 else 0

    # Analyze motion and content
    prev_frame = None
    motion_scores = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            diff = cv2.absdiff(gray, prev_frame)
            motion_score = np.mean(diff) / 255.0
            motion_scores.append(motion_score)

        prev_frame = gray

    cap.release()

    # Calculate content characteristics
    avg_motion = np.mean(motion_scores) if motion_scores else 0

    # Generate recommendations based on content
    recommendations = {
        "codec": {
            "current": codec,
            "recommended": (
                "H.264"
                if codec.lower() not in ["avc1", "h264"]
                else "Current codec is suitable"
            ),
            "reason": "H.264 provides good compression and wide compatibility",
        },
        "resolution": {
            "current": f"{width}x{height}",
            "recommended": (
                "Current resolution is suitable"
                if width >= 1280
                else "Consider upscaling to 720p"
            ),
            "reason": "Higher resolution provides better quality for most content",
        },
        "bitrate": {
            "current": round(bitrate / 1000, 2),
            "recommended": (
                "Current bitrate is suitable"
                if bitrate >= 2500
                else "Consider increasing bitrate"
            ),
            "reason": "Higher bitrate provides better quality",
        },
        "fps": {
            "current": round(fps, 2),
            "recommended": (
                "Current FPS is suitable" if fps >= 24 else "Consider increasing FPS"
            ),
            "reason": "Higher FPS provides smoother motion",
        },
    }

    # Add content-specific recommendations
    if avg_motion > 0.3:
        recommendations["content_specific"] = {
            "type": "High motion content",
            "suggestions": [
                "Consider higher bitrate for better motion quality",
                "Maintain current FPS for smooth motion",
            ],
        }
    elif avg_motion < 0.1:
        recommendations["content_specific"] = {
            "type": "Static content",
            "suggestions": [
                "Consider lower bitrate as motion is minimal",
                "Current FPS is sufficient",
            ],
        }

    return {
        "conversion_recommendations": recommendations,
        "content_analysis": {
            "motion_level": round(avg_motion * 100, 1),
            "duration": round(duration, 2),
            "file_size_mb": round(file_size / (1024 * 1024), 2),
        },
    }


def compare_videos(video_path1: str, video_path2: str) -> Dict:
    """
    Compare two videos for similarity and differences.

    Args:
        video_path1: Path to the first video file
        video_path2: Path to the second video file

    Returns:
        Dictionary containing comparison analysis
    """
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    if not cap1.isOpened() or not cap2.isOpened():
        raise ValueError("Could not open one or both video files")

    # Get video properties
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    duration1 = frame_count1 / fps1 if fps1 > 0 else 0

    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    duration2 = frame_count2 / fps2 if fps2 > 0 else 0

    # Calculate frame differences
    frame_diffs = []
    frame_similarities = []
    min_frames = min(frame_count1, frame_count2)

    for _ in range(min_frames):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        # Resize frames to same size for comparison
        if (width1, height1) != (width2, height2):
            frame2 = cv2.resize(frame2, (width1, height1))

        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calculate frame difference
        diff = cv2.absdiff(gray1, gray2)
        diff_score = np.mean(diff) / 255.0
        frame_diffs.append(diff_score)

        # Calculate frame similarity
        similarity = 1 - diff_score
        frame_similarities.append(similarity)

    cap1.release()
    cap2.release()

    # Calculate comparison metrics
    avg_diff = np.mean(frame_diffs) if frame_diffs else 0
    avg_similarity = np.mean(frame_similarities) if frame_similarities else 0

    # Determine video relationship
    relationship = "Different"
    if avg_similarity > 0.95:
        relationship = "Identical"
    elif avg_similarity > 0.8:
        relationship = "Very Similar"
    elif avg_similarity > 0.5:
        relationship = "Similar"

    return {
        "comparison_analysis": {
            "relationship": relationship,
            "average_similarity": round(avg_similarity * 100, 2),
            "average_difference": round(avg_diff * 100, 2),
            "frames_compared": min_frames,
        },
        "video1_details": {
            "resolution": f"{width1}x{height1}",
            "fps": round(fps1, 2),
            "duration": round(duration1, 2),
            "frame_count": frame_count1,
        },
        "video2_details": {
            "resolution": f"{width2}x{height2}",
            "fps": round(fps2, 2),
            "duration": round(duration2, 2),
            "frame_count": frame_count2,
        },
        "differences": {
            "resolution_different": (width1, height1) != (width2, height2),
            "fps_different": abs(fps1 - fps2) > 0.1,
            "duration_different": abs(duration1 - duration2) > 0.1,
            "frame_count_different": frame_count1 != frame_count2,
        },
    }


def get_enhancement_recommendations(video_path: str) -> Dict:
    """
    Generate recommendations for video enhancement and quality improvement.

    Args:
        video_path: Path to the video file

    Returns:
        Dictionary containing enhancement recommendations
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video file")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    # Analyze frame quality and content
    frame_qualities = []
    motion_scores = []
    noise_scores = []
    prev_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate frame quality
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        laplacian_var = cv2.Laplacian(blur, cv2.CV_64F).var()
        frame_qualities.append(laplacian_var)

        # Calculate motion
        if prev_frame is not None:
            diff = cv2.absdiff(gray, prev_frame)
            motion_score = np.mean(diff) / 255.0
            motion_scores.append(motion_score)

        # Calculate noise level
        noise = cv2.fastNlMeansDenoising(gray)
        noise_score = np.mean(np.abs(gray - noise)) / 255.0
        noise_scores.append(noise_score)

        prev_frame = gray

    cap.release()

    # Calculate metrics
    avg_quality = np.mean(frame_qualities) if frame_qualities else 0
    avg_motion = np.mean(motion_scores) if motion_scores else 0
    avg_noise = np.mean(noise_scores) if noise_scores else 0

    # Generate enhancement recommendations
    recommendations = {
        "quality_improvements": [],
        "technical_enhancements": [],
        "content_optimizations": [],
    }

    # Quality-based recommendations
    if avg_quality < 100:
        recommendations["quality_improvements"].extend(
            [
                "Increase resolution for better clarity",
                "Improve lighting conditions",
                "Use higher quality camera settings",
            ]
        )
    elif avg_quality < 200:
        recommendations["quality_improvements"].extend(
            ["Consider slight resolution increase", "Optimize exposure settings"]
        )

    # Noise-based recommendations
    if avg_noise > 0.2:
        recommendations["technical_enhancements"].extend(
            ["Apply noise reduction", "Improve ISO settings", "Consider using a tripod"]
        )

    # Motion-based recommendations
    if avg_motion > 0.3:
        recommendations["content_optimizations"].extend(
            [
                "Use image stabilization",
                "Consider higher FPS for smoother motion",
                "Optimize camera movement",
            ]
        )
    elif avg_motion < 0.1:
        recommendations["content_optimizations"].extend(
            ["Add more dynamic content", "Consider time-lapse for static scenes"]
        )

    # Resolution recommendations
    if width < 1280:
        recommendations["technical_enhancements"].append(
            "Upgrade to HD resolution (1280x720)"
        )
    elif width < 1920:
        recommendations["technical_enhancements"].append(
            "Consider Full HD resolution (1920x1080)"
        )

    # FPS recommendations
    if fps < 24:
        recommendations["technical_enhancements"].append(
            "Increase FPS to at least 24 for smoother playback"
        )
    elif fps < 30 and avg_motion > 0.2:
        recommendations["technical_enhancements"].append(
            "Consider 30 FPS for better motion quality"
        )

    return {
        "enhancement_recommendations": recommendations,
        "quality_metrics": {
            "overall_quality": round(avg_quality, 2),
            "motion_level": round(avg_motion * 100, 1),
            "noise_level": round(avg_noise * 100, 1),
        },
        "technical_details": {
            "resolution": f"{width}x{height}",
            "fps": round(fps, 2),
            "duration": round(duration, 2),
        },
    }


def analyze_video_metadata(video_path: str) -> Dict:
    """
    Analyze video metadata and technical specifications.

    Args:
        video_path: Path to the video file

    Returns:
        Dictionary containing metadata analysis
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video file")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    # Get codec information
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    # Get additional metadata
    brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    contrast = cap.get(cv2.CAP_PROP_CONTRAST)
    saturation = cap.get(cv2.CAP_PROP_SATURATION)
    hue = cap.get(cv2.CAP_PROP_HUE)

    # Calculate file size and bitrate
    file_size = os.path.getsize(video_path)
    file_size_mb = file_size / (1024 * 1024)
    bitrate = (file_size * 8) / duration if duration > 0 else 0

    cap.release()

    # Determine video quality category
    quality_category = "Low"
    if width >= 1920:
        quality_category = "High"
    elif width >= 1280:
        quality_category = "Medium"

    # Determine codec compatibility
    codec_compatibility = {
        "codec": codec,
        "is_web_compatible": codec.lower() in ["avc1", "h264", "hevc", "h265"],
        "is_mobile_compatible": codec.lower() in ["avc1", "h264"],
        "is_streaming_compatible": codec.lower() in ["avc1", "h264", "hevc", "h265"],
    }

    return {
        "metadata_analysis": {
            "quality_category": quality_category,
            "resolution": f"{width}x{height}",
            "aspect_ratio": f"{width/height:.2f}",
            "fps": round(fps, 2),
            "duration": round(duration, 2),
            "frame_count": frame_count,
            "file_size_mb": round(file_size_mb, 2),
            "bitrate_kbps": round(bitrate / 1000, 2),
        },
        "codec_analysis": codec_compatibility,
        "technical_parameters": {
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "saturation": round(saturation, 2),
            "hue": round(hue, 2),
        },
        "compatibility_info": {
            "web_playback": codec_compatibility["is_web_compatible"],
            "mobile_playback": codec_compatibility["is_mobile_compatible"],
            "streaming_ready": codec_compatibility["is_streaming_compatible"],
        },
    }
