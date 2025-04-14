import os
import traceback
import numpy as np
from flask import Blueprint, request, jsonify, current_app
from pathlib import Path

from models.whisper_model import transcriber
from models.summarizer import summarizer
from models.sentiment_model import sentiment_analyzer
from utils.audio_processor import (
    extract_audio_from_video,
    analyze_audio_features,
    determine_content_type,
)
from utils.file_handler import save_uploaded_file, get_file_type, clean_up_files
from utils.video_analyzer import (
    analyze_video_quality,
    analyze_video_compression,
    analyze_video_content,
)
from utils.media_advisor import (
    analyze_streaming_readiness,
    analyze_video_archive,
    detect_duplicate_frames,
    get_conversion_recommendations,
    get_enhancement_recommendations,
    analyze_video_metadata,
    compare_videos,
)

# Create a Blueprint for API routes
api_bp = Blueprint("api", __name__)


def convert_numpy_values(obj):
    """Convert numpy values to Python native types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_values(v) for v in obj]
    elif hasattr(obj, "dtype") and np.issubdtype(obj.dtype, np.floating):
        return float(obj)
    elif hasattr(obj, "dtype") and np.issubdtype(obj.dtype, np.integer):
        return int(obj)
    return obj


@api_bp.route("/upload", methods=["POST"])
def upload_file():
    """Handle file upload and return upload_url (file path)"""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file
    temp_file_path = save_uploaded_file(file, current_app.config["UPLOAD_FOLDER"])
    file_type = get_file_type(temp_file_path)

    return jsonify({"upload_url": temp_file_path, "file_type": file_type})


@api_bp.route("/transcript", methods=["POST"])
def transcribe_audio():
    """Start a new transcription task and return task ID"""
    data = request.json
    if not data or "audio_url" not in data:
        return jsonify({"error": "Missing audio_url parameter"}), 400

    media_path = data["audio_url"]
    task_id = os.path.basename(media_path).split(".")[0]
    file_type = get_file_type(media_path)

    return jsonify({"id": task_id, "status": "queued", "file_type": file_type})


@api_bp.route("/transcript/<task_id>", methods=["GET"])
def get_transcript_status(task_id):
    """Check transcription status and return results"""
    potential_files = [
        f for f in os.listdir(current_app.config["UPLOAD_FOLDER"]) if task_id in f
    ]

    if not potential_files:
        return jsonify({"error": "Task not found"}), 404

    media_path = os.path.join(current_app.config["UPLOAD_FOLDER"], potential_files[0])
    file_type = get_file_type(media_path)
    is_video = file_type == "video"

    try:
        audio_path = media_path

        # If it's a video, extract the audio first
        if is_video:
            audio_path = extract_audio_from_video(media_path)

        # Load and analyze audio
        features, _, _ = analyze_audio_features(audio_path)

        # Transcribe the audio
        transcript = transcriber.transcribe(audio_path)

        # Clean up extracted audio file if this was a video
        if is_video and os.path.exists(audio_path) and audio_path != media_path:
            clean_up_files([audio_path])

        return jsonify(
            {
                "id": task_id,
                "status": "completed",
                "text": transcript,
                "file_type": file_type,
                "audio_features": features,
            }
        )
    except Exception as e:
        print(f"Error in transcription: {str(e)}")
        print(traceback.format_exc())
        return (
            jsonify(
                {
                    "id": task_id,
                    "status": "failed",
                    "file_type": file_type,
                    "error": str(e),
                }
            ),
            500,
        )


@api_bp.route("/summarize", methods=["POST"])
def summarize_text():
    """Generate summary from transcript text"""
    data = request.json
    if not data or "transcript" not in data:
        return jsonify({"error": "Missing transcript parameter"}), 400

    transcript = data["transcript"]
    features = data.get("features", {})

    try:
        # Add sentiment analysis to summarizer if needed
        if not hasattr(summarizer, "sentiment_analyzer"):
            summarizer.sentiment_analyzer = sentiment_analyzer.model

        # Generate summary
        summary = summarizer.generate_summary(transcript, features)
        return jsonify({"summary": summary})
    except Exception as e:
        print(f"Error in summarize_text endpoint: {str(e)}")
        print(traceback.format_exc())
        return (
            jsonify(
                {
                    "summary": f"Failed to generate summary: {str(e)}. The media lasts {features.get('duration', 0):.1f} seconds.",
                    "error": str(e),
                }
            ),
            500,
        )


@api_bp.route("/analyze_media", methods=["POST"])
def analyze_media():
    """Full media analysis pipeline - upload, transcribe, and summarize in one request"""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file
    media_path = save_uploaded_file(file, current_app.config["UPLOAD_FOLDER"])
    file_type = get_file_type(media_path)
    is_video = file_type == "video"

    try:
        audio_path = media_path

        # If it's a video, extract the audio first
        if is_video:
            print(f"Extracting audio from video: {media_path}")
            audio_path = extract_audio_from_video(media_path)
            print(f"Audio extracted to: {audio_path}")

        # Load and analyze audio
        features, _, _ = analyze_audio_features(audio_path)
        features["media_type"] = file_type

        # Transcribe audio
        print(f"Transcribing media file: {audio_path}")
        transcript = transcriber.transcribe(audio_path)
        print(f"Transcription complete, length: {len(transcript)} chars")

        # Determine content type based on audio characteristics
        content_type, word_count, speech_density = determine_content_type(
            transcript, features["duration"]
        )

        # Generate enhanced summary
        print("Generating summary...")
        summary = summarizer.generate_summary(transcript, features)
        print(f"Summary generated, length: {len(summary)} chars")

        # Clean up temporary files
        files_to_clean = []
        if is_video and audio_path != media_path:
            files_to_clean.append(audio_path)
        files_to_clean.append(media_path)
        clean_up_files(files_to_clean)

        response = {
            "mediaType": file_type,
            "contentDescription": transcript or "No speech detected",
            "contentType": content_type,
            "contentSummary": summary,
            "audioFeatures": features,
            "wordCount": word_count,
            "speechDensity": speech_density,
        }

        # Convert numpy values to Python types
        response = convert_numpy_values(response)
        return jsonify(response)

    except Exception as e:
        print("Error in analyze_media:")
        print(traceback.format_exc())
        return (
            jsonify(
                {
                    "mediaType": file_type,
                    "contentDescription": "Media content analysis failed",
                    "contentType": "Unknown",
                    "contentSummary": f"Unable to analyze media content due to an error: {str(e)}",
                    "error": str(e),
                }
            ),
            500,
        )


# Endpoint for backward compatibility
@api_bp.route("/analyze_audio", methods=["POST"])
def analyze_audio():
    """Redirect to analyze_media for backward compatibility"""
    return analyze_media()


@api_bp.route("/analyze_video_quality", methods=["POST"])
def analyze_video_quality_endpoint():
    """Analyze video quality metrics including resolution, bitrate, and compression artifacts"""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file
    video_path = save_uploaded_file(file, current_app.config["UPLOAD_FOLDER"])
    file_type = get_file_type(video_path)

    if file_type != "video":
        clean_up_files([video_path])
        return jsonify({"error": "File is not a video"}), 400

    try:
        # Analyze video quality
        quality_metrics = analyze_video_quality(video_path)
        compression_analysis = analyze_video_compression(video_path)
        content_analysis = analyze_video_content(video_path)

        # Clean up the uploaded file
        clean_up_files([video_path])

        return jsonify(
            {
                "video_quality": quality_metrics,
                "compression_analysis": compression_analysis,
                "content_analysis": content_analysis,
            }
        )
    except Exception as e:
        print(f"Error in video quality analysis: {str(e)}")
        print(traceback.format_exc())
        clean_up_files([video_path])
        return jsonify({"error": str(e)}), 500


@api_bp.route("/inspect_media", methods=["POST"])
def inspect_media():
    """Comprehensive media inspection including quality, content, and technical details"""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file
    media_path = save_uploaded_file(file, current_app.config["UPLOAD_FOLDER"])
    file_type = get_file_type(media_path)

    try:
        inspection_results = {
            "file_type": file_type,
            "file_name": file.filename,
            "file_size_mb": round(os.path.getsize(media_path) / (1024 * 1024), 2),
        }

        if file_type == "video":
            # Video-specific analysis
            inspection_results.update(
                {
                    "video_quality": analyze_video_quality(media_path),
                    "compression_analysis": analyze_video_compression(media_path),
                    "content_analysis": analyze_video_content(media_path),
                }
            )

            # Extract and analyze audio from video
            audio_path = extract_audio_from_video(media_path)
            audio_features, _, _ = analyze_audio_features(audio_path)
            inspection_results["audio_features"] = audio_features
            clean_up_files([audio_path])
        elif file_type == "audio":
            # Audio-specific analysis
            audio_features, _, _ = analyze_audio_features(media_path)
            inspection_results["audio_features"] = audio_features

        # Clean up the uploaded file
        clean_up_files([media_path])

        return jsonify(inspection_results)
    except Exception as e:
        print(f"Error in media inspection: {str(e)}")
        print(traceback.format_exc())
        clean_up_files([media_path])
        return jsonify({"error": str(e)}), 500


@api_bp.route("/streaming_readiness", methods=["POST"])
def check_streaming_readiness():
    """Analyze video for streaming readiness"""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file
    video_path = save_uploaded_file(file, current_app.config["UPLOAD_FOLDER"])
    file_type = get_file_type(video_path)

    if file_type != "video":
        clean_up_files([video_path])
        return jsonify({"error": "File is not a video"}), 400

    try:
        # Analyze streaming readiness
        analysis = analyze_streaming_readiness(video_path)

        # Clean up the uploaded file
        clean_up_files([video_path])

        return jsonify(analysis)
    except Exception as e:
        print(f"Error in streaming readiness analysis: {str(e)}")
        print(traceback.format_exc())
        clean_up_files([video_path])
        return jsonify({"error": str(e)}), 500


@api_bp.route("/archive_analysis", methods=["POST"])
def analyze_archive():
    """Analyze video for archival purposes"""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file
    video_path = save_uploaded_file(file, current_app.config["UPLOAD_FOLDER"])
    file_type = get_file_type(video_path)

    if file_type != "video":
        clean_up_files([video_path])
        return jsonify({"error": "File is not a video"}), 400

    try:
        # Analyze video archive
        analysis = analyze_video_archive(video_path)

        # Clean up the uploaded file
        clean_up_files([video_path])

        return jsonify(analysis)
    except Exception as e:
        print(f"Error in archive analysis: {str(e)}")
        print(traceback.format_exc())
        clean_up_files([video_path])
        return jsonify({"error": str(e)}), 500


@api_bp.route("/detect_duplicates", methods=["POST"])
def check_duplicates():
    """Detect duplicate frames in a video"""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Get threshold from query parameters
    threshold = float(request.args.get("threshold", 0.95))

    # Save the uploaded file
    video_path = save_uploaded_file(file, current_app.config["UPLOAD_FOLDER"])
    file_type = get_file_type(video_path)

    if file_type != "video":
        clean_up_files([video_path])
        return jsonify({"error": "File is not a video"}), 400

    try:
        # Detect duplicate frames
        analysis = detect_duplicate_frames(video_path, threshold)

        # Clean up the uploaded file
        clean_up_files([video_path])

        return jsonify(analysis)
    except Exception as e:
        print(f"Error in duplicate detection: {str(e)}")
        print(traceback.format_exc())
        clean_up_files([video_path])
        return jsonify({"error": str(e)}), 500


@api_bp.route("/conversion_advice", methods=["POST"])
def get_conversion_advice():
    """Get recommendations for video conversion"""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file
    video_path = save_uploaded_file(file, current_app.config["UPLOAD_FOLDER"])
    file_type = get_file_type(video_path)

    if file_type != "video":
        clean_up_files([video_path])
        return jsonify({"error": "File is not a video"}), 400

    try:
        # Get conversion recommendations
        recommendations = get_conversion_recommendations(video_path)

        # Clean up the uploaded file
        clean_up_files([video_path])

        return jsonify(recommendations)
    except Exception as e:
        print(f"Error in conversion advice: {str(e)}")
        print(traceback.format_exc())
        clean_up_files([video_path])
        return jsonify({"error": str(e)}), 500


@api_bp.route("/batch_analyze", methods=["POST"])
def batch_analyze():
    """Analyze multiple media files in a single request"""
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist("files")
    if not files or all(file.filename == "" for file in files):
        return jsonify({"error": "No selected files"}), 400

    results = []
    errors = []

    for file in files:
        if file.filename == "":
            continue

        try:
            # Save the uploaded file
            media_path = save_uploaded_file(file, current_app.config["UPLOAD_FOLDER"])
            file_type = get_file_type(media_path)

            # Perform comprehensive analysis
            analysis_result = {
                "file_name": file.filename,
                "file_type": file_type,
                "file_size_mb": round(os.path.getsize(media_path) / (1024 * 1024), 2),
            }

            if file_type == "video":
                # Video-specific analysis
                analysis_result.update(
                    {
                        "video_quality": analyze_video_quality(media_path),
                        "streaming_readiness": analyze_streaming_readiness(media_path),
                        "archive_analysis": analyze_video_archive(media_path),
                        "duplicate_analysis": detect_duplicate_frames(media_path),
                        "conversion_recommendations": get_conversion_recommendations(
                            media_path
                        ),
                    }
                )

                # Extract and analyze audio from video
                audio_path = extract_audio_from_video(media_path)
                audio_features, _, _ = analyze_audio_features(audio_path)
                analysis_result["audio_features"] = audio_features
                clean_up_files([audio_path])
            elif file_type == "audio":
                # Audio-specific analysis
                audio_features, _, _ = analyze_audio_features(media_path)
                analysis_result["audio_features"] = audio_features

            results.append(analysis_result)

            # Clean up the uploaded file
            clean_up_files([media_path])

        except Exception as e:
            errors.append({"file_name": file.filename, "error": str(e)})
            print(f"Error processing {file.filename}: {str(e)}")
            print(traceback.format_exc())
            if "media_path" in locals():
                clean_up_files([media_path])

    return jsonify(
        {
            "results": results,
            "errors": errors,
            "total_files": len(files),
            "successful_analyses": len(results),
            "failed_analyses": len(errors),
        }
    )


@api_bp.route("/compare_videos", methods=["POST"])
def compare_videos_endpoint():
    """Compare two videos for similarity and differences"""
    if "file1" not in request.files or "file2" not in request.files:
        return jsonify({"error": "Both video files are required"}), 400

    file1 = request.files["file1"]
    file2 = request.files["file2"]

    if file1.filename == "" or file2.filename == "":
        return jsonify({"error": "Both video files must be selected"}), 400

    # Save the uploaded files
    video_path1 = save_uploaded_file(file1, current_app.config["UPLOAD_FOLDER"])
    video_path2 = save_uploaded_file(file2, current_app.config["UPLOAD_FOLDER"])

    file_type1 = get_file_type(video_path1)
    file_type2 = get_file_type(video_path2)

    if file_type1 != "video" or file_type2 != "video":
        clean_up_files([video_path1, video_path2])
        return jsonify({"error": "Both files must be videos"}), 400

    try:
        # Compare videos
        comparison = compare_videos(video_path1, video_path2)

        # Clean up the uploaded files
        clean_up_files([video_path1, video_path2])

        return jsonify(comparison)
    except Exception as e:
        print(f"Error in video comparison: {str(e)}")
        print(traceback.format_exc())
        clean_up_files([video_path1, video_path2])
        return jsonify({"error": str(e)}), 500


@api_bp.route("/enhancement_recommendations", methods=["POST"])
def get_video_enhancement_recommendations():
    """Get recommendations for video enhancement and quality improvement"""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file
    video_path = save_uploaded_file(file, current_app.config["UPLOAD_FOLDER"])
    file_type = get_file_type(video_path)

    if file_type != "video":
        clean_up_files([video_path])
        return jsonify({"error": "File is not a video"}), 400

    try:
        # Get enhancement recommendations
        recommendations = get_enhancement_recommendations(video_path)

        # Clean up the uploaded file
        clean_up_files([video_path])

        return jsonify(recommendations)
    except Exception as e:
        print(f"Error in enhancement recommendations: {str(e)}")
        print(traceback.format_exc())
        clean_up_files([video_path])
        return jsonify({"error": str(e)}), 500


@api_bp.route("/metadata_analysis", methods=["POST"])
def analyze_video_metadata_endpoint():
    """Analyze video metadata and technical specifications"""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file
    video_path = save_uploaded_file(file, current_app.config["UPLOAD_FOLDER"])
    file_type = get_file_type(video_path)

    if file_type != "video":
        clean_up_files([video_path])
        return jsonify({"error": "File is not a video"}), 400

    try:
        # Analyze video metadata
        analysis = analyze_video_metadata(video_path)

        # Clean up the uploaded file
        clean_up_files([video_path])

        return jsonify(analysis)
    except Exception as e:
        print(f"Error in metadata analysis: {str(e)}")
        print(traceback.format_exc())
        clean_up_files([video_path])
        return jsonify({"error": str(e)}), 500


@api_bp.route("/comprehensive_analysis", methods=["POST"])
def comprehensive_video_analysis():
    """Perform comprehensive video analysis including all available metrics"""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file
    video_path = save_uploaded_file(file, current_app.config["UPLOAD_FOLDER"])
    file_type = get_file_type(video_path)

    if file_type != "video":
        clean_up_files([video_path])
        return jsonify({"error": "File is not a video"}), 400

    try:
        # Perform all available analyses
        analysis_results = {
            "basic_info": {
                "file_name": file.filename,
                "file_type": file_type,
                "file_size_mb": round(os.path.getsize(video_path) / (1024 * 1024), 2),
            },
            "video_quality": analyze_video_quality(video_path),
            "streaming_readiness": analyze_streaming_readiness(video_path),
            "archive_analysis": analyze_video_archive(video_path),
            "duplicate_analysis": detect_duplicate_frames(video_path),
            "conversion_recommendations": get_conversion_recommendations(video_path),
            "enhancement_recommendations": get_enhancement_recommendations(video_path),
            "metadata_analysis": analyze_video_metadata(video_path),
        }

        # Extract and analyze audio from video
        audio_path = extract_audio_from_video(video_path)
        audio_features, _, _ = analyze_audio_features(audio_path)
        analysis_results["audio_features"] = audio_features
        clean_up_files([audio_path])

        # Clean up the uploaded file
        clean_up_files([video_path])

        return jsonify(analysis_results)
    except Exception as e:
        print(f"Error in comprehensive analysis: {str(e)}")
        print(traceback.format_exc())
        clean_up_files([video_path])
        return jsonify({"error": str(e)}), 500
