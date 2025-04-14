from typing import Dict, Any, List
from werkzeug.utils import secure_filename
import os
from config import Config


class ValidationError(Exception):
    """Custom exception for validation errors"""

    pass


def validate_file(file: Any, allowed_extensions: Dict[str, set]) -> Dict[str, Any]:
    """
    Validate uploaded file.

    Args:
        file: File object from request
        allowed_extensions: Dictionary of allowed extensions by file type

    Returns:
        Dictionary containing validated file information

    Raises:
        ValidationError: If file validation fails
    """
    if not file:
        raise ValidationError("No file provided")

    if file.filename == "":
        raise ValidationError("No selected file")

    # Secure the filename
    filename = secure_filename(file.filename)

    # Get file extension
    ext = filename.rsplit(".", 1)[1].lower() if "." in filename else ""

    # Check if extension is allowed
    file_type = None
    for type_name, extensions in allowed_extensions.items():
        if ext in extensions:
            file_type = type_name
            break

    if not file_type:
        raise ValidationError(
            f"File type not allowed. Allowed types: {allowed_extensions}"
        )

    return {"filename": filename, "extension": ext, "type": file_type}


def validate_video_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate video analysis parameters.

    Args:
        params: Dictionary of parameters to validate

    Returns:
        Dictionary containing validated parameters

    Raises:
        ValidationError: If parameter validation fails
    """
    validated = {}

    # Validate threshold parameters
    if "threshold" in params:
        try:
            threshold = float(params["threshold"])
            if not 0 <= threshold <= 1:
                raise ValidationError("Threshold must be between 0 and 1")
            validated["threshold"] = threshold
        except ValueError:
            raise ValidationError("Threshold must be a number")

    # Validate quality parameters
    if "quality_threshold" in params:
        try:
            quality = float(params["quality_threshold"])
            if quality < 0:
                raise ValidationError("Quality threshold cannot be negative")
            validated["quality_threshold"] = quality
        except ValueError:
            raise ValidationError("Quality threshold must be a number")

    return validated


def validate_batch_request(files: List[Any]) -> List[Dict[str, Any]]:
    """
    Validate batch analysis request.

    Args:
        files: List of file objects from request

    Returns:
        List of dictionaries containing validated file information

    Raises:
        ValidationError: If batch validation fails
    """
    if not files:
        raise ValidationError("No files provided")

    if all(file.filename == "" for file in files):
        raise ValidationError("No selected files")

    validated_files = []
    for file in files:
        if file.filename == "":
            continue
        validated_files.append(validate_file(file, Config.ALLOWED_EXTENSIONS))

    return validated_files


def validate_file_path(file_path: str) -> bool:
    """
    Validate file path exists and is within allowed directory.

    Args:
        file_path: Path to validate

    Returns:
        True if path is valid

    Raises:
        ValidationError: If path validation fails
    """
    if not file_path:
        raise ValidationError("File path cannot be empty")

    if not os.path.exists(file_path):
        raise ValidationError("File does not exist")

    # Check if file is within upload directory
    upload_dir = os.path.abspath(Config.UPLOAD_FOLDER)
    file_path = os.path.abspath(file_path)

    if not file_path.startswith(upload_dir):
        raise ValidationError("File path is outside allowed directory")

    return True
