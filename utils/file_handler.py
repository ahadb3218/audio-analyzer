import os
import time
from pathlib import Path
import shutil


def save_uploaded_file(file, upload_folder):
    """Save an uploaded file and return its path"""
    # Create upload directory if it doesn't exist
    os.makedirs(upload_folder, exist_ok=True)

    file_extension = os.path.splitext(file.filename)[1].lower()
    temp_file_path = os.path.join(
        upload_folder, f"upload_{int(time.time())}{file_extension}"
    )
    file.save(temp_file_path)

    return temp_file_path


def get_file_type(file_path):
    """Determine if a file is video or audio based on extension"""
    file_extension = Path(file_path).suffix.lower()
    is_video = file_extension in [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    return "video" if is_video else "audio"


def clean_up_files(files_list):
    """Remove temporary files"""
    for file_path in files_list:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Warning: Failed to remove temporary file {file_path}: {str(e)}")
