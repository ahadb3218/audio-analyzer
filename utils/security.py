from flask import request, current_app
from functools import wraps
import time
from typing import Dict, Tuple
import hashlib
import os
from config import Config


class RateLimiter:
    """Rate limiter implementation"""

    def __init__(self, max_requests: int = 100, window: int = 60):
        self.max_requests = max_requests
        self.window = window
        self.requests: Dict[str, list] = {}

    def is_rate_limited(self, key: str) -> Tuple[bool, int]:
        """
        Check if request should be rate limited.

        Args:
            key: Unique identifier for the client

        Returns:
            Tuple of (is_limited, remaining_requests)
        """
        current_time = time.time()

        # Clean old requests
        if key in self.requests:
            self.requests[key] = [
                req_time
                for req_time in self.requests[key]
                if current_time - req_time < self.window
            ]
        else:
            self.requests[key] = []

        # Check if rate limited
        if len(self.requests[key]) >= self.max_requests:
            return True, 0

        # Add new request
        self.requests[key].append(current_time)
        remaining = self.max_requests - len(self.requests[key])

        return False, remaining


# Create rate limiter instance
rate_limiter = RateLimiter()


def rate_limit(f):
    """Rate limiting decorator"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get client identifier (IP address or API key)
        client_id = request.remote_addr
        if "X-API-Key" in request.headers:
            client_id = request.headers["X-API-Key"]

        # Check rate limit
        is_limited, remaining = rate_limiter.is_rate_limited(client_id)

        if is_limited:
            return {
                "error": "Rate limit exceeded",
                "retry_after": rate_limiter.window,
            }, 429

        # Add remaining requests to response headers
        response = f(*args, **kwargs)
        if isinstance(response, tuple):
            response_obj, status_code = response
        else:
            response_obj, status_code = response, 200

        if isinstance(response_obj, dict):
            response_obj["X-RateLimit-Remaining"] = remaining
            response_obj["X-RateLimit-Reset"] = int(time.time() + rate_limiter.window)

        return response_obj, status_code

    return decorated_function


def secure_filename(filename: str) -> str:
    """
    Generate a secure filename.

    Args:
        filename: Original filename

    Returns:
        Secure filename
    """
    # Get file extension
    ext = filename.rsplit(".", 1)[1].lower() if "." in filename else ""

    # Generate random name
    random_name = hashlib.sha256(os.urandom(32)).hexdigest()

    # Combine with extension
    return f"{random_name}.{ext}"


def validate_api_key(api_key: str) -> bool:
    """
    Validate API key.

    Args:
        api_key: API key to validate

    Returns:
        True if API key is valid
    """
    # In production, implement proper API key validation
    # This is just a placeholder implementation
    return bool(api_key and len(api_key) >= 32)


def require_api_key(f):
    """Decorator to require API key"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get("X-API-Key")

        if not api_key:
            return {"error": "API key required"}, 401

        if not validate_api_key(api_key):
            return {"error": "Invalid API key"}, 403

        return f(*args, **kwargs)

    return decorated_function


def sanitize_file_path(file_path: str) -> str:
    """
    Sanitize file path to prevent directory traversal.

    Args:
        file_path: Path to sanitize

    Returns:
        Sanitized path
    """
    # Convert to absolute path
    abs_path = os.path.abspath(file_path)

    # Ensure path is within upload directory
    if not abs_path.startswith(os.path.abspath(Config.UPLOAD_FOLDER)):
        raise ValueError("Invalid file path")

    return abs_path
