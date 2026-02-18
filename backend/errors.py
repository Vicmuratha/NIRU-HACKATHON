"""
SafEye — Centralised Error Handling
Uniform JSON error responses for all API endpoints.
"""

import logging
import traceback
from flask import jsonify, request

logger = logging.getLogger(__name__)


class SafEyeError(Exception):
    """Base application error with status code and payload."""

    def __init__(self, message: str, status_code: int = 400, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class ValidationError(SafEyeError):
    """Input validation failed."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, status_code=400, details=details)


class AuthenticationError(SafEyeError):
    """Authentication required or failed."""

    def __init__(self, message: str = "Authentication required"):
        super().__init__(message, status_code=401)


class NotFoundError(SafEyeError):
    """Resource not found."""

    def __init__(self, message: str = "Resource not found"):
        super().__init__(message, status_code=404)


class AnalysisError(SafEyeError):
    """Media analysis failed."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, status_code=500, details=details)


def _error_response(message: str, status_code: int, details: dict | None = None):
    """Build a uniform JSON error body."""
    body = {
        "error": message,
        "status": status_code,
    }
    if details:
        body["details"] = details
    return jsonify(body), status_code


def init_error_handlers(app):
    """Register all error handlers on the Flask app."""

    @app.errorhandler(SafEyeError)
    def handle_app_error(exc):
        logger.warning("Application error: %s (status=%d)", exc.message, exc.status_code)
        return _error_response(exc.message, exc.status_code, exc.details)

    @app.errorhandler(400)
    def handle_400(exc):
        return _error_response("Bad request", 400)

    @app.errorhandler(401)
    def handle_401(exc):
        return _error_response("Authentication required", 401)

    @app.errorhandler(403)
    def handle_403(exc):
        return _error_response("Forbidden", 403)

    @app.errorhandler(404)
    def handle_404(exc):
        # Don't return JSON 404 for HTML pages
        if request.path.startswith("/api/"):
            return _error_response("Endpoint not found", 404)
        return exc  # Let Flask handle template 404s

    @app.errorhandler(405)
    def handle_405(exc):
        return _error_response("Method not allowed", 405)

    @app.errorhandler(413)
    def handle_413(exc):
        return _error_response("File too large. Maximum size is 50 MB.", 413)

    @app.errorhandler(429)
    def handle_429(exc):
        return _error_response("Rate limit exceeded. Please try again later.", 429)

    @app.errorhandler(500)
    def handle_500(exc):
        logger.error(
            "Unhandled server error on %s %s: %s",
            request.method, request.path, exc,
            exc_info=True,
        )
        return _error_response("Internal server error", 500)

    @app.errorhandler(Exception)
    def handle_unhandled(exc):
        logger.critical(
            "Unhandled exception on %s %s: %s\n%s",
            request.method, request.path, exc,
            traceback.format_exc(),
        )
        return _error_response("An unexpected error occurred", 500)

    logger.info("Error handlers registered")
