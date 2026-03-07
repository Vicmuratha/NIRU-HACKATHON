"""
SafEye — Application Configuration
Centralised, environment-aware configuration for all deployment targets.
"""

import os
import secrets
from datetime import timedelta


class _Base:
    """Shared settings across all environments."""

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ── Flask core ──
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY") or os.getenv("SECRET_KEY", secrets.token_hex(32))
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_hex(32))
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)

    # ── Session / cookies ──
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_PATH = "/"
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)

    # ── CORS ──
    FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

    # ── Upload limits ──
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 50 * 1024 * 1024))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, os.getenv("UPLOAD_FOLDER", "uploads"))
    ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff"}
    ALLOWED_AUDIO_EXTENSIONS = {"wav", "mp3", "ogg", "flac", "m4a", "aac"}
    ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
    ALLOWED_DOC_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS  # documents are images for OCR
    MAX_VIDEO_DURATION = int(os.getenv("MAX_VIDEO_DURATION", 30))  # seconds
    VIDEO_KEYFRAMES = int(os.getenv("VIDEO_KEYFRAMES", 5))  # frames to extract

    # ── Models ──
    MODELS_DIR = os.path.join(BASE_DIR, os.getenv("MODELS_DIR", "models"))
    DOWNLOAD_MODELS_ON_STARTUP = os.getenv("DOWNLOAD_MODELS_ON_STARTUP", "false").strip().lower() in {"1", "true", "yes"}

    # ── Database ──
    DATABASE_PATH = os.path.join(BASE_DIR, "users.db")

    # ── Logging ──
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FORMAT = os.getenv("LOG_FORMAT", "text")  # "text" or "json"

    # ── Rate limiting ──
    RATELIMIT_ENABLED = os.getenv("RATELIMIT_ENABLED", "true").lower() in {"1", "true", "yes"}
    RATELIMIT_DEFAULT = os.getenv("RATELIMIT_DEFAULT", "200/hour")
    RATELIMIT_ANALYSIS = os.getenv("RATELIMIT_ANALYSIS", "30/minute")

    # ── Security headers ──
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
        "Content-Security-Policy": (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: https:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'"
        ),
    }

    # ── Version ──
    APP_VERSION = "3.4.0"


class DevelopmentConfig(_Base):
    """Local development — verbose logging, debug mode."""
    DEBUG = True
    TESTING = False
    SESSION_COOKIE_SECURE = False
    LOG_LEVEL = "DEBUG"
    LOG_FORMAT = "text"


class ProductionConfig(_Base):
    """Production — strict security, JSON logs, no debug."""
    DEBUG = False
    TESTING = False
    SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "true").lower() in {"1", "true", "yes"}
    LOG_FORMAT = os.getenv("LOG_FORMAT", "json")

    def __init__(self):
        super().__init__()
        # Warn if using default secret keys
        if self.SECRET_KEY == "super_secret_hackathon_key":
            import warnings
            warnings.warn(
                "⚠️  FLASK_SECRET_KEY is set to the default value. "
                "Set a strong random key via the FLASK_SECRET_KEY env var.",
                UserWarning, stacklevel=2,
            )


class TestingConfig(_Base):
    """Automated tests — temp DB file, fast, deterministic."""
    DEBUG = False
    TESTING = True
    SESSION_COOKIE_SECURE = False
    DATABASE_PATH = os.path.join(
        _Base.BASE_DIR, "test_users.db"
    )
    RATELIMIT_ENABLED = False


_configs = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}


def get_config():
    """Return a config instance matching FLASK_ENV (default: production)."""
    env = os.getenv("FLASK_ENV", "production").lower()
    cls = _configs.get(env, ProductionConfig)
    return cls()
