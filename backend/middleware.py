"""
SafEye — Security Middleware
Production-grade security headers, rate limiting, input validation, and request logging.
"""

import os
import re
import time
import uuid
import logging
import hashlib
from datetime import datetime
from functools import wraps
from collections import defaultdict
from threading import Lock, Thread

from flask import request, jsonify, g, current_app
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  SECURITY HEADERS
# ═══════════════════════════════════════════════════════════

def add_security_headers(response):
    """Inject production security headers into every response."""
    headers = current_app.config.get("SECURITY_HEADERS", {})
    for key, value in headers.items():
        response.headers.setdefault(key, value)

    # HSTS only when served over HTTPS
    if request.is_secure:
        response.headers.setdefault(
            "Strict-Transport-Security",
            "max-age=31536000; includeSubDomains",
        )

    # Remove server banner
    response.headers.pop("Server", None)
    return response


# ═══════════════════════════════════════════════════════════
#  IN-PROCESS RATE LIMITER  (no Redis dependency)
# ═══════════════════════════════════════════════════════════

class InMemoryRateLimiter:
    """
    Simple sliding-window rate limiter backed by a dict.
    Sufficient for single-process / low-traffic deployments.
    For multi-process, swap for Flask-Limiter + Redis.
    """

    def __init__(self):
        self._buckets: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    @staticmethod
    def _parse_limit(limit_str: str) -> tuple[int, int]:
        """Parse '30/minute' → (max_requests=30, window_seconds=60)."""
        parts = limit_str.strip().split("/")
        count = int(parts[0])
        unit = parts[1].lower() if len(parts) > 1 else "hour"
        windows = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}
        return count, windows.get(unit, 3600)

    def _client_key(self) -> str:
        """Fingerprint the client: IP + optional auth user."""
        ip = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown")
        ip = ip.split(",")[0].strip()
        user = g.get("current_user_email", "anon")
        raw = f"{ip}:{user}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def is_allowed(self, limit_str: str) -> bool:
        max_req, window = self._parse_limit(limit_str)
        key = self._client_key()
        now = time.time()

        with self._lock:
            bucket = self._buckets[key]
            # Prune expired entries
            self._buckets[key] = [t for t in bucket if now - t < window]
            if len(self._buckets[key]) >= max_req:
                return False
            self._buckets[key].append(now)
            return True

    def cleanup(self, max_age: int = 7200):
        """Periodically call to free memory from stale keys."""
        now = time.time()
        with self._lock:
            stale = [k for k, v in self._buckets.items() if not v or now - v[-1] > max_age]
            for k in stale:
                del self._buckets[k]


_limiter = InMemoryRateLimiter()

# Periodic cleanup of stale rate limit buckets (every 60 minutes)
import atexit

def _schedule_limiter_cleanup():
    """Run rate limiter cleanup every 60 minutes in a daemon thread."""
    import time as _time
    def _cleanup_loop():
        while True:
            _time.sleep(3600)
            try:
                _limiter.cleanup()
            except Exception:
                pass
    t = Thread(target=_cleanup_loop, daemon=True)
    t.start()

_schedule_limiter_cleanup()


def rate_limit(limit_str: str | None = None):
    """Decorator that applies rate limiting to an endpoint."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not current_app.config.get("RATELIMIT_ENABLED", True):
                return fn(*args, **kwargs)
            actual_limit = limit_str or current_app.config.get("RATELIMIT_DEFAULT", "200/hour")
            if not _limiter.is_allowed(actual_limit):
                logger.warning("Rate limit exceeded for %s on %s", _limiter._client_key(), request.path)
                return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429
            return fn(*args, **kwargs)
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════
#  INPUT VALIDATION HELPERS
# ═══════════════════════════════════════════════════════════

def validate_file_upload(field: str = "file", allowed_extensions: set | None = None):
    """
    Validate an uploaded file. Returns (file, error_response).
    On success error_response is None; on failure file is None.
    """
    if field not in request.files:
        return None, (jsonify({"error": f"No '{field}' field in request"}), 400)

    file = request.files[field]
    if file.filename == "" or file.filename is None:
        return None, (jsonify({"error": "No file selected"}), 400)

    filename = secure_filename(file.filename)
    if not filename:
        return None, (jsonify({"error": "Invalid filename"}), 400)

    if allowed_extensions:
        ext = os.path.splitext(filename)[1].lower().lstrip(".")
        if ext not in allowed_extensions:
            return None, (
                jsonify({"error": f"File type '.{ext}' not allowed. Accepted: {', '.join(sorted(allowed_extensions))}"}),
                400,
            )

    return file, None


# ── Text sanitization ──────────────────────────────────────

# Compiled once for performance
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_BLOCK_RE = re.compile(r"<script[\s\S]*?</script>", re.IGNORECASE)
_STYLE_BLOCK_RE = re.compile(r"<style[\s\S]*?</style>", re.IGNORECASE)
_EVENT_HANDLER_RE = re.compile(r"\bon\w+\s*=", re.IGNORECASE)


def sanitize_text(text: str) -> str:
    """
    Strip HTML tags, <script>/<style> blocks, and inline event handlers
    from user-supplied text to prevent stored-XSS and log injection.
    """
    if not isinstance(text, str):
        return text
    text = _SCRIPT_BLOCK_RE.sub("", text)
    text = _STYLE_BLOCK_RE.sub("", text)
    text = _HTML_TAG_RE.sub("", text)
    text = _EVENT_HANDLER_RE.sub("", text)
    return text.strip()


def validate_json_body(*required_fields: str, sanitize: bool = True):
    """
    Validate that the request body is JSON and contains the required fields.
    When *sanitize* is True (default), string values in the required fields
    are run through sanitize_text() to strip HTML/script content.
    Returns (data, error_response).
    """
    data = request.get_json(silent=True)
    if data is None:
        return None, (jsonify({"error": "Request body must be valid JSON"}), 400)

    missing = [f for f in required_fields if not data.get(f)]
    if missing:
        return None, (jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400)

    if sanitize:
        for field in required_fields:
            if isinstance(data.get(field), str):
                data[field] = sanitize_text(data[field])

    return data, None


# ═══════════════════════════════════════════════════════════
#  REQUEST LOGGING
# ═══════════════════════════════════════════════════════════

def log_request():
    """Record request start time and assign a unique request ID for tracing."""
    g.request_start = time.time()
    # Use client-provided ID or generate a new UUID4
    g.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())


def log_response(response):
    """Log request completion with latency and attach X-Request-ID header."""
    request_id = getattr(g, "request_id", str(uuid.uuid4()))
    response.headers["X-Request-ID"] = request_id

    duration = time.time() - getattr(g, "request_start", time.time())
    if request.path.startswith("/api/"):
        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.path,
                "status": response.status_code,
                "duration_ms": round(duration * 1000, 1),
                "remote_addr": request.headers.get("X-Forwarded-For", request.remote_addr),
            },
        )
    return response


# ═══════════════════════════════════════════════════════════
#  REGISTER MIDDLEWARE
# ═══════════════════════════════════════════════════════════

def init_security(app):
    """Attach all security middleware to a Flask app."""
    app.after_request(add_security_headers)
    app.before_request(log_request)
    app.after_request(log_response)
    logger.info("Security middleware initialised (headers, rate limiting, request logging)")
