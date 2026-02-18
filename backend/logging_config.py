"""
SafEye — Structured Logging
JSON logging for production, human-readable for development.
"""

import json
import logging
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON — ideal for log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Merge extra fields (from logger.info("msg", extra={...}))
        for key in ("method", "path", "status", "duration_ms", "remote_addr", "user_id", "error"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val

        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


class PrettyFormatter(logging.Formatter):
    """Coloured, human-readable logs for local development."""

    COLORS = {
        "DEBUG": "\033[36m",     # cyan
        "INFO": "\033[32m",      # green
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
        "CRITICAL": "\033[1;31m",  # bold red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        ts = datetime.now().strftime("%H:%M:%S")
        msg = record.getMessage()
        base = f"{color}{ts} [{record.levelname:>7}]{self.RESET} {record.name}: {msg}"

        # Append extras inline
        extras = []
        for key in ("method", "path", "status", "duration_ms"):
            val = getattr(record, key, None)
            if val is not None:
                extras.append(f"{key}={val}")
        if extras:
            base += f"  ({', '.join(extras)})"

        if record.exc_info and record.exc_info[1]:
            base += "\n" + self.formatException(record.exc_info)

        return base


def setup_logging(level: str = "INFO", fmt: str = "text"):
    """
    Configure the root logger.
    
    Args:
        level: DEBUG, INFO, WARNING, ERROR, CRITICAL
        fmt: "json" for production, "text" for development
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates on reload
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(root.level)

    if fmt == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(PrettyFormatter())

    root.addHandler(handler)

    # Quiet noisy libraries
    for noisy in ("urllib3", "werkzeug", "transformers", "PIL", "tensorflow", "torch"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).debug("Logging configured: level=%s, format=%s", level, fmt)
