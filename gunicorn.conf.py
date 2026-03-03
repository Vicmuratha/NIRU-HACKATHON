"""
SafEye — Gunicorn Production Configuration v3.2
Usage:  gunicorn -c gunicorn.conf.py app:app
"""

import multiprocessing
import os

# ─── Bind ───
bind = f"{os.getenv('HOST', '0.0.0.0')}:{os.getenv('PORT', '7860')}"

# ─── Workers ───
# For ML workloads: keep workers low (models are memory-heavy).
# Use threads for concurrency within each worker.
workers = int(os.getenv("WORKERS", min(2, multiprocessing.cpu_count())))
threads = int(os.getenv("THREADS", 4))
worker_class = "gthread"

# ─── Timeouts ───
# Long timeout because first request loads AI models (~30-60s)
timeout = int(os.getenv("TIMEOUT", 600))
graceful_timeout = 30
keepalive = 5

# ─── Request limits ───
max_requests = 1000          # Recycle workers to prevent memory leaks
max_requests_jitter = 50     # Stagger restarts
limit_request_line = 8190
limit_request_fields = 100

# ─── Temporary file uploads ───
# Large media uploads (images, audio) use temp files instead of memory
tmp_upload_dir = os.getenv("TMP_UPLOAD_DIR", "/tmp/safeye-uploads")
worker_tmp_dir = "/dev/shm"  # Use RAM-backed filesystem for worker heartbeats

# ─── Logging ───
accesslog = "-"              # stdout
errorlog = "-"               # stderr
loglevel = os.getenv("LOG_LEVEL", "info").lower()
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" %(D)sμs'

# ─── Process naming ───
proc_name = "safeye"

# ─── Preload ───
# Preload app so models are loaded once in the master, then forked.
# Saves memory when running >1 worker.
preload_app = True

# ─── Server header ───
# Don't leak framework version
# (handled at response level too, but this covers edge cases)

# ─── Forwarded headers ───
# Trust X-Forwarded-* headers from reverse proxy (nginx, Cloudflare, etc.)
forwarded_allow_ips = os.getenv("FORWARDED_ALLOW_IPS", "*")
secure_scheme_headers = {
    "X-Forwarded-Proto": "https",
}


# ─── Hooks ───
def on_starting(server):
    server.log.info("SafEye v3.2 production server starting...")
    # Ensure temp upload directory exists
    os.makedirs(tmp_upload_dir, exist_ok=True)


def post_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)


def worker_exit(server, worker):
    server.log.info("Worker exited (pid: %s)", worker.pid)


def worker_abort(server, worker):
    server.log.warning("Worker ABORTED (pid: %s) — possible timeout", worker.pid)
