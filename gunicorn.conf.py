"""
SafEye — Gunicorn Production Configuration
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

# ─── Hooks ───
def on_starting(server):
    server.log.info("SafEye production server starting...")


def post_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)


def worker_exit(server, worker):
    server.log.info("Worker exited (pid: %s)", worker.pid)
