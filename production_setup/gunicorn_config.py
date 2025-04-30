import multiprocessing

# Gunicorn configuration
bind = "0.0.0.0:8000"  # Bind to all interfaces on port 8000
workers = multiprocessing.cpu_count() * 2 + 1  # Rule of thumb for number of workers
worker_class = "uvicorn.workers.UvicornWorker"  # Use Uvicorn worker for FastAPI
timeout = 120  # Increase timeout for long-running requests
keepalive = 5  # Number of seconds to wait for requests on a keep-alive connection
max_requests = 1000  # Restart workers after handling this many requests
max_requests_jitter = 50  # Add randomness to max_requests to avoid all workers restarting at once
accesslog = "/var/log/gunicorn/access.log"  # Access log file location
errorlog = "/var/log/gunicorn/error.log"  # Error log file location
loglevel = "info"  # Log level