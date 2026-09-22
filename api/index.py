import os
import sys

# Ensure repository root is in sys.path for backend and model loading
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.app import app

class VercelFixMiddleware:
    """
    WSGI Middleware for Vercel Python Serverless Functions.
    Vercel sets the rewritten request path to `api/index.py`,
    while placing the real client request path (e.g. `/api/movies?page=1`)
    in `HTTP_X_MATCHED_PATH` or `REQUEST_URI`.
    This middleware restores the original PATH_INFO and QUERY_STRING so
    Flask routes match effortlessly.
    """
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        matched = environ.get("HTTP_X_MATCHED_PATH") or environ.get("REQUEST_URI")
        if matched:
            if "?" in matched:
                path, qs = matched.split("?", 1)
                environ["PATH_INFO"] = path
                if not environ.get("QUERY_STRING"):
                    environ["QUERY_STRING"] = qs
            else:
                environ["PATH_INFO"] = matched
        return self.wsgi_app(environ, start_response)

app.wsgi_app = VercelFixMiddleware(app.wsgi_app)
