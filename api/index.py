import os
import sys
import urllib.parse

# Ensure repository root is in sys.path for backend and model loading
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.app import app

class VercelRouteMiddleware:
    """
    WSGI Middleware for Vercel Python Serverless Functions.
    Extracts the targeted route from `__route` query parameter passed by
    Vercel's rewrite rule, cleans up the query string, and updates PATH_INFO
    so Flask routes match perfectly.
    """
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        qs = environ.get("QUERY_STRING", "")
        params = urllib.parse.parse_qs(qs, keep_blank_values=True)
        if "__route" in params:
            route = params.pop("__route", [""])[0].strip("/")
            environ["QUERY_STRING"] = urllib.parse.urlencode([(k, v) for k, vs in params.items() for v in vs])
            environ["PATH_INFO"] = f"/api/{route}" if route else "/api/health"
        return self.wsgi_app(environ, start_response)

app.wsgi_app = VercelRouteMiddleware(app.wsgi_app)
