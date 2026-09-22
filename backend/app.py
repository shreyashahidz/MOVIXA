import os
import sys

# Ensure root workspace is in sys.path when running app.py directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS

from backend.routes.recommend import recommend_bp
from backend.routes.sentiment import sentiment_bp
from backend.config import FRONTEND_DIR

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

# Register API blueprints for both /api and direct prefix
app.register_blueprint(recommend_bp, url_prefix="/api")
app.register_blueprint(recommend_bp, url_prefix="", name="recommend_root")
app.register_blueprint(sentiment_bp, url_prefix="/api")
app.register_blueprint(sentiment_bp, url_prefix="", name="sentiment_root")


@app.route("/api/health", methods=["GET"])
@app.route("/health", methods=["GET"])
@app.route("/api", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "MOVIXA — Movies. Moods. Moments. Movie Recommendation & Sentiment System",
        "version": "1.0.0"
    })


# Serve Frontend SPA
@app.route("/")
def serve_index():
    if os.path.exists(os.path.join(FRONTEND_DIR, "index.html")):
        return send_from_directory(FRONTEND_DIR, "index.html")
    return jsonify({
        "status": "ok",
        "service": "MOVIXA — Movies. Moods. Moments.",
        "message": "API backend active"
    })


@app.route("/<path:path>")
def serve_static(path):
    if path.startswith("api") or path.startswith("health"):
        # Never treat API or health routes as static files
        return jsonify({"error": "Endpoint not found", "path": path}), 404
    if os.path.exists(os.path.join(FRONTEND_DIR, path)):
        return send_from_directory(FRONTEND_DIR, path)
    if os.path.exists(os.path.join(FRONTEND_DIR, "index.html")):
        return send_from_directory(FRONTEND_DIR, "index.html")
    return jsonify({"error": "Static file not found", "path": path}), 404


@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "Resource not found",
        "path": request.path,
        "full_path": request.full_path,
        "headers": {k: v for k, v in request.headers.items() if any(x in k.lower() for x in ['vercel', 'matched', 'forward', 'host'])},
        "environ": {k: str(request.environ[k]) for k in request.environ if any(x in k.lower() for x in ['matched', 'uri', 'path', 'url'])}
    }), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    print("Starting Movie Recommendation & Sentiment API Server on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
