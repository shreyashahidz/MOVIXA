import os
import pandas as pd
from flask import Blueprint, request, jsonify
from backend.services.sentiment_service import SentimentService
from backend.config import METRICS_LOG_PATH

sentiment_bp = Blueprint("sentiment", __name__)
sentiment_service = SentimentService()


@sentiment_bp.route("/sentiment", methods=["POST"])
def predict_sentiment():
    """
    Predict sentiment for a given review text string.
    Request body:
      { "review": "This movie was incredible..." }
    """
    data = request.get_json(silent=True) or {}
    text = data.get("review", "")
    
    if not text or not text.strip():
        return jsonify({"error": "Empty review text provided"}), 400

    result = sentiment_service.predict_sentiment(text)
    return jsonify(result)


@sentiment_bp.route("/movie/<int:movie_id>/reviews", methods=["GET"])
def get_movie_reviews(movie_id):
    """
    Fetch stored reviews + sentiment tags for a specific movie.
    """
    reviews = sentiment_service.get_movie_reviews(movie_id)
    return jsonify(reviews)


@sentiment_bp.route("/movie/<int:movie_id>/reviews", methods=["POST"])
def add_movie_review(movie_id):
    """
    Submit a user review for a movie, score its sentiment live, and store it.
    Request body:
      { "review": "Loved the cinematography!", "user": "Alex" }
    """
    data = request.get_json(silent=True) or {}
    text = data.get("review", "")
    user = data.get("user", "Movie Enthusiast")

    if not text or not text.strip():
        return jsonify({"error": "Review content cannot be empty"}), 400

    saved_review = sentiment_service.add_movie_review(movie_id, text, user_name=user)
    return jsonify(saved_review), 201


@sentiment_bp.route("/metrics", methods=["GET"])
def get_metrics():
    """
    Retrieve logged validation metrics from multiple runs.
    """
    if os.path.exists(METRICS_LOG_PATH):
        try:
            df = pd.read_csv(METRICS_LOG_PATH)
            return jsonify(df.to_dict(orient="records"))
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify([])
