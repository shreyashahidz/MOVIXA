from flask import Blueprint, request, jsonify
from backend.services.recommender import RecommenderService

recommend_bp = Blueprint("recommend", __name__)
recommender_service = RecommenderService()


@recommend_bp.route("/featured", methods=["GET"])
def get_featured():
    """
    Get top visually stunning movies for the landing hero auto-scroll slider.
    Query params: limit, industry (all, bollywood, hollywood)
    """
    limit = min(20, max(3, int(request.args.get("limit", 8))))
    industry = request.args.get("industry", "all").strip().lower()
    movies = recommender_service.get_featured_carousel(limit=limit, industry=industry)
    return jsonify(movies)


@recommend_bp.route("/latest", methods=["GET"])
def get_latest():
    """
    Get latest releases for the 'LATEST RELEASE' carousel row.
    Query params: limit, industry (all, bollywood, hollywood)
    """
    limit = min(30, max(5, int(request.args.get("limit", 15))))
    industry = request.args.get("industry", "all").strip().lower()
    movies = recommender_service.get_latest_releases(limit=limit, industry=industry)
    return jsonify(movies)


@recommend_bp.route("/trending", methods=["GET"])
def get_trending():
    """
    Get top trending movies with ranks for the 'TOP TRENDING' carousel row.
    Query params: limit, industry (all, bollywood, hollywood)
    """
    limit = min(30, max(5, int(request.args.get("limit", 15))))
    industry = request.args.get("industry", "all").strip().lower()
    movies = recommender_service.get_top_trending(limit=limit, industry=industry)
    return jsonify(movies)


@recommend_bp.route("/movies/by-ids", methods=["GET", "POST"])
def get_movies_by_ids():
    """
    Get full movie details for a list of movie IDs (used for Watchlist).
    """
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        ids = data.get("ids", [])
    else:
        ids_str = request.args.get("ids", "")
        ids = [i.strip() for i in ids_str.split(",") if i.strip()]

    movies = recommender_service.get_movies_by_ids(ids)
    return jsonify(movies)


@recommend_bp.route("/movies", methods=["GET"])
def get_movies():
    """
    Search and list movies with pagination, genre, and industry filtering.
    """
    query = request.args.get("q", "").strip()
    genre = request.args.get("genre", "").strip()
    industry = request.args.get("industry", "all").strip().lower()
    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1
    try:
        limit = min(50, max(1, int(request.args.get("limit", 20))))
    except ValueError:
        limit = 20

    data = recommender_service.get_movies(query=query, genre=genre, industry=industry, page=page, limit=limit)
    return jsonify(data)


# Robust handlers for both /api/movie/<id> and /api/movies/<id>
@recommend_bp.route("/movie/<movie_id>", methods=["GET"])
@recommend_bp.route("/movies/<movie_id>", methods=["GET"])
def get_movie(movie_id):
    """
    Get detailed information for a single movie by ID or Title.
    """
    movie = recommender_service.get_movie_by_id(movie_id)
    if not movie:
        return jsonify({"error": "Movie not found"}), 404
    return jsonify(movie)


@recommend_bp.route("/recommend/<movie_id>", methods=["GET"])
@recommend_bp.route("/recommendations/<movie_id>", methods=["GET"])
def get_recommendations(movie_id):
    """
    Get cluster-based recommendations for a selected movie.
    """
    try:
        top_n = min(20, max(1, int(request.args.get("top_n", 10))))
    except ValueError:
        top_n = 10

    recs = recommender_service.get_recommendations(movie_id, top_n=top_n)
    return jsonify(recs)


@recommend_bp.route("/genres", methods=["GET"])
def get_genres():
    """
    Retrieve list of available unique movie genres for filter chips.
    """
    genres = recommender_service.get_genres()
    return jsonify(genres)
