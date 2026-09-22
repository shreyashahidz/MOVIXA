import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
PUBLIC_DIR = os.path.join(BASE_DIR, "public")
FRONTEND_DIR = PUBLIC_DIR if os.path.isdir(PUBLIC_DIR) else os.path.join(BASE_DIR, "frontend")

MOVIES_CLEAN_PATH = os.path.join(DATA_DIR, "processed", "movies_clean.csv")
REVIEWS_CLEAN_PATH = os.path.join(DATA_DIR, "processed", "reviews_clean.csv")
USER_REVIEWS_PATH = os.path.join(DATA_DIR, "processed", "user_reviews.json")

CLUSTER_MODEL_PATH = os.path.join(MODELS_DIR, "cluster_model.pkl")
SVD_PATH = os.path.join(MODELS_DIR, "svd.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
COMBINED_FEATURES_PATH = os.path.join(MODELS_DIR, "combined_features.npy")

SENTIMENT_MODEL_PATH = os.path.join(MODELS_DIR, "sentiment_model.pkl")
TFIDF_VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
METRICS_LOG_PATH = os.path.join(MODELS_DIR, "metrics_log.csv")
ELBOW_PLOT_PATH = os.path.join(MODELS_DIR, "elbow_plot.png")
