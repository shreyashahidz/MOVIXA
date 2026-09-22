import os
import json
import ast
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.sparse import hstack

from backend.utils.preprocessing import create_tags, clean_text


def parse_genres_readable(val):
    if not val or not isinstance(val, str) or val.strip() == "[]":
        return "Unknown"
    try:
        items = json.loads(val)
        names = [item.get("name") for item in items if "name" in item]
        return ", ".join(names) if names else "Unknown"
    except Exception:
        pass
    try:
        items = ast.literal_eval(val)
        names = [item.get("name") for item in items if isinstance(item, dict) and "name" in item]
        return ", ".join(names) if names else "Unknown"
    except Exception:
        return "Unknown"


def main():
    print("=== Training Clustering Model (Recommender Module) ===")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(base_dir, "data", "raw", "tmdb_5000_movies.csv")
    proc_dir = os.path.join(base_dir, "data", "processed")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print(f"Loading raw movies from {raw_path}...")
    df = pd.read_csv(raw_path)
    print(f"Original shape: {df.shape}")

    # Drop nulls in core fields
    df = df.dropna(subset=["title", "overview", "genres"]).copy()
    df["overview"] = df["overview"].fillna("")
    df["genres"] = df["genres"].fillna("[]")
    df["keywords"] = df["keywords"].fillna("[]")
    df["vote_average"] = df["vote_average"].fillna(0.0)
    df["popularity"] = df["popularity"].fillna(0.0)
    df["release_date"] = df["release_date"].fillna("N/A")

    print(f"Shape after filtering: {df.shape}")

    # Create formatted genres and tags
    print("Building composite metadata tags...")
    df["genres_clean"] = df["genres"].apply(parse_genres_readable)
    df["tags"] = [
        create_tags(ov, g, kw)
        for ov, g, kw in zip(df["overview"], df["genres"], df["keywords"])
    ]

    # TF-IDF Vectorization
    print("Computing TF-IDF on tags...")
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["tags"])
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    # Dimensionality Reduction with TruncatedSVD
    print("Applying TruncatedSVD(n_components=50)...")
    svd = TruncatedSVD(n_components=50, random_state=42)
    svd_features = svd.fit_transform(tfidf_matrix)
    explained_var = svd.explained_variance_ratio_.sum()
    print(f"Explained variance with 50 components: {explained_var:.4f}")

    # Scaling numeric features
    print("Scaling numeric features (vote_average, popularity)...")
    scaler = StandardScaler()
    num_features = scaler.fit_transform(df[["vote_average", "popularity"]])

    # Combine SVD text features + scaled numeric features
    combined_features = np.hstack([svd_features, num_features])
    print(f"Combined feature matrix shape: {combined_features.shape}")

    # Elbow Method & Silhouette Evaluation
    print("Running Elbow & Silhouette analysis for k in [2..18]...")
    k_range = range(2, 19, 2)
    inertias = []
    silhouettes = []
    for k in k_range:
        km_test = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = km_test.fit_predict(combined_features)
        inertias.append(km_test.inertia_)
        sil = silhouette_score(combined_features, labels, sample_size=1500, random_state=42)
        silhouettes.append(sil)
        print(f"  k={k:02d} | Inertia: {km_test.inertia_:,.1f} | Silhouette: {sil:.4f}")

    # Plot Elbow Curve and Silhouette Scores
    plot_path = os.path.join(models_dir, "elbow_plot.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(k_range, inertias, "o-", color="#e50914", linewidth=2, markersize=6)
    ax1.set_title("Elbow Method (Inertia vs. Clusters)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia (Sum of Squared Distances)")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(k_range, silhouettes, "s-", color="#f5c518", linewidth=2, markersize=6)
    ax2.set_title("Silhouette Score vs. Clusters", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Coefficient")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved cluster evaluation plot to {plot_path}")

    # Final KMeans model with optimal k=15 (as defined in playbook)
    optimal_k = 15
    print(f"Training final KMeans model with k={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(combined_features)

    # Save models and features
    print("Saving model artifacts...")
    joblib.dump(kmeans, os.path.join(models_dir, "cluster_model.pkl"))
    joblib.dump(svd, os.path.join(models_dir, "svd.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(tfidf, os.path.join(models_dir, "movie_tfidf.pkl"))
    np.save(os.path.join(models_dir, "combined_features.npy"), combined_features)

    # Save cleaned movies dataframe
    clean_cols = [
        "id", "title", "genres_clean", "overview", "vote_average", 
        "vote_count", "popularity", "release_date", "cluster", "tagline"
    ]
    movies_clean_path = os.path.join(proc_dir, "movies_clean.csv")
    df[clean_cols].to_csv(movies_clean_path, index=False)
    print(f"Saved cleaned movies ({len(df)} records) to {movies_clean_path}")

    # Summary of clusters
    cluster_counts = df["cluster"].value_counts().sort_index()
    print("\nCluster Distribution:")
    for c_id, count in cluster_counts.items():
        sample_titles = df[df["cluster"] == c_id]["title"].head(3).tolist()
        print(f"  Cluster {c_id:02d}: {count:4d} movies | Examples: {', '.join(sample_titles)}")

    print("Clustering training completed successfully!")


if __name__ == "__main__":
    main()
