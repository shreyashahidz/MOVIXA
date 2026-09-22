import os
import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from backend.utils.preprocessing import clean_text


def evaluate_clustering(models_dir, data_dir):
    print("\n--- Evaluating Clustering Recommender Model ---")
    model_path = os.path.join(models_dir, "cluster_model.pkl")
    features_path = os.path.join(models_dir, "combined_features.npy")
    movies_path = os.path.join(data_dir, "processed", "movies_clean.csv")

    if not (os.path.exists(model_path) and os.path.exists(features_path)):
        print("Clustering artifacts missing! Run train_clustering.py first.")
        return None

    kmeans = joblib.load(model_path)
    X = np.load(features_path)
    df = pd.read_csv(movies_path)
    labels = kmeans.labels_

    # Compute metrics (subsample for silhouette for high evaluation speed)
    sil_score = silhouette_score(X, labels, sample_size=min(2000, len(X)), random_state=42)
    db_score = davies_bouldin_score(X, labels)
    inertia = kmeans.inertia_
    n_clusters = kmeans.n_clusters

    print(f"Clusters: {n_clusters}")
    print(f"Inertia: {inertia:,.2f}")
    print(f"Silhouette Score: {sil_score:.4f} (Higher is better, range [-1, 1])")
    print(f"Davies-Bouldin Index: {db_score:.4f} (Lower is better, minimum 0)")

    return {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "module": "Clustering_Recommender",
        "algorithm": f"KMeans(k={n_clusters})",
        "metric_1_name": "Silhouette_Score",
        "metric_1_val": round(float(sil_score), 4),
        "metric_2_name": "Davies_Bouldin",
        "metric_2_val": round(float(db_score), 4),
        "metric_3_name": "Inertia",
        "metric_3_val": round(float(inertia), 2),
        "metric_4_name": "Total_Items",
        "metric_4_val": len(df)
    }


def evaluate_sentiment(models_dir, data_dir):
    print("\n--- Evaluating Supervised Sentiment Model ---")
    model_path = os.path.join(models_dir, "sentiment_model.pkl")
    vec_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    sample_path = os.path.join(data_dir, "processed", "reviews_clean.csv")

    if not (os.path.exists(model_path) and os.path.exists(vec_path) and os.path.exists(sample_path)):
        print("Sentiment artifacts missing! Run train_sentiment.py first.")
        return None

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    df = pd.read_csv(sample_path)

    X_tfidf = vectorizer.transform(df["cleaned_review"].fillna(""))
    y_true = df["label"].values
    y_pred = model.predict(X_tfidf)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Evaluated on {len(df)} test/validation records:")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    return {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "module": "Sentiment_Classifier",
        "algorithm": type(model).__name__,
        "metric_1_name": "Accuracy",
        "metric_1_val": round(float(acc), 4),
        "metric_2_name": "Precision",
        "metric_2_val": round(float(prec), 4),
        "metric_3_name": "Recall",
        "metric_3_val": round(float(rec), 4),
        "metric_4_name": "F1_Score",
        "metric_4_val": round(float(f1), 4)
    }


def main():
    print("=== Running Experiment 10 Evaluation & Metric Tracker ===")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    data_dir = os.path.join(base_dir, "data")
    metrics_log_path = os.path.join(models_dir, "metrics_log.csv")

    records = []
    clust_metrics = evaluate_clustering(models_dir, data_dir)
    if clust_metrics:
        records.append(clust_metrics)

    sent_metrics = evaluate_sentiment(models_dir, data_dir)
    if sent_metrics:
        records.append(sent_metrics)

    if records:
        df_new = pd.DataFrame(records)
        if os.path.exists(metrics_log_path):
            df_existing = pd.read_csv(metrics_log_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new

        df_combined.to_csv(metrics_log_path, index=False)
        print(f"\nSuccessfully logged evaluation metrics to {metrics_log_path}")
        print("\n--- Current Metrics Log ---")
        print(df_combined.to_string(index=False))


if __name__ == "__main__":
    main()
