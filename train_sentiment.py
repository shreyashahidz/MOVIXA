import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from backend.utils.preprocessing import clean_text


def main():
    print("=== Training Sentiment Analysis Model (Supervised Classification) ===")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(base_dir, "data", "raw", "imdb_reviews.csv")
    proc_dir = os.path.join(base_dir, "data", "processed")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print(f"Loading raw reviews from {raw_path}...")
    df = pd.read_csv(raw_path)
    print(f"Original shape: {df.shape}")

    # Ensure no nulls
    df = df.dropna(subset=["review", "sentiment"]).copy()
    
    # Stratified sampling of 20,000 reviews for high accuracy and fast hyperparameter tuning
    sample_size = min(20000, len(df))
    df_sample, _ = train_test_split(
        df,
        train_size=sample_size,
        stratify=df["sentiment"],
        random_state=42
    )
    df_sample = df_sample.copy().reset_index(drop=True)
    print(f"Using stratified sample of {len(df_sample)} reviews for training & tuning.")

    # Clean text
    print("Cleaning review texts (HTML removal, lowercasing, stopword stripping, lemmatization)...")
    df_sample["cleaned_review"] = df_sample["review"].apply(clean_text)

    # Encode labels: positive -> 1, negative -> 0
    df_sample["label"] = df_sample["sentiment"].map({"positive": 1, "negative": 0})

    # Train / Test split: 70% train, 30% test (stratified)
    print("Performing 70/30 Stratified Train-Test split...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df_sample["cleaned_review"],
        df_sample["label"],
        test_size=0.30,
        random_state=42,
        stratify=df_sample["label"]
    )
    print(f"Train size: {len(X_train_raw)} | Test size: {len(X_test_raw)}")

    # TF-IDF Vectorization
    print("Fitting TfidfVectorizer(max_features=5000, ngram_range=(1,2))...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)
    print(f"TF-IDF feature matrix shape: {X_train_tfidf.shape}")

    # Step 3: Model Comparison & Hyperparameter Tuning
    print("\n--- Comparing Models with Hyperparameter Tuning ---")

    # 1. Logistic Regression with GridSearchCV
    print("\nTuning Logistic Regression (C in [0.1, 1.0, 10.0])...")
    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        param_grid={"C": [0.1, 1.0, 10.0]},
        cv=3,
        scoring="f1",
        n_jobs=-1
    )
    lr_grid.fit(X_train_tfidf, y_train)
    best_lr = lr_grid.best_estimator_
    y_pred_lr = best_lr.predict(X_test_tfidf)
    f1_lr = f1_score(y_test, y_pred_lr)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print(f"Best Logistic Regression: C={lr_grid.best_params_['C']} | Test Acc: {acc_lr:.4f} | Test F1: {f1_lr:.4f}")

    # 2. Multinomial Naive Bayes
    print("\nTuning Multinomial Naive Bayes (alpha in [0.5, 1.0, 2.0])...")
    nb_grid = GridSearchCV(
        MultinomialNB(),
        param_grid={"alpha": [0.5, 1.0, 2.0]},
        cv=3,
        scoring="f1",
        n_jobs=-1
    )
    nb_grid.fit(X_train_tfidf, y_train)
    best_nb = nb_grid.best_estimator_
    y_pred_nb = best_nb.predict(X_test_tfidf)
    f1_nb = f1_score(y_test, y_pred_nb)
    acc_nb = accuracy_score(y_test, y_pred_nb)
    print(f"Best Naive Bayes: alpha={nb_grid.best_params_['alpha']} | Test Acc: {acc_nb:.4f} | Test F1: {f1_nb:.4f}")

    # 3. LinearSVC
    print("\nTuning LinearSVC (C in [0.1, 1.0])...")
    svc_grid = GridSearchCV(
        LinearSVC(random_state=42, max_iter=2000),
        param_grid={"C": [0.1, 1.0]},
        cv=3,
        scoring="f1",
        n_jobs=-1
    )
    svc_grid.fit(X_train_tfidf, y_train)
    best_svc = svc_grid.best_estimator_
    y_pred_svc = best_svc.predict(X_test_tfidf)
    f1_svc = f1_score(y_test, y_pred_svc)
    acc_svc = accuracy_score(y_test, y_pred_svc)
    print(f"Best LinearSVC: C={svc_grid.best_params_['C']} | Test Acc: {acc_svc:.4f} | Test F1: {f1_svc:.4f}")

    # Pick champion model (prefer LogisticRegression if close, as it provides native predict_proba)
    models_perf = [
        ("Logistic Regression", best_lr, f1_lr, acc_lr),
        ("Multinomial NB", best_nb, f1_nb, acc_nb),
        ("LinearSVC", best_svc, f1_svc, acc_svc),
    ]
    # Sort primarily by F1 score
    models_perf.sort(key=lambda x: x[2], reverse=True)
    champion_name, champion_model, champion_f1, champion_acc = models_perf[0]
    
    # If LinearSVC won by a negligible margin (< 0.005), select Logistic Regression for smooth probability calibration
    if champion_name == "LinearSVC" and abs(f1_svc - f1_lr) < 0.008:
        champion_name, champion_model, champion_f1, champion_acc = ("Logistic Regression", best_lr, f1_lr, acc_lr)

    print(f"\nChampion Model Selected: {champion_name} (F1: {champion_f1:.4f}, Acc: {champion_acc:.4f})")
    
    # Detailed classification report on champion
    y_pred_champ = champion_model.predict(X_test_tfidf)
    print("\nChampion Classification Report:")
    print(classification_report(y_test, y_pred_champ, target_names=["Negative", "Positive"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_champ))

    # Save artifacts
    print("\nSaving sentiment model and vectorizer...")
    joblib.dump(champion_model, os.path.join(models_dir, "sentiment_model.pkl"))
    joblib.dump(vectorizer, os.path.join(models_dir, "tfidf_vectorizer.pkl"))

    # Save cleaned sample data
    proc_sample_path = os.path.join(proc_dir, "reviews_clean.csv")
    df_sample[["review", "cleaned_review", "sentiment", "label"]].to_csv(proc_sample_path, index=False)
    print(f"Saved processed reviews sample to {proc_sample_path}")

    print("Sentiment training completed successfully!")


if __name__ == "__main__":
    main()
