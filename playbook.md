# Project Playbook: Interactive Movie Recommendation & Sentiment System

## 1. Project Overview

| Field | Detail |
|---|---|
| Title | Interactive Movie Recommendation & Sentiment System |
| Modules | Supervised Sentiment Analysis + Clustering Algorithms |
| Course Mapping | Experiment 10 — Mini Project Implementation for Web-Based ML Applications (AI&ML, TCET) |
| ML Tasks | Classification (sentiment: positive/negative/neutral) + Clustering (movie grouping for recommendations) |
| Stack | Python, Scikit-Learn, Pandas, NumPy, Flask (backend API), Streamlit or minimal HTML/CSS (frontend placeholder) |

**Core idea:** Users search/select a movie → system recommends similar movies using **clustering** (K-Means on movie features/genre-embeddings), and users can submit or browse reviews → a **supervised classifier** predicts sentiment (positive/negative) on review text using TF-IDF + Logistic Regression/SVM.

---

## 2. Datasets

1. **Recommendation module:** [TMDB 5000 Movies dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) or MovieLens `movies.csv` — columns: `title, genres, overview, keywords, cast, director, vote_average, popularity`.
2. **Sentiment module:** IMDB 50K Movie Reviews dataset (`review`, `sentiment` labeled positive/negative) — Kaggle: `lakshmi25npathi/imdb-dataset-of-50k-movie-reviews`.

Place both under `data/raw/`.

---

## 3. Repository Structure

```
movie-rec-sentiment/
├── data/
│   ├── raw/                     # original CSVs
│   └── processed/                # cleaned/feature-engineered data
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_clustering_model.ipynb
│   └── 03_sentiment_model.ipynb
├── models/
│   ├── cluster_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── sentiment_model.pkl
│   └── scaler.pkl
├── backend/
│   ├── app.py                   # Flask entrypoint
│   ├── config.py
│   ├── routes/
│   │   ├── recommend.py
│   │   └── sentiment.py
│   ├── services/
│   │   ├── recommender.py       # loads cluster model, returns similar movies
│   │   └── sentiment_service.py # loads TF-IDF + classifier, predicts sentiment
│   ├── utils/
│   │   └── preprocessing.py     # shared text cleaning fns
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js                # (placeholder; will be redone in Antigravity)
├── train_clustering.py          # standalone script: Step 2+3 for recommender
├── train_sentiment.py           # standalone script: Step 2+3 for sentiment
├── evaluate.py                  # Step 4: metrics + validation tracking
├── README.md
└── requirements.txt
```

---

## 4. ML Pipeline (maps to Experiment's 4 Steps)

### Step 1 — Problem Definition
- Task A (Regression/Clustering): group movies into clusters based on genre, overview embeddings, and popularity → recommend top-N movies from the same cluster as a selected movie.
- Task B (Classification): predict sentiment of a review string as Positive/Negative.

### Step 2 — Data Acquisition & Preprocessing
**Movies data:**
- Drop nulls in `title`, `overview`, `genres`.
- Combine `overview + genres + keywords` into a single `tags` text column.
- Vectorize `tags` using `TfidfVectorizer(max_features=5000, stop_words='english')`.
- Scale numeric features (`vote_average`, `popularity`) with `StandardScaler`.
- Optional: reduce TF-IDF dimensionality with `TruncatedSVD` (PCA doesn't work well on sparse TF-IDF; use `TruncatedSVD(n_components=50)`).

**Reviews data:**
- Lowercase, strip HTML tags, remove punctuation/stopwords, lemmatize (use `nltk` or `spacy`).
- Encode label: positive → 1, negative → 0.
- Train/test split 70/30, `stratify=y`.

### Step 3 — Model Development
**Clustering (recommender):**
```python
from sklearn.cluster import KMeans
km = KMeans(n_clusters=15, random_state=42, n_init=10)
km.fit(reduced_features)   # SVD-reduced TF-IDF + scaled numeric cols
```
- Use Elbow Method + Silhouette Score to pick optimal `k`.
- Save `cluster_model.pkl`, `svd.pkl`, `scaler.pkl`.
- Recommendation logic: given a movie, find its cluster label → return top-N movies in same cluster sorted by `vote_average`/cosine similarity within cluster.

**Sentiment (classification):**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

params = {'C': [0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(max_iter=1000), params, cv=5, scoring='f1')
grid.fit(X_train_tfidf, y_train)
```
- Compare Logistic Regression vs LinearSVC vs Multinomial Naive Bayes; pick best by F1.
- Save `sentiment_model.pkl` + `tfidf_vectorizer.pkl`.

### Step 4 — Evaluation & Validation Tracking
- Clustering: Silhouette Score, Davies-Bouldin Index; log for each `k` tried (2–20) in `evaluate.py`, save a plot `elbow_plot.png`.
- Sentiment: Accuracy, Precision, Recall, F1, Confusion Matrix; log metrics per run to `models/metrics_log.csv` (append mode) so multiple training runs are tracked over time, exactly as the experiment sheet requires ("Track validation metrics across multiple training runs").

---

## 5. Backend (Flask API)

### `requirements.txt`
```
flask
flask-cors
scikit-learn
pandas
numpy
nltk
joblib
```

### API Endpoints

| Method | Route | Purpose | Request Body | Response |
|---|---|---|---|---|
| GET | `/api/movies` | List/search movies (paginated) | `?q=&page=` | `[{id, title, genres, poster_hint}]` |
| GET | `/api/recommend/<movie_id>` | Get cluster-based recommendations | — | `[{id, title, genres, score}]` (top 10) |
| POST | `/api/sentiment` | Predict sentiment for a review | `{ "review": "text..." }` | `{ "sentiment": "positive", "confidence": 0.87 }` |
| GET | `/api/movie/<movie_id>/reviews` | Fetch stored reviews + sentiment tags | — | `[{review, sentiment}]` |
| GET | `/api/health` | Health check | — | `{status: "ok"}` |

### `backend/app.py` (skeleton)
```python
from flask import Flask
from flask_cors import CORS
from routes.recommend import recommend_bp
from routes.sentiment import sentiment_bp

app = Flask(__name__)
CORS(app)
app.register_blueprint(recommend_bp, url_prefix="/api")
app.register_blueprint(sentiment_bp, url_prefix="/api")

@app.get("/api/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(debug=True, port=5000)
```

### `backend/services/recommender.py` (skeleton)
```python
import joblib, pandas as pd

cluster_model = joblib.load("models/cluster_model.pkl")
svd = joblib.load("models/svd.pkl")
movies_df = pd.read_csv("data/processed/movies_clean.csv")

def get_recommendations(movie_id, top_n=10):
    row = movies_df[movies_df.id == movie_id]
    cluster_label = row["cluster"].values[0]
    similar = movies_df[movies_df.cluster == cluster_label]
    similar = similar[similar.id != movie_id].sort_values("vote_average", ascending=False)
    return similar.head(top_n).to_dict(orient="records")
```

### `backend/services/sentiment_service.py` (skeleton)
```python
import joblib
from utils.preprocessing import clean_text

vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
model = joblib.load("models/sentiment_model.pkl")

def predict_sentiment(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec).max() if hasattr(model, "predict_proba") else None
    return {"sentiment": "positive" if pred == 1 else "negative", "confidence": round(float(proba), 2) if proba else None}
```

---

## 6. Frontend (placeholder — clean & minimal, to be refined later)

Keep it a single-page app with three sections. Use plain HTML/CSS/JS so it's lightweight and easy for Antigravity to later rebuild/restyle.

**Sections:**
1. **Search & Browse** — search bar + movie grid (title, genre chips).
2. **Movie Detail + Recommendations** — click a movie → shows overview + "You may also like" grid (from `/api/recommend/<id>`).
3. **Sentiment Checker** — textarea + "Analyze" button → calls `/api/sentiment`, shows badge (green=positive/red=negative) + confidence bar.

**Design direction:** dark cinematic theme, card-based grid layout, subtle hover elevation, one accent color (e.g. amber/gold like IMDB), system font stack, responsive grid (`auto-fill, minmax(180px, 1fr)`). Keep JS vanilla `fetch()` calls to the Flask API — no framework needed at this stage.

---

## 7. Build Order (give this sequence to Antigravity)

1. Set up repo structure above.
2. Write `train_clustering.py` — load movies data → preprocess → TF-IDF+SVD → KMeans → save models + `elbow_plot.png`.
3. Write `train_sentiment.py` — load reviews → clean → TF-IDF → GridSearchCV over LogisticRegression/LinearSVC/NB → save best model.
4. Write `evaluate.py` — compute & log metrics for both models to `models/metrics_log.csv`.
5. Build Flask backend (`app.py`, routes, services) wired to saved models.
6. Build minimal frontend (3 sections above) hitting the Flask API.
7. Test end-to-end: search → recommend → sentiment check.
8. (Optional) Dockerize: `Dockerfile` for backend, serve frontend via Flask static folder or separate simple server.

---

## 8. Deliverables Checklist (matches experiment rubric)

- [ ] Problem definition + ML task identification documented in `README.md`
- [ ] EDA notebook with data cleaning, scaling, encoding, PCA/SVD, clustering
- [ ] Train/test split (70/30) shown in notebooks
- [ ] Trained clustering model + trained classification model saved as `.pkl`
- [ ] Hyperparameter tuning (GridSearchCV) documented with results table
- [ ] Evaluation metrics (Accuracy/Precision/Recall/F1 for sentiment; Silhouette/Davies-Bouldin for clustering) logged across multiple runs
- [ ] Flask backend with working API endpoints
- [ ] Deployed/functional frontend interface
- [ ] Screenshots + final report/README summarizing results
