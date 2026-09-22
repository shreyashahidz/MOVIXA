import os
import sys
import json
import time
import urllib.request
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.utils.preprocessing import clean_text

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MOVIES_CSV = os.path.join(PROCESSED_DIR, "movies_clean.csv")
POSTER_CACHE_JSON = os.path.join(PROCESSED_DIR, "poster_cache.json")

API_KEY = "8265bd1679663a7ea12ac168da84d2e8"
TMDB_POSTER_BASE = "https://image.tmdb.org/t/p/w500"
TMDB_BACKDROP_BASE = "https://image.tmdb.org/t/p/original"

GENRE_MAP = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime",
    99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History",
    27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance",
    878: "Science Fiction", 10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western"
}

def tmdb_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            time.sleep(0.5)
    return {}

def main():
    print("=== Fetching Comprehensive Bollywood Horror Collection from TMDB ===")

    df = pd.read_csv(MOVIES_CSV) if os.path.exists(MOVIES_CSV) else pd.DataFrame()
    kmeans = joblib.load(os.path.join(MODELS_DIR, "cluster_model.pkl"))
    svd = joblib.load(os.path.join(MODELS_DIR, "svd.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    tfidf = joblib.load(os.path.join(MODELS_DIR, "movie_tfidf.pkl"))

    poster_cache = {}
    if os.path.exists(POSTER_CACHE_JSON):
        with open(POSTER_CACHE_JSON, "r", encoding="utf-8") as f:
            poster_cache = json.load(f)

    # 1. Fetch pages of Hindi horror sorted by vote_count and popularity
    horror_urls = [
        f"https://api.themoviedb.org/3/discover/movie?api_key={API_KEY}&with_original_language=hi&with_genres=27&sort_by=vote_count.desc&page={p}"
        for p in range(1, 6)
    ] + [
        f"https://api.themoviedb.org/3/discover/movie?api_key={API_KEY}&with_original_language=hi&with_genres=27&sort_by=popularity.desc&page={p}"
        for p in range(1, 6)
    ]

    new_horror = []
    seen_ids = set()

    for url in horror_urls:
        data = tmdb_get(url)
        for m in data.get("results", []):
            mid = m.get("id")
            if not mid or mid in seen_ids:
                continue
            seen_ids.add(mid)

            title = m.get("title", "").strip()
            overview = m.get("overview", "").strip()
            poster_path = m.get("poster_path")
            if not title or not poster_path:
                continue

            # Skip inappropriate titles
            if any(bad in title.lower() for bad in ["lust", "porn", "erotic", "sex"]):
                continue

            g_names = [GENRE_MAP.get(gid, "Horror") for gid in m.get("genre_ids", [])]
            if "Horror" not in g_names:
                g_names.insert(0, "Horror")
            genres_clean = ", ".join(g_names)

            vote_avg = round(float(m.get("vote_average", 6.5)), 1)
            pop = round(float(m.get("popularity", 50.0)), 2)
            rel_date = m.get("release_date") or "2020-01-01"
            backdrop_path = m.get("backdrop_path") or poster_path

            tag_text = clean_text(f"{genres_clean} {overview}")
            tfidf_v = tfidf.transform([tag_text])
            svd_v = svd.transform(tfidf_v)
            num_v = scaler.transform([[vote_avg, pop]])
            feat_v = np.hstack([svd_v, num_v])
            cluster_id = int(kmeans.predict(feat_v)[0])

            p_url = f"{TMDB_POSTER_BASE}{poster_path}"
            b_url = f"{TMDB_BACKDROP_BASE}{backdrop_path}"

            poster_cache[str(mid)] = {
                "poster_url": p_url,
                "backdrop_url": b_url,
                "runtime": 135
            }

            rec = {
                "id": mid,
                "title": title,
                "genres_clean": genres_clean,
                "overview": overview or f"A chilling Bollywood horror experience in {title}.",
                "vote_average": vote_avg,
                "vote_count": int(m.get("vote_count", 500)),
                "popularity": pop,
                "release_date": rel_date,
                "cluster": cluster_id,
                "tagline": "Fear has a new address.",
                "poster_url": p_url,
                "backdrop_url": b_url,
                "runtime": 135,
                "industry": "bollywood"
            }
            new_horror.append(rec)

    print(f"Discovered {len(new_horror)} Bollywood horror films.")

    # Explicit top famous horror blockbusters
    top_famous = [
        (1112426, "Stree 2", "Comedy, Horror", 200.0, "O Stree Raksha Karna."),
        (1187619, "Shaitaan", "Horror, Thriller", 190.0, "Do not let evil cross your threshold."),
        (695962, "Bhool Bhulaiyaa 2", "Comedy, Horror", 185.0, "Ami Je Tomar returns."),
        (980599, "Bhool Bhulaiyaa 3", "Comedy, Horror", 182.0, "The spirit unleashes."),
        (1187058, "Munjya", "Comedy, Horror, Fantasy", 180.0, "Fear has a new mischievous name."),
        (538858, "Tumbbad", "Fantasy, Horror, Thriller", 175.0, "Fear the greed that awakens Hastar."),
        (19025, "Bhool Bhulaiyaa", "Comedy, Horror, Mystery", 170.0, "Manjulika is back."),
        (799177, "Bhediya", "Action, Comedy, Horror", 165.0, "The jungle awakens inside."),
        (533991, "Stree", "Comedy, Horror", 160.0, "O Stree Kal Aana."),
        (20731, "Raaz", "Horror, Romance, Mystery", 155.0, "Some secrets should never be revealed."),
        (19951, "1920", "Horror, Mystery, Romance", 150.0, "A house possessed by terror."),
        (714338, "Bulbbul", "Horror, Drama, Mystery", 145.0, "A dark fairytale from Bengal."),
        (616880, "Bhoot: Part One - The Haunted Ship", "Horror, Thriller", 140.0, "Sea of terror."),
        (186747, "Ek Thi Daayan", "Horror, Thriller", 135.0, "The witch next door."),
        (711643, "Chhorii", "Horror, Drama, Thriller", 130.0, "The fields hide a dark secret."),
        (1229078, "Kakuda", "Comedy, Horror", 125.0, "Open the door before Tuesday 7:15 PM.")
    ]

    for mid, title, genres, pop, tagline in top_famous:
        m_info = tmdb_get(f"https://api.themoviedb.org/3/movie/{mid}?api_key={API_KEY}")
        if m_info and m_info.get("title"):
            p_path = m_info.get("poster_path")
            b_path = m_info.get("backdrop_path") or p_path
            p_url = f"{TMDB_POSTER_BASE}{p_path}" if p_path else None
            b_url = f"{TMDB_BACKDROP_BASE}{b_path}" if b_path else None

            overview = m_info.get("overview") or f"{title} — an iconic Bollywood horror film."
            vote_avg = round(float(m_info.get("vote_average", 7.5)), 1)
            rel_date = m_info.get("release_date") or "2024-01-01"
            runtime = int(m_info.get("runtime") or 140)

            tag_text = clean_text(f"{genres} {overview}")
            tfidf_v = tfidf.transform([tag_text])
            svd_v = svd.transform(tfidf_v)
            num_v = scaler.transform([[vote_avg, pop]])
            feat_v = np.hstack([svd_v, num_v])
            cluster_id = int(kmeans.predict(feat_v)[0])

            if p_url:
                poster_cache[str(mid)] = {
                    "poster_url": p_url,
                    "backdrop_url": b_url,
                    "runtime": runtime
                }

            rec = {
                "id": mid,
                "title": m_info.get("title", title),
                "genres_clean": genres,
                "overview": overview,
                "vote_average": vote_avg,
                "vote_count": int(m_info.get("vote_count", 1200)),
                "popularity": pop,
                "release_date": rel_date,
                "cluster": cluster_id,
                "tagline": tagline,
                "poster_url": p_url,
                "backdrop_url": b_url,
                "runtime": runtime,
                "industry": "bollywood"
            }
            # Remove any existing copy
            new_horror = [r for r in new_horror if r["id"] != mid]
            df = df[df["id"] != mid]
            new_horror.append(rec)

    # Merge into dataframe
    if new_horror:
        horror_df = pd.DataFrame(new_horror)
        df = pd.concat([df, horror_df], ignore_index=True)

    df = df.drop_duplicates(subset=["id"], keep="last")
    df.to_csv(MOVIES_CSV, index=False)

    with open(POSTER_CACHE_JSON, "w", encoding="utf-8") as f:
        json.dump(poster_cache, f, indent=2)

    b_horror = df[(df["industry"] == "bollywood") & df["genres_clean"].str.lower().str.contains("horror", na=False)]
    print(f"Total Bollywood Horror movies now in catalog: {len(b_horror)}")
    print("Top Horror Titles:", b_horror.sort_values(by="popularity", ascending=False)["title"].head(10).tolist())

if __name__ == "__main__":
    main()
