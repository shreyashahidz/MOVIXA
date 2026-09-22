import os
import json
import time
import urllib.request
import joblib
import numpy as np
import pandas as pd
import sys

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
        except Exception as e:
            time.sleep(0.5)
    return {}

def main():
    print("=== Fetching Enriched Bollywood & Modern Hollywood Dataset from TMDB ===")

    # Load existing clean movies
    existing_df = pd.read_csv(MOVIES_CSV) if os.path.exists(MOVIES_CSV) else pd.DataFrame()
    print(f"Existing catalog records: {len(existing_df)}")

    # Load trained ML models for clustering assignment
    kmeans = joblib.load(os.path.join(MODELS_DIR, "cluster_model.pkl"))
    svd = joblib.load(os.path.join(MODELS_DIR, "svd.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    tfidf = joblib.load(os.path.join(MODELS_DIR, "movie_tfidf.pkl"))

    # Load poster cache
    poster_cache = {}
    if os.path.exists(POSTER_CACHE_JSON):
        try:
            with open(POSTER_CACHE_JSON, "r", encoding="utf-8") as f:
                poster_cache = json.load(f)
        except Exception:
            poster_cache = {}

    seen_ids = set(existing_df["id"].tolist()) if not existing_df.empty else set()
    new_records = []

    # 1. Fetch Bollywood Movies
    print("\nFetching Hindi / Bollywood cinema from TMDB...")
    b_pages = [
        f"https://api.themoviedb.org/3/discover/movie?api_key={API_KEY}&with_original_language=hi&sort_by=popularity.desc&page={p}"
        for p in range(1, 21)
    ] + [
        f"https://api.themoviedb.org/3/discover/movie?api_key={API_KEY}&with_original_language=hi&sort_by=vote_count.desc&page={p}"
        for p in range(1, 11)
    ] + [
        f"https://api.themoviedb.org/3/discover/movie?api_key={API_KEY}&with_original_language=hi&primary_release_date.gte=2023-01-01&primary_release_date.lte=2024-12-31&sort_by=popularity.desc&page={p}"
        for p in range(1, 11)
    ]

    for p_url in b_pages:
        data = tmdb_get(p_url)
        results = data.get("results", [])
        for m in results:
            mid = m.get("id")
            if not mid or mid in seen_ids:
                continue
            title = m.get("title", "").strip()
            overview = m.get("overview", "").strip()
            poster_path = m.get("poster_path")
            if not title or not overview or not poster_path:
                continue

            genres = [GENRE_MAP.get(gid, "Cinema") for gid in m.get("genre_ids", [])]
            genres_clean = ", ".join(genres) if genres else "Drama"
            vote_avg = round(float(m.get("vote_average", 7.0)), 1)
            pop = round(float(m.get("popularity", 50.0)), 2)
            rel_date = m.get("release_date") or "2024-01-01"
            backdrop_path = m.get("backdrop_path") or poster_path

            # Feature extraction for ML cluster
            tag_text = clean_text(f"{genres_clean} {overview}")
            tfidf_v = tfidf.transform([tag_text])
            svd_v = svd.transform(tfidf_v)
            num_v = scaler.transform([[vote_avg, pop]])
            feat_v = np.hstack([svd_v, num_v])
            cluster_id = int(kmeans.predict(feat_v)[0])

            poster_url = f"{TMDB_POSTER_BASE}{poster_path}"
            backdrop_url = f"{TMDB_BACKDROP_BASE}{backdrop_path}"

            rec = {
                "id": mid,
                "title": title,
                "genres_clean": genres_clean,
                "overview": overview,
                "vote_average": vote_avg,
                "vote_count": int(m.get("vote_count", 100)),
                "popularity": pop,
                "release_date": rel_date,
                "cluster": cluster_id,
                "tagline": "",
                "poster_url": poster_url,
                "backdrop_url": backdrop_url,
                "runtime": 145,
                "industry": "bollywood"
            }
            new_records.append(rec)
            seen_ids.add(mid)

            poster_cache[str(mid)] = {
                "poster_url": poster_url,
                "backdrop_url": backdrop_url,
                "runtime": 145
            }

    print(f"Collected {len(new_records)} new Bollywood movies.")

    # 2. Fetch Modern 2023-2024 Hollywood Blockbusters
    print("\nFetching Modern 2023-2024 Hollywood releases from TMDB...")
    h_pages = [
        f"https://api.themoviedb.org/3/discover/movie?api_key={API_KEY}&with_original_language=en&primary_release_date.gte=2023-01-01&primary_release_date.lte=2024-12-31&sort_by=popularity.desc&vote_count.gte=300&page={p}"
        for p in range(1, 6)
    ]
    h_count = 0
    for p_url in h_pages:
        data = tmdb_get(p_url)
        results = data.get("results", [])
        for m in results:
            mid = m.get("id")
            if not mid or mid in seen_ids:
                continue
            title = m.get("title", "").strip()
            overview = m.get("overview", "").strip()
            poster_path = m.get("poster_path")
            if not title or not overview or not poster_path:
                continue

            genres = [GENRE_MAP.get(gid, "Cinema") for gid in m.get("genre_ids", [])]
            genres_clean = ", ".join(genres) if genres else "Action"
            vote_avg = round(float(m.get("vote_average", 7.5)), 1)
            pop = round(float(m.get("popularity", 120.0)), 2)
            rel_date = m.get("release_date") or "2024-01-01"
            backdrop_path = m.get("backdrop_path") or poster_path

            tag_text = clean_text(f"{genres_clean} {overview}")
            tfidf_v = tfidf.transform([tag_text])
            svd_v = svd.transform(tfidf_v)
            num_v = scaler.transform([[vote_avg, pop]])
            feat_v = np.hstack([svd_v, num_v])
            cluster_id = int(kmeans.predict(feat_v)[0])

            poster_url = f"{TMDB_POSTER_BASE}{poster_path}"
            backdrop_url = f"{TMDB_BACKDROP_BASE}{backdrop_path}"

            rec = {
                "id": mid,
                "title": title,
                "genres_clean": genres_clean,
                "overview": overview,
                "vote_average": vote_avg,
                "vote_count": int(m.get("vote_count", 500)),
                "popularity": pop,
                "release_date": rel_date,
                "cluster": cluster_id,
                "tagline": "",
                "poster_url": poster_url,
                "backdrop_url": backdrop_url,
                "runtime": 130,
                "industry": "hollywood"
            }
            new_records.append(rec)
            seen_ids.add(mid)
            h_count += 1

            poster_cache[str(mid)] = {
                "poster_url": poster_url,
                "backdrop_url": backdrop_url,
                "runtime": 130
            }

    print(f"Collected {h_count} modern Hollywood movies.")

    # 3. Specific Iconic Bollywood Movies Guarantee
    print("\nEnsuring all top Bollywood blockbusters are present with real posters...")
    must_have_titles = [
        ("Stree 2", "1112426"),
        ("Fighter", "784651"),
        ("Kalki 2898-AD", "801688"),
        ("Jawan", "872906"),
        ("Animal", "781732"),
        ("12th Fail", "1163258"),
        ("Pathaan", "864692"),
        ("Dunki", "960876"),
        ("Tiger 3", "720557"),
        ("Chandu Champion", "1020951"),
        ("Munjya", "1187058"),
        ("Shaitaan", "1187619"),
        ("Article 370", "1233531"),
        ("KILL", "1160018"),
        ("Crew", "1045931"),
        ("Bade Miyan Chote Miyan", "936622"),
        ("3 Idiots", "20453"),
        ("Dangal", "360814"),
        ("Dilwale Dulhania Le Jayenge", "19404"),
        ("Sholay", "12259"),
        ("RRR", "579974"),
        ("K.G.F: Chapter 2", "587412"),
        ("Drishyam 2", "1029827"),
        ("Tumbbad", "538858"),
        ("Andhadhun", "534780"),
        ("Bajrangi Bhaijaan", "348892"),
        ("Zindagi Na Milegi Dobara", "61202")
    ]

    for title, mid_str in must_have_titles:
        mid = int(mid_str)
        m_info = tmdb_get(f"https://api.themoviedb.org/3/movie/{mid}?api_key={API_KEY}")
        if m_info and m_info.get("title"):
            p_path = m_info.get("poster_path")
            b_path = m_info.get("backdrop_path") or p_path
            p_url = f"{TMDB_POSTER_BASE}{p_path}" if p_path else None
            b_url = f"{TMDB_BACKDROP_BASE}{b_path}" if b_path else None
            tag = m_info.get("tagline", "")
            overview = m_info.get("overview") or f"A blockbuster cinematic journey of {title}."
            g_names = [g["name"] for g in m_info.get("genres", [])]
            genres_clean = ", ".join(g_names) if g_names else "Action, Drama"
            vote_avg = round(float(m_info.get("vote_average", 8.0)), 1)
            pop = round(float(m_info.get("popularity", 180.0)), 2)
            rel_date = m_info.get("release_date") or "2024-01-01"
            runtime = int(m_info.get("runtime") or 150)

            tag_text = clean_text(f"{genres_clean} {overview}")
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

            # If already in existing_df or new_records, update it; otherwise add it
            rec = {
                "id": mid,
                "title": m_info.get("title", title),
                "genres_clean": genres_clean,
                "overview": overview,
                "vote_average": vote_avg,
                "vote_count": int(m_info.get("vote_count", 1500)),
                "popularity": pop,
                "release_date": rel_date,
                "cluster": cluster_id,
                "tagline": tag,
                "poster_url": p_url,
                "backdrop_url": b_url,
                "runtime": runtime,
                "industry": "bollywood"
            }

            # Remove previous version if any
            if not existing_df.empty:
                existing_df = existing_df[existing_df["id"] != mid]
            new_records = [r for r in new_records if r["id"] != mid]
            new_records.append(rec)

    # Combine and save
    if new_records:
        new_df = pd.DataFrame(new_records)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = existing_df

    # Deduplicate by ID
    combined_df = combined_df.drop_duplicates(subset=["id"], keep="last")

    # Fix any missing industry labels
    if "industry" not in combined_df.columns:
        combined_df["industry"] = "hollywood"
    combined_df["industry"] = combined_df["industry"].fillna("hollywood")

    # Ensure all Bollywood rows are properly tagged
    b_indices = combined_df[combined_df["id"].isin([r["id"] for r in new_records if r["industry"] == "bollywood"])].index
    combined_df.loc[b_indices, "industry"] = "bollywood"

    combined_df.to_csv(MOVIES_CSV, index=False)
    with open(POSTER_CACHE_JSON, "w", encoding="utf-8") as f:
        json.dump(poster_cache, f, indent=2)

    total_bollywood = len(combined_df[combined_df["industry"] == "bollywood"])
    total_hollywood = len(combined_df[combined_df["industry"] != "bollywood"])

    print(f"\nSuccessfully enriched dataset!")
    print(f"Total Movies: {len(combined_df)}")
    print(f"Bollywood Movies: {total_bollywood}")
    print(f"Hollywood Movies: {total_hollywood}")
    print(f"Cached posters: {len(poster_cache)}")

if __name__ == "__main__":
    main()
