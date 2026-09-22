import os
import json
import urllib.request
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOVIES_PATH = os.path.join(BASE_DIR, "data", "processed", "movies_clean.csv")
POSTER_CACHE_PATH = os.path.join(BASE_DIR, "data", "processed", "poster_cache.json")

API_KEY = "8265bd1679663a7ea12ac168da84d2e8"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
TMDB_BACKDROP_BASE = "https://image.tmdb.org/t/p/original"

def load_cache():
    if os.path.exists(POSTER_CACHE_PATH):
        try:
            with open(POSTER_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(cache):
    with open(POSTER_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f)

def fetch_movie_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            p_path = data.get("poster_path")
            b_path = data.get("backdrop_path")
            runtime = data.get("runtime", 120)
            return movie_id, {
                "poster_url": f"{TMDB_IMAGE_BASE}{p_path}" if p_path else None,
                "backdrop_url": f"{TMDB_BACKDROP_BASE}{b_path}" if b_path else None,
                "runtime": runtime
            }
    except Exception:
        return movie_id, None

def main():
    print("=== Fetching Real Movie Posters from TMDB ===")
    df = pd.read_csv(MOVIES_PATH)
    cache = load_cache()
    print(f"Existing cache entries: {len(cache)}")

    # Prioritize:
    # 1. Top trending by popularity
    # 2. Latest releases by release_date
    # 3. Top rated by vote_average
    # 4. First 600 movies in catalog
    popular_ids = df.sort_values(by="popularity", ascending=False).head(300)["id"].tolist()
    latest_ids = df.sort_values(by="release_date", ascending=False).head(200)["id"].tolist()
    top_rated_ids = df[df["vote_count"] > 1000].sort_values(by="vote_average", ascending=False).head(200)["id"].tolist()
    all_first_ids = df.head(400)["id"].tolist()

    target_ids = list(dict.fromkeys(popular_ids + latest_ids + top_rated_ids + all_first_ids))
    missing_ids = [mid for mid in target_ids if str(mid) not in cache]
    print(f"Target movies: {len(target_ids)} | Missing from cache: {len(missing_ids)}")

    if missing_ids:
        print("Fetching missing poster URLs using 25 worker threads...")
        with ThreadPoolExecutor(max_workers=25) as executor:
            futures = {executor.submit(fetch_movie_poster, mid): mid for mid in missing_ids}
            done = 0
            for future in as_completed(futures):
                mid, result = future.result()
                if result:
                    cache[str(mid)] = result
                done += 1
                if done % 50 == 0 or done == len(missing_ids):
                    print(f"  Fetched {done}/{len(missing_ids)} posters...", end="\r")

        print(f"\nCompleted! Total cached posters: {len(cache)}")
        save_cache(cache)

    # Update movies_clean.csv with poster_url and backdrop_url
    print("Updating movies_clean.csv with poster URLs...")
    df["poster_url"] = df["id"].apply(lambda mid: cache.get(str(mid), {}).get("poster_url") if str(mid) in cache else None)
    df["backdrop_url"] = df["id"].apply(lambda mid: cache.get(str(mid), {}).get("backdrop_url") if str(mid) in cache else None)
    df["runtime"] = df["id"].apply(lambda mid: cache.get(str(mid), {}).get("runtime", 120) if str(mid) in cache else 120)
    df.to_csv(MOVIES_PATH, index=False)
    print(f"Successfully updated {MOVIES_PATH} with {df['poster_url'].notna().sum()} poster URLs!")

if __name__ == "__main__":
    main()
