import os
import urllib.request
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

TMDB_URL = "https://raw.githubusercontent.com/vamshi121/TMDB-5000-Movie-Dataset/master/tmdb_5000_movies.csv"
IMDB_URL = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"

TMDB_DEST = os.path.join(RAW_DIR, "tmdb_5000_movies.csv")
IMDB_DEST = os.path.join(RAW_DIR, "imdb_reviews.csv")


def download_file(url, destination, name):
    if os.path.exists(destination) and os.path.getsize(destination) > 1000:
        print(f"[{name}] already exists at {destination} ({os.path.getsize(destination):,} bytes). Skipping download.")
        return True

    print(f"[{name}] Downloading from {url}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=60) as response, open(destination, "wb") as out_file:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB chunks
            
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    print(f"[{name}] Progress: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({pct:.1f}%)", end="\r")
                else:
                    print(f"[{name}] Downloaded: {downloaded / (1024*1024):.1f} MB", end="\r")
                    
        print(f"\n[{name}] Successfully downloaded to {destination} ({os.path.getsize(destination):,} bytes)")
        return True
    except Exception as e:
        print(f"\n[{name}] Download failed: {e}")
        if os.path.exists(destination):
            os.remove(destination)
        return False


def verify_datasets():
    print("\n--- Verifying Datasets ---")
    if os.path.exists(TMDB_DEST):
        df_tmdb = pd.read_csv(TMDB_DEST, nrows=5)
        print(f"TMDB Dataset: Found {os.path.getsize(TMDB_DEST):,} bytes. Columns: {list(df_tmdb.columns)}")
    else:
        print("TMDB Dataset: Missing!")

    if os.path.exists(IMDB_DEST):
        df_imdb = pd.read_csv(IMDB_DEST, nrows=5)
        print(f"IMDB Dataset: Found {os.path.getsize(IMDB_DEST):,} bytes. Columns: {list(df_imdb.columns)}")
    else:
        print("IMDB Dataset: Missing!")


if __name__ == "__main__":
    print("Starting dataset acquisition...")
    s1 = download_file(TMDB_URL, TMDB_DEST, "TMDB 5000 Movies")
    s2 = download_file(IMDB_URL, IMDB_DEST, "IMDB Reviews")
    verify_datasets()
