import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from backend.config import (
    MOVIES_CLEAN_PATH, CLUSTER_MODEL_PATH, SVD_PATH, SCALER_PATH, COMBINED_FEATURES_PATH,
    DATA_DIR
)

POSTER_CACHE_PATH = os.path.join(DATA_DIR, "processed", "poster_cache.json")


class RecommenderService:
    def __init__(self):
        self.movies_df = None
        self.cluster_model = None
        self.svd = None
        self.scaler = None
        self.combined_features = None
        self.poster_cache = {}
        self._load_resources()

    def _load_resources(self):
        if os.path.exists(POSTER_CACHE_PATH):
            try:
                with open(POSTER_CACHE_PATH, "r", encoding="utf-8") as f:
                    self.poster_cache = json.load(f)
            except Exception:
                self.poster_cache = {}

        if os.path.exists(MOVIES_CLEAN_PATH):
            self.movies_df = pd.read_csv(MOVIES_CLEAN_PATH)
            self.movies_df["id"] = self.movies_df["id"].astype(int)
            self.movies_df["vote_average"] = self.movies_df["vote_average"].fillna(0.0).astype(float)
            self.movies_df["vote_count"] = self.movies_df["vote_count"].fillna(0).astype(int)
            self.movies_df["popularity"] = self.movies_df["popularity"].fillna(0.0).astype(float)
            self.movies_df["overview"] = self.movies_df["overview"].fillna("No overview available.")
            self.movies_df["tagline"] = self.movies_df["tagline"].fillna("")
            self.movies_df["release_date"] = self.movies_df["release_date"].fillna("2024-01-01")
            self.movies_df["cluster"] = self.movies_df["cluster"].fillna(0).astype(int)
            if "runtime" not in self.movies_df.columns:
                self.movies_df["runtime"] = 135
            else:
                self.movies_df["runtime"] = self.movies_df["runtime"].fillna(135).astype(int)
            if "industry" not in self.movies_df.columns:
                self.movies_df["industry"] = "hollywood"
            # Exclude adult / inappropriate titles (classroom safe)
            inapp_mask = self.movies_df["title"].str.lower().str.contains("lust|porn|erotic|tante siska", na=False)
            self.movies_df = self.movies_df[~inapp_mask].copy()
        else:
            self.movies_df = pd.DataFrame()

        if os.path.exists(CLUSTER_MODEL_PATH):
            self.cluster_model = joblib.load(CLUSTER_MODEL_PATH)

        if os.path.exists(SVD_PATH):
            self.svd = joblib.load(SVD_PATH)

        if os.path.exists(SCALER_PATH):
            self.scaler = joblib.load(SCALER_PATH)

        if os.path.exists(COMBINED_FEATURES_PATH):
            self.combined_features = np.load(COMBINED_FEATURES_PATH)

    def _ensure_poster(self, movie: dict) -> dict:
        mid = str(movie.get("id"))
        p_url = movie.get("poster_url")
        b_url = movie.get("backdrop_url")

        if p_url and isinstance(p_url, str) and p_url.startswith("http"):
            return movie

        if mid in self.poster_cache:
            entry = self.poster_cache[mid]
            if entry.get("poster_url"):
                movie["poster_url"] = entry["poster_url"]
            if entry.get("backdrop_url"):
                movie["backdrop_url"] = entry["backdrop_url"]
            if entry.get("runtime"):
                movie["runtime"] = entry["runtime"]
            return movie

        bg_colors = [
            "#1e293b", "#0f172a", "#1e1b4b", "#172554", "#14532d", 
            "#701a75", "#3b0764", "#450a0a", "#422006", "#27272a"
        ]
        color = bg_colors[int(movie.get("id", 0)) % len(bg_colors)]
        movie["poster_hint"] = {
            "title": movie.get("title", "Movie"),
            "year": str(movie.get("release_date", ""))[:4] if movie.get("release_date") else "2024",
            "bg_color": color,
            "rating": round(float(movie.get("vote_average", 7.5)), 1)
        }
        return movie

    def filter_by_industry(self, df: pd.DataFrame, industry: str):
        if not industry or industry.lower() == "all":
            return df
        ind_lower = industry.strip().lower()
        if ind_lower == "bollywood":
            return df[df["industry"] == "bollywood"]
        elif ind_lower == "hollywood":
            return df[df["industry"] != "bollywood"]
        return df

    def get_featured_carousel(self, limit: int = 8, industry: str = "all"):
        if self.movies_df.empty:
            return []

        ind_lower = (industry or "all").strip().lower()
        if ind_lower == "bollywood":
            b_df = self.movies_df[
                (self.movies_df["industry"] == "bollywood") & 
                self.movies_df["poster_url"].notna() &
                self.movies_df["backdrop_url"].notna()
            ].sort_values(by="popularity", ascending=False).head(limit)
            movies = b_df.to_dict(orient="records")
        elif ind_lower == "hollywood":
            h_df = self.movies_df[
                (self.movies_df["industry"] != "bollywood") & 
                self.movies_df["poster_url"].notna() &
                self.movies_df["backdrop_url"].notna()
            ].sort_values(by="popularity", ascending=False).head(limit)
            movies = h_df.to_dict(orient="records")
        else:
            # "all": Blend top 4 Bollywood and top 4 Hollywood blockbusters so ALL has a distinct mix!
            half = limit // 2
            b_sample = self.movies_df[
                (self.movies_df["industry"] == "bollywood") & 
                self.movies_df["poster_url"].notna() &
                self.movies_df["backdrop_url"].notna()
            ].sort_values(by="popularity", ascending=False).head(half).to_dict(orient="records")

            h_sample = self.movies_df[
                (self.movies_df["industry"] != "bollywood") & 
                self.movies_df["poster_url"].notna() &
                self.movies_df["backdrop_url"].notna()
            ].sort_values(by="popularity", ascending=False).head(limit - len(b_sample)).to_dict(orient="records")

            # Interleave them: H1, B1, H2, B2...
            interleaved = []
            max_len = max(len(h_sample), len(b_sample))
            for i in range(max_len):
                if i < len(h_sample):
                    interleaved.append(h_sample[i])
                if i < len(b_sample):
                    interleaved.append(b_sample[i])
            movies = interleaved[:limit]

        for m in movies:
            self._ensure_poster(m)
        return movies

    def get_latest_releases(self, limit: int = 15, industry: str = "all"):
        if self.movies_df.empty:
            return []

        ind_lower = (industry or "all").strip().lower()
        df = self.filter_by_industry(self.movies_df, industry)

        valid = df[
            (df["release_date"] != "N/A") & 
            df["poster_url"].notna()
        ].copy()

        if ind_lower == "all":
            b_latest = valid[valid["industry"] == "bollywood"].sort_values(by="release_date", ascending=False).head(limit // 2).to_dict(orient="records")
            h_latest = valid[valid["industry"] != "bollywood"].sort_values(by="release_date", ascending=False).head(limit - len(b_latest)).to_dict(orient="records")
            
            interleaved = []
            for i in range(max(len(b_latest), len(h_latest))):
                if i < len(b_latest):
                    interleaved.append(b_latest[i])
                if i < len(h_latest):
                    interleaved.append(h_latest[i])
            movies = interleaved[:limit]
        else:
            sorted_df = valid.sort_values(by="release_date", ascending=False).head(limit)
            movies = sorted_df.to_dict(orient="records")

        for m in movies:
            self._ensure_poster(m)
        return movies

    def get_top_trending(self, limit: int = 15, industry: str = "all"):
        if self.movies_df.empty:
            return []

        ind_lower = (industry or "all").strip().lower()
        df = self.filter_by_industry(self.movies_df, industry)
        valid = df[df["poster_url"].notna()].copy()

        if ind_lower == "all":
            b_top = valid[valid["industry"] == "bollywood"].sort_values(by="popularity", ascending=False).head(limit // 2).to_dict(orient="records")
            h_top = valid[valid["industry"] != "bollywood"].sort_values(by="popularity", ascending=False).head(limit - len(b_top)).to_dict(orient="records")

            interleaved = []
            for i in range(max(len(h_top), len(b_top))):
                if i < len(h_top):
                    interleaved.append(h_top[i])
                if i < len(b_top):
                    interleaved.append(b_top[i])
            movies = interleaved[:limit]
        else:
            trending = valid.sort_values(by="popularity", ascending=False).head(limit)
            movies = trending.to_dict(orient="records")

        for idx, m in enumerate(movies):
            m["rank"] = idx + 1
            self._ensure_poster(m)
        return movies

    def get_movies_by_ids(self, ids: list):
        if self.movies_df.empty or not ids:
            return []

        clean_ids = [int(i) for i in ids if str(i).isdigit()]
        matched = self.movies_df[self.movies_df["id"].isin(clean_ids)].copy()
        movies = matched.to_dict(orient="records")
        for m in movies:
            self._ensure_poster(m)
        return movies

    def get_movies(self, query: str = "", genre: str = "", industry: str = "all", page: int = 1, limit: int = 20):
        if self.movies_df.empty:
            return {"total": 0, "page": page, "limit": limit, "movies": []}

        df = self.filter_by_industry(self.movies_df, industry)
        if query:
            q_lower = query.strip().lower()
            df = df[df["title"].str.lower().str.contains(q_lower, na=False)]

        if genre and genre.lower() != "all":
            g_lower = genre.strip().lower()
            df = df[df["genres_clean"].str.lower().str.contains(g_lower, na=False)]

        if "popularity" in df.columns:
            df = df.sort_values(by=["popularity", "vote_average"], ascending=[False, False])

        total = len(df)
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paged_df = df.iloc[start_idx:end_idx].copy()

        movies = paged_df.to_dict(orient="records")
        for m in movies:
            self._ensure_poster(m)

        return {
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": int(np.ceil(total / limit)) if limit > 0 else 1,
            "movies": movies
        }

    def get_movie_by_id(self, movie_id):
        if self.movies_df.empty:
            return None
        
        matches = pd.DataFrame()
        try:
            mid_int = int(movie_id)
            matches = self.movies_df[self.movies_df["id"] == mid_int]
        except ValueError:
            pass

        if matches.empty:
            # Fallback search by title
            m_str = str(movie_id).strip().lower()
            matches = self.movies_df[self.movies_df["title"].str.lower() == m_str]

        if matches.empty:
            return None

        movie = matches.iloc[0].to_dict()
        self._ensure_poster(movie)
        return movie

    def get_recommendations(self, movie_id, top_n: int = 10):
        if self.movies_df.empty:
            return []

        movie = self.get_movie_by_id(movie_id)
        if not movie:
            return []

        mid = int(movie["id"])
        target_cluster = int(movie.get("cluster", 0))
        target_genres = [g.strip().lower() for g in str(movie.get("genres_clean", "")).split(",") if g.strip()]

        # Try cluster + SVD similarity if vector is within precomputed matrix
        matches = self.movies_df[self.movies_df["id"] == mid]
        if not matches.empty:
            target_idx = matches.index[0]
            if self.combined_features is not None and target_idx < len(self.combined_features):
                cluster_mask = (self.movies_df["cluster"] == target_cluster) & (self.movies_df["id"] != mid)
                cluster_indices = [i for i in self.movies_df[cluster_mask].index if i < len(self.combined_features)]

                if len(cluster_indices) > 0:
                    target_vector = self.combined_features[target_idx].reshape(1, -1)
                    candidate_vectors = self.combined_features[cluster_indices]
                    sim_scores = cosine_similarity(target_vector, candidate_vectors)[0]

                    candidate_df = self.movies_df.loc[cluster_indices].copy()
                    candidate_df["similarity_score"] = np.round(sim_scores, 4)
                    candidate_df["blend_score"] = (candidate_df["similarity_score"] * 0.70) + ((candidate_df["vote_average"] / 10.0) * 0.30)
                    top_candidates = candidate_df.sort_values(by=["blend_score", "vote_average"], ascending=[False, False]).head(top_n)
                    results = top_candidates.to_dict(orient="records")
                    for m in results:
                        self._ensure_poster(m)

                    if len(results) >= top_n:
                        return results

                    # If cluster has fewer than top_n, supplement with closest genre & thematic matches
                    existing_ids = set([m["id"] for m in results] + [mid])
                    supplements = self.movies_df[~self.movies_df["id"].isin(existing_ids)].copy()
                    def calc_supp_score(row):
                        row_genres = [g.strip().lower() for g in str(row.get("genres_clean", "")).split(",") if g.strip()]
                        overlap = len(set(target_genres).intersection(set(row_genres)))
                        score = (overlap * 0.5) + ((row.get("vote_average", 7.0) / 10.0) * 0.3) + ((row.get("popularity", 50.0) / 300.0) * 0.2)
                        return score

                    supplements["similarity_score"] = supplements.apply(calc_supp_score, axis=1)
                    needed = top_n - len(results)
                    supp_candidates = supplements.sort_values(by=["similarity_score", "vote_average"], ascending=[False, False]).head(needed)
                    supp_results = supp_candidates.to_dict(orient="records")
                    for sm in supp_results:
                        self._ensure_poster(sm)
                    results.extend(supp_results)
                    return results

        # Fallback: Genre overlap + Cluster matching (guarantees recommendations always work!)
        candidates = self.movies_df[self.movies_df["id"] != mid].copy()
        def calc_overlap(row):
            row_genres = [g.strip().lower() for g in str(row.get("genres_clean", "")).split(",") if g.strip()]
            overlap = len(set(target_genres).intersection(set(row_genres)))
            same_cluster = 1 if row.get("cluster") == target_cluster else 0
            same_industry = 1 if row.get("industry") == movie.get("industry") else 0
            score = (overlap * 0.4) + (same_cluster * 0.3) + (same_industry * 0.2) + ((row.get("vote_average", 7.0) / 10.0) * 0.1)
            return score

        candidates["similarity_score"] = candidates.apply(calc_overlap, axis=1)
        top_candidates = candidates.sort_values(by=["similarity_score", "vote_average"], ascending=[False, False]).head(top_n)
        results = top_candidates.to_dict(orient="records")
        for m in results:
            self._ensure_poster(m)
        return results

    def get_genres(self):
        if self.movies_df.empty:
            return []
        all_genres = set()
        for g_str in self.movies_df["genres_clean"].dropna():
            parts = [p.strip() for p in g_str.split(",") if p.strip() and p.strip() != "Unknown"]
            all_genres.update(parts)
        return sorted(list(all_genres))
