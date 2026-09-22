/**
 * CineSense AI — Modern OTT Streaming Web Application
 * Features:
 * - 1-Second Right-to-Left Auto-Transition Hero Slider with Mini Thumbnails
 * - Real-time Industry Switching (Bollywood / Hollywood / All Cinema)
 * - Latest Releases Row with Real TMDB Movie Thumbnails
 * - Top Trending Row with Large Numbered Overlays (1, 2, 3...)
 * - Select Genre Blocks (Romantic, Thriller, Comedy, Adventure, Action, Sci-Fi, etc.)
 * - Dynamic Watchlist Section & Buttons with LocalStorage Persistence
 * - Full Catalog Search & KMeans Cluster Recommendations
 * - Supervised Sentiment Analysis Studio & Model Metrics
 */

const API_BASE = "";

// State
const state = {
  featuredMovies: [],
  currentSlideIndex: 0,
  slideTimer: null,
  slideDuration: 3000, // 3 seconds per movie as requested
  timerStartTime: null,
  timerInterval: null,
  isPaused: false,

  selectedIndustry: "all", // "all", "bollywood", "hollywood"

  watchlistIds: new Set(),
  watchlistMovies: [],

  catalogMovies: [],
  catalogPage: 1,
  catalogTotalPages: 1,
  catalogTotal: 0,
  searchQuery: "",
  selectedGenre: "All",
  searchDebounce: null,

  activeModalMovieId: null
};

// Sleek Cinematic SVG Poster Fallbacks (Guarantees NO unrelated movie like Avatar ever appears)
function getFallbackPoster(title = "Movie", genre = "Cinema") {
  const safeTitle = (title || "Movie").replace(/[<>&"]/g, "").slice(0, 24);
  const safeGenre = (genre || "Cinema").replace(/[<>&"]/g, "").slice(0, 20);
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="300" height="450" viewBox="0 0 300 450">
    <defs>
      <linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#24070c"/>
        <stop offset="60%" stop-color="#110305"/>
        <stop offset="100%" stop-color="#050102"/>
      </linearGradient>
    </defs>
    <rect width="300" height="450" fill="url(#g)" stroke="rgba(229,9,20,0.3)" stroke-width="2" rx="12"/>
    <circle cx="150" cy="180" r="54" fill="#e50914" opacity="0.16"/>
    <text x="150" y="196" font-family="sans-serif" font-size="46" text-anchor="middle">🎬</text>
    <text x="150" y="270" font-family="'Inter', sans-serif" font-weight="800" font-size="18" fill="#ffffff" text-anchor="middle">${safeTitle}</text>
    <text x="150" y="296" font-family="'Inter', sans-serif" font-weight="600" font-size="13" fill="#ff4d4d" text-anchor="middle">${safeGenre}</text>
    <text x="150" y="420" font-family="'Inter', sans-serif" font-weight="700" font-size="11" fill="#888888" letter-spacing="2" text-anchor="middle">MOVIXA</text>
  </svg>`;
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

function getFallbackBackdrop(title = "Cinema") {
  const safeTitle = (title || "Cinema").replace(/[<>&"]/g, "").slice(0, 30);
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">
    <defs>
      <radialGradient id="rg" cx="50%" cy="40%" r="70%">
        <stop offset="0%" stop-color="#360810"/>
        <stop offset="50%" stop-color="#140305"/>
        <stop offset="100%" stop-color="#030102"/>
      </radialGradient>
    </defs>
    <rect width="1280" height="720" fill="url(#rg)"/>
    <text x="640" y="360" font-family="'Bebas Neue', sans-serif" font-size="72" letter-spacing="4" fill="rgba(229,9,20,0.22)" text-anchor="middle">${safeTitle.toUpperCase()}</text>
  </svg>`;
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

// DOM Cache
const dom = {
  // Navbar
  navWatchlistCount: document.getElementById("nav-watchlist-count"),
  headerWatchlistCount: document.getElementById("header-watchlist-count"),
  navSearchTrigger: document.getElementById("nav-search-trigger"),

  // Hero
  heroSlidesWrapper: document.getElementById("hero-slides-wrapper"),
  heroTitle: document.getElementById("hero-title"),
  heroStudioTag: document.getElementById("hero-studio-tag"),
  heroSubtag: document.getElementById("hero-subtag"),
  heroYear: document.getElementById("hero-year"),
  heroRuntime: document.getElementById("hero-runtime"),
  heroLang: document.getElementById("hero-lang"),
  heroGenres: document.getElementById("hero-genres"),
  heroRating: document.getElementById("hero-rating"),
  heroOverview: document.getElementById("hero-overview"),
  heroPlayBtn: document.getElementById("hero-play-btn"),
  heroWatchlistToggle: document.getElementById("hero-watchlist-toggle"),
  heroWatchlistText: document.getElementById("hero-watchlist-text"),
  heroPrevBtn: document.getElementById("hero-prev-btn"),
  heroNextBtn: document.getElementById("hero-next-btn"),
  heroThumbsContainer: document.getElementById("hero-thumbs-container"),
  heroTimerFill: document.getElementById("hero-timer-fill"),
  heroSection: document.getElementById("hero-carousel"),

  // Industry Switcher
  industryStatusBadge: document.getElementById("industry-status-badge"),

  // Tracks
  latestTrack: document.getElementById("latest-cards-track"),
  trendingTrack: document.getElementById("trending-cards-track"),
  genreTrack: document.getElementById("genre-cards-track"),
  watchlistTrack: document.getElementById("watchlist-cards-track"),
  watchlistTotalCount: document.getElementById("watchlist-total-count"),

  // Catalog
  catalogSearchInput: document.getElementById("catalog-search-input"),
  catalogClearSearch: document.getElementById("catalog-clear-search"),
  catalogGenrePills: document.getElementById("catalog-genre-pills"),
  catalogCountText: document.getElementById("catalog-count-text"),
  catalogMovieGrid: document.getElementById("catalog-movie-grid"),
  catalogPrevPage: document.getElementById("catalog-prev-page"),
  catalogNextPage: document.getElementById("catalog-next-page"),
  catalogPageLabel: document.getElementById("catalog-page-label"),

  // Modal
  movieModal: document.getElementById("movie-modal"),
  modalCloseCross: document.getElementById("modal-close-cross"),
  modalContentArea: document.getElementById("modal-content-area")
};

// =========================================================================
// Initialization
// =========================================================================
document.addEventListener("DOMContentLoaded", () => {
  loadWatchlistFromStorage();
  initIndustrySwitcher();
  initHeroSlider();
  loadLatestReleases();
  loadTopTrending();
  loadGenreButtons();
  loadCatalogMovies();
  initEventListeners();
});

// =========================================================================
// 0. INDUSTRY SWITCHER (Bollywood / Hollywood / All Cinema)
// =========================================================================
function initIndustrySwitcher() {
  document.querySelectorAll(".industry-pill-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".industry-pill-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const ind = btn.dataset.industry;
      selectIndustry(ind);
    });
  });
}

function selectIndustry(industry) {
  state.selectedIndustry = industry;
  state.catalogPage = 1;

  // Update badge
  if (dom.industryStatusBadge) {
    if (industry === "bollywood") {
      dom.industryStatusBadge.textContent = "Showing Bollywood Cinema";
    } else if (industry === "hollywood") {
      dom.industryStatusBadge.textContent = "Showing Hollywood Cinema";
    } else {
      dom.industryStatusBadge.textContent = "Showing All Cinema";
    }
  }

  // Update pills UI
  document.querySelectorAll(".industry-pill-btn").forEach(b => {
    b.classList.toggle("active", b.dataset.industry === industry);
  });

  // Reload all real-world dynamic rows with the selected industry!
  initHeroSlider();
  loadLatestReleases();
  loadTopTrending();
  loadCatalogMovies();
}

// =========================================================================
// 1. WATCHLIST PERSISTENCE & MANAGEMENT
// =========================================================================
function loadWatchlistFromStorage() {
  try {
    const raw = localStorage.getItem("movixa_watchlist") || localStorage.getItem("cinesense_watchlist");
    if (raw) {
      const arr = JSON.parse(raw);
      state.watchlistIds = new Set(arr.map(Number));
    }
  } catch (e) {
    state.watchlistIds = new Set();
  }
  updateWatchlistBadge();
  renderWatchlistRow();
}

function saveWatchlistToStorage() {
  localStorage.setItem("movixa_watchlist", JSON.stringify(Array.from(state.watchlistIds)));
  updateWatchlistBadge();
  renderWatchlistRow();
}

function toggleWatchlist(movieId) {
  const idNum = Number(movieId);
  if (state.watchlistIds.has(idNum)) {
    state.watchlistIds.delete(idNum);
  } else {
    state.watchlistIds.add(idNum);
  }
  saveWatchlistToStorage();

  // Update button states throughout UI
  document.querySelectorAll(`.watchlist-btn-ref[data-id="${idNum}"]`).forEach(btn => {
    btn.classList.toggle("active", state.watchlistIds.has(idNum));
    btn.textContent = state.watchlistIds.has(idNum) ? "✓" : "＋";
  });

  // Update hero watchlist button if active
  if (state.featuredMovies.length && state.featuredMovies[state.currentSlideIndex]?.id === idNum) {
    updateHeroWatchlistBtnState();
  }
}

function updateWatchlistBadge() {
  const count = state.watchlistIds.size;
  dom.navWatchlistCount.textContent = count;
  dom.headerWatchlistCount.textContent = count;
  dom.watchlistTotalCount.textContent = count;
}

async function renderWatchlistRow() {
  if (state.watchlistIds.size === 0) {
    dom.watchlistTrack.innerHTML = `
      <div class="empty-watchlist-msg">
        <span class="icon">🔖</span>
        <div>
          <h4 style="color: #fff; margin-bottom: 0.3rem;">Your Watchlist is Empty</h4>
          <p style="font-size: 0.88rem; color: var(--text-dim);">Browse movies from Latest Releases, Top Trending, or Full Catalog and click <strong>＋ Watchlist</strong> to save them here.</p>
        </div>
      </div>
    `;
    return;
  }

  dom.watchlistTrack.innerHTML = `<div style="padding: 2rem; color: var(--text-dim);">Loading watchlist items...</div>`;

  try {
    const ids = Array.from(state.watchlistIds);
    const res = await fetch(`${API_BASE}/api/movies/by-ids?ids=${ids.join(",")}`);
    if (!res.ok) throw new Error();
    const movies = await res.json();
    state.watchlistMovies = movies;

    if (!movies.length) {
      dom.watchlistTrack.innerHTML = `<div style="padding: 2rem; color: var(--text-dim);">No matching movies found.</div>`;
      return;
    }

    dom.watchlistTrack.innerHTML = "";
    movies.forEach(m => {
      const card = createPosterCard(m, { showRemoveBtn: true });
      dom.watchlistTrack.appendChild(card);
    });
  } catch (err) {
    dom.watchlistTrack.innerHTML = `<div style="padding: 2rem; color: var(--brand-red);">Failed to load watchlist.</div>`;
  }
}

// =========================================================================
// 2. HERO SLIDER (1-second Right-to-Left Auto-Transition)
// =========================================================================
async function initHeroSlider() {
  try {
    const res = await fetch(`${API_BASE}/api/featured?limit=8&industry=${state.selectedIndustry}`);
    if (!res.ok) throw new Error();
    state.featuredMovies = await res.json();

    if (!state.featuredMovies.length) return;

    renderHeroSlides();
    renderHeroThumbs();
    updateHeroSlide(0);
    startHeroTimer();
  } catch (err) {
    console.error("Hero slider load error:", err);
  }
}

function renderHeroSlides() {
  dom.heroSlidesWrapper.innerHTML = "";
  state.featuredMovies.forEach((m, idx) => {
    const slide = document.createElement("div");
    slide.className = `hero-slide ${idx === 0 ? 'active' : ''}`;
    slide.dataset.index = idx;

    const bgImage = m.backdrop_url || m.poster_url || getFallbackBackdrop(m.title);
    slide.style.backgroundImage = `url('${bgImage}')`;
    dom.heroSlidesWrapper.appendChild(slide);
  });
}

function renderHeroThumbs() {
  dom.heroThumbsContainer.innerHTML = "";
  state.featuredMovies.forEach((m, idx) => {
    const thumb = document.createElement("div");
    thumb.className = `hero-thumb-item ${idx === 0 ? 'active' : ''}`;
    thumb.dataset.index = idx;
    const thumbImg = m.backdrop_url || m.poster_url || getFallbackPoster(m.title, m.genres_clean);
    if (thumbImg) {
      thumb.style.backgroundImage = `url('${thumbImg}')`;
    } else {
      thumb.style.background = "#220609";
    }

    thumb.addEventListener("click", () => {
      goToHeroSlide(idx);
    });
    dom.heroThumbsContainer.appendChild(thumb);
  });
}

function updateHeroSlide(index) {
  state.currentSlideIndex = index;
  const m = state.featuredMovies[index];
  if (!m) return;

  // Update active slide class with right-to-left transition
  const slides = dom.heroSlidesWrapper.querySelectorAll(".hero-slide");
  slides.forEach((s, idx) => {
    s.classList.toggle("active", idx === index);
  });

  // Update thumbs
  const thumbs = dom.heroThumbsContainer.querySelectorAll(".hero-thumb-item");
  thumbs.forEach((t, idx) => {
    t.classList.toggle("active", idx === index);
  });

  // Animate text info
  dom.heroTitle.style.animation = "none";
  dom.heroTitle.offsetHeight; /* trigger reflow */
  dom.heroTitle.style.animation = "titleFadeIn 0.35s ease";

  dom.heroTitle.textContent = m.title;
  dom.heroYear.textContent = m.release_date ? m.release_date.slice(0, 4) : "2024";
  dom.heroRuntime.textContent = m.runtime ? `${Math.floor(m.runtime / 60)}h ${m.runtime % 60}m` : "2h 15m";
  dom.heroGenres.textContent = m.genres_clean || "Cinema";
  dom.heroRating.textContent = `⭐ ${Number(m.vote_average || 7.5).toFixed(1)}`;
  dom.heroOverview.textContent = m.overview || "No overview available.";

  updateHeroWatchlistBtnState();
}

function updateHeroWatchlistBtnState() {
  const m = state.featuredMovies[state.currentSlideIndex];
  if (!m) return;
  const inList = state.watchlistIds.has(m.id);
  dom.heroWatchlistToggle.classList.toggle("in-watchlist", inList);
  dom.heroWatchlistText.textContent = inList ? "In Watchlist (Saved)" : "Add to Watchlist";
}

function nextHeroSlide() {
  if (!state.featuredMovies.length) return;
  const nextIdx = (state.currentSlideIndex + 1) % state.featuredMovies.length;
  updateHeroSlide(nextIdx);
  resetTimerProgress();
}

function prevHeroSlide() {
  if (!state.featuredMovies.length) return;
  const prevIdx = (state.currentSlideIndex - 1 + state.featuredMovies.length) % state.featuredMovies.length;
  updateHeroSlide(prevIdx);
  resetTimerProgress();
}

function goToHeroSlide(idx) {
  updateHeroSlide(idx);
  resetTimerProgress();
}

function startHeroTimer() {
  clearInterval(state.slideTimer);
  clearInterval(state.timerInterval);

  state.timerStartTime = Date.now();

  // Progress Bar Animation (fill over 1000ms / 1 second)
  state.timerInterval = setInterval(() => {
    if (state.isPaused) return;
    const elapsed = Date.now() - state.timerStartTime;
    const pct = Math.min(100, (elapsed / state.slideDuration) * 100);
    dom.heroTimerFill.style.width = `${pct}%`;
  }, 40);

  // Main 1-second auto-transition ticker
  state.slideTimer = setInterval(() => {
    if (!state.isPaused) {
      nextHeroSlide();
    }
  }, state.slideDuration);
}

function resetTimerProgress() {
  state.timerStartTime = Date.now();
  dom.heroTimerFill.style.width = "0%";
}

// =========================================================================
// 3. LATEST RELEASES & TOP TRENDING ROWS
// =========================================================================
async function loadLatestReleases() {
  try {
    const res = await fetch(`${API_BASE}/api/latest?limit=15&industry=${state.selectedIndustry}`);
    if (!res.ok) throw new Error();
    const movies = await res.json();

    dom.latestTrack.innerHTML = "";
    movies.forEach(m => {
      const card = createPosterCard(m);
      dom.latestTrack.appendChild(card);
    });
  } catch (err) {
    dom.latestTrack.innerHTML = `<div style="padding: 2rem; color: var(--text-dim);">Unable to load latest releases.</div>`;
  }
}

async function loadTopTrending() {
  try {
    const res = await fetch(`${API_BASE}/api/trending?limit=15&industry=${state.selectedIndustry}`);
    if (!res.ok) throw new Error();
    const movies = await res.json();

    dom.trendingTrack.innerHTML = "";
    movies.forEach((m, idx) => {
      const card = createPosterCard(m, { rank: idx + 1 });
      dom.trendingTrack.appendChild(card);
    });
  } catch (err) {
    dom.trendingTrack.innerHTML = `<div style="padding: 2rem; color: var(--text-dim);">Unable to load trending movies.</div>`;
  }
}

// =========================================================================
// 4. POSTER CARD BUILDER (Real Movie Poster Thumbnails)
// =========================================================================
function createPosterCard(movie, options = {}) {
  const card = document.createElement("div");
  card.className = `movie-poster-card ${options.rank ? 'trending-card' : ''}`;
  card.dataset.id = movie.id;

  const year = movie.release_date ? movie.release_date.slice(0, 4) : "2024";
  const rating = Number(movie.vote_average || 0).toFixed(1);
  const inWatchlist = state.watchlistIds.has(movie.id);

  const fallbackDataUri = getFallbackPoster(movie.title, movie.genres_clean);
  const posterUrl = movie.poster_url;
  let imgHtml = "";
  if (posterUrl) {
    imgHtml = `<img class="poster-img-layer" src="${posterUrl}" alt="${escapeHtml(movie.title)}" loading="lazy" onerror="this.onerror=null; this.src='${fallbackDataUri}';">`;
  } else {
    imgHtml = `<img class="poster-img-layer" src="${fallbackDataUri}" alt="${escapeHtml(movie.title)}" loading="lazy">`;
  }

  let rankHtml = "";
  if (options.rank) {
    rankHtml = `<div class="rank-number-overlay">${options.rank}</div>`;
  }

  card.innerHTML = `
    ${imgHtml}
    ${rankHtml}
    <div class="card-hover-overlay">
      <h4 class="overlay-title">${escapeHtml(movie.title)}</h4>
      <div class="overlay-genres">${escapeHtml(movie.genres_clean || 'Film')}</div>
      <div class="overlay-meta">
        <span class="overlay-rating">⭐ ${rating}</span>
        <span>${year}</span>
      </div>
      <div class="card-action-btns">
        <button class="btn-card-action" onclick="event.stopPropagation(); openMovieModal(${movie.id})">
          Explore
        </button>
        <button class="btn-card-watchlist watchlist-btn-ref ${inWatchlist ? 'active' : ''}" 
                data-id="${movie.id}" 
                title="${inWatchlist ? 'Remove from Watchlist' : 'Add to Watchlist'}"
                onclick="event.stopPropagation(); toggleWatchlist(${movie.id})">
          ${inWatchlist ? '✓' : '＋'}
        </button>
      </div>
    </div>
  `;

  card.addEventListener("click", () => openMovieModal(movie.id));
  return card;
}

// =========================================================================
// 5. GENRE BLOCKS
// =========================================================================
function loadGenreButtons() {
  dom.genreTrack.querySelectorAll(".genre-block-card").forEach(block => {
    block.addEventListener("click", () => {
      const genre = block.dataset.genre;
      selectGenreAndScroll(genre);
    });
  });
}

function selectGenreAndScroll(genre) {
  state.selectedGenre = genre;
  state.catalogPage = 1;

  dom.catalogGenrePills.querySelectorAll(".pill-chip").forEach(chip => {
    chip.classList.toggle("active", chip.dataset.genre === genre);
  });

  loadCatalogMovies();

  const el = document.getElementById("explore-catalog-sec");
  if (el) {
    window.scrollTo({ top: el.offsetTop - 70, behavior: "smooth" });
  }
}

// =========================================================================
// 6. CATALOG EXPLORER (With Real Posters & KMeans Recommender)
// =========================================================================
async function loadCatalogMovies() {
  dom.catalogCountText.textContent = "Loading catalog movies...";
  dom.catalogMovieGrid.innerHTML = `
    <div style="grid-column: 1 / -1; text-align: center; padding: 4rem; color: var(--text-dim);">
      <p>Fetching movies from catalog...</p>
    </div>
  `;

  try {
    const params = new URLSearchParams({
      page: state.catalogPage,
      limit: 20,
      q: state.searchQuery,
      genre: state.selectedGenre === "All" ? "" : state.selectedGenre,
      industry: state.selectedIndustry
    });

    const res = await fetch(`${API_BASE}/api/movies?${params.toString()}`);
    if (!res.ok) throw new Error();
    const data = await res.json();

    state.catalogMovies = data.movies || [];
    state.catalogTotal = data.total || 0;
    state.catalogTotalPages = data.total_pages || 1;

    renderCatalogGrid();
    updateCatalogPagination();
  } catch (err) {
    dom.catalogMovieGrid.innerHTML = `
      <div style="grid-column: 1 / -1; text-align: center; padding: 3rem; color: var(--brand-red);">
        ⚠️ Failed to load catalog. Ensure backend server is active.
      </div>
    `;
  }
}

function renderCatalogGrid() {
  if (!state.catalogMovies.length) {
    dom.catalogMovieGrid.innerHTML = `
      <div style="grid-column: 1 / -1; text-align: center; padding: 4rem; color: var(--text-gray);">
        <h3>No movies found</h3>
        <p style="color: var(--text-dim);">Try searching another title or choosing another genre.</p>
      </div>
    `;
    dom.catalogCountText.textContent = "Showing 0 movies";
    return;
  }

  dom.catalogCountText.textContent = `Showing ${(state.catalogPage - 1) * 20 + 1}–${Math.min(state.catalogPage * 20, state.catalogTotal)} of ${state.catalogTotal.toLocaleString()} movies (Genre: ${state.selectedGenre})`;

  dom.catalogMovieGrid.innerHTML = "";
  state.catalogMovies.forEach(m => {
    const card = createPosterCard(m);
    dom.catalogMovieGrid.appendChild(card);
  });
}

function updateCatalogPagination() {
  dom.catalogPageLabel.textContent = `Page ${state.catalogPage} of ${Math.max(1, state.catalogTotalPages)}`;
  dom.catalogPrevPage.disabled = state.catalogPage <= 1;
  dom.catalogNextPage.disabled = state.catalogPage >= state.catalogTotalPages;
}

// =========================================================================
// 7. MOVIE DETAIL & CLUSTER RECOMMENDATION MODAL
//    (Formatted Exactly as Requested: Interstellar, Tagline, Rating, Year,
//     Runtime, Genres, Cluster #, and Overview Paragraph)
// =========================================================================
async function openMovieModal(movieId) {
  state.activeModalMovieId = movieId;
  dom.modalContentArea.innerHTML = `
    <div style="text-align: center; padding: 5rem; color: var(--text-dim);">
      <div style="font-size: 2.5rem; margin-bottom: 1rem;">⏳</div>
      <p>Loading movie insights, AI cluster recommendations, and audience reviews...</p>
    </div>
  `;
  dom.movieModal.style.display = "flex";
  document.body.style.overflow = "hidden";

  let movie = null;
  let recs = [];
  let reviews = [];

  try {
    // 1. Fetch movie details
    const movieRes = await fetch(`${API_BASE}/api/movie/${movieId}`);
    if (movieRes.ok) {
      movie = await movieRes.json();
    }
  } catch (e) {
    console.warn("API movie fetch error:", e);
  }

  // Fallback to local memory search if API failed or returned error
  if (!movie || movie.error) {
    movie = state.catalogMovies.find(m => m.id == movieId) || 
            state.featuredMovies.find(m => m.id == movieId) ||
            state.watchlistMovies.find(m => m.id == movieId);
  }

  if (!movie) {
    dom.modalContentArea.innerHTML = `
      <div style="text-align: center; padding: 4rem; color: var(--brand-red);">
        <h3>⚠️ Movie details temporarily unavailable.</h3>
        <p style="color: var(--text-dim); margin-top: 0.5rem;">Please close and select another movie.</p>
      </div>
    `;
    return;
  }

  // 2. Fetch recommendations
  try {
    const recsRes = await fetch(`${API_BASE}/api/recommend/${movie.id}?top_n=8`);
    if (recsRes.ok) {
      recs = await recsRes.json();
    }
  } catch (e) {
    console.warn("Recs fetch error:", e);
  }
  if (!Array.isArray(recs)) recs = [];

  // 3. Fetch reviews
  try {
    const reviewsRes = await fetch(`${API_BASE}/api/movie/${movie.id}/reviews`);
    if (reviewsRes.ok) {
      reviews = await reviewsRes.json();
    }
  } catch (e) {
    console.warn("Reviews fetch error:", e);
  }
  if (!Array.isArray(reviews)) reviews = [];

  renderModalView(movie, recs, reviews);
}

function renderModalView(movie, recs, reviews) {
  const year = movie.release_date ? movie.release_date.slice(0, 4) : "2024";
  const rating = Number(movie.vote_average || 7.5).toFixed(1);
  const runtime = movie.runtime ? Number(movie.runtime) : 135;
  const runtimeStr = `${Math.floor(runtime / 60)}h ${runtime % 60}m`;
  const inWatchlist = state.watchlistIds.has(movie.id);

  const fallbackPoster = getFallbackPoster(movie.title, movie.genres_clean);
  const posterImg = movie.poster_url || fallbackPoster;
  const tagline = movie.tagline ? `“${escapeHtml(movie.tagline)}”` : "";
  const clusterNum = movie.cluster !== undefined ? movie.cluster : 0;
  const genresStr = movie.genres_clean || "Cinema";

  // Recommended Movies
  let recsHtml = "";
  if (recs && recs.length) {
    recsHtml = recs.map(rec => {
      const recFallback = getFallbackPoster(rec.title, rec.genres_clean);
      const recImg = rec.poster_url || recFallback;
      const matchPct = Math.round((rec.similarity_score || 0.82) * 100);
      return `
        <div class="modal-rec-card" onclick="openMovieModal(${rec.id})">
          <img class="modal-rec-thumb" src="${recImg}" alt="${escapeHtml(rec.title)}" loading="lazy" onerror="this.onerror=null; this.src='${recFallback}';">
          <div class="modal-rec-body">
            <span class="modal-rec-title">${escapeHtml(rec.title)}</span>
            <span class="modal-rec-score">⭐ ${Number(rec.vote_average || 0).toFixed(1)} • ${matchPct}% match</span>
          </div>
        </div>
      `;
    }).join("");
  } else {
    recsHtml = `<p style="color: var(--text-dim); grid-column: 1 / -1;">No similar cluster movies found.</p>`;
  }

  // Reviews
  let reviewsHtml = "";
  if (reviews && reviews.length) {
    reviewsHtml = reviews.map(rev => {
      const isPos = rev.sentiment === "positive";
      return `
        <div class="review-item-card">
          <div class="review-top-meta">
            <span class="reviewer-name">👤 ${escapeHtml(rev.user || 'Movie Reviewer')}</span>
            <span class="sentiment-pill ${isPos ? 'positive' : 'negative'}">
              ${isPos ? '👍 Positive' : '👎 Negative'} (${Math.round((rev.confidence || 0.88) * 100)}%)
            </span>
          </div>
          <p class="review-text-content">${escapeHtml(rev.review)}</p>
        </div>
      `;
    }).join("");
  } else {
    reviewsHtml = `<p style="color: var(--text-dim);">No audience reviews yet. Be the first to review!</p>`;
  }

  // RENDER MODAL WITH EXACT LAYOUT SPECIFIED BY USER
  dom.modalContentArea.innerHTML = `
    <div class="modal-movie-top">
      <div class="modal-poster-side">
        <img src="${posterImg}" alt="${escapeHtml(movie.title)}" onerror="this.onerror=null; this.src='${fallbackPoster}';">
      </div>
      <div class="modal-info-side">
        <h2 class="modal-title">${escapeHtml(movie.title)}</h2>
        ${tagline ? `<div class="modal-tagline">${tagline}</div>` : ''}

        <div class="modal-meta-chips">
          <span class="modal-chip">⭐ ${rating} / 10</span>
          <span class="modal-chip">📅 ${year}</span>
          <span class="modal-chip">⏱️ ${runtimeStr}</span>
          <span class="modal-chip">🏷️ ${escapeHtml(genresStr)}</span>
          <span class="modal-chip modal-cluster-chip">Cluster #${clusterNum}</span>
        </div>

        <p class="modal-overview">${escapeHtml(movie.overview)}</p>

        <div class="modal-action-bar">
          <button class="btn btn-outline" id="modal-watchlist-toggle-btn" onclick="toggleModalWatchlist(${movie.id})">
            ${inWatchlist ? '✓ Saved in Watchlist' : '＋ Add to Watchlist'}
          </button>
        </div>
      </div>
    </div>

    <!-- Recommendations -->
    <h3 class="modal-recs-title">SIMILAR MOVIES (CLUSTER #${clusterNum})</h3>
    <div class="modal-recs-grid">
      ${recsHtml}
    </div>

    <!-- Audience Reviews & Sentiment Intelligence Studio -->
    <div class="modal-reviews-block">
      <h3 style="font-family: var(--font-display); font-size: 1.8rem; color: #fff; margin-bottom: 0.8rem;">
        AUDIENCE REVIEWS & LIVE SENTIMENT STUDIO
      </h3>

      <!-- Review & Sentiment Test Form -->
      <form id="modal-review-form" class="modal-review-form">
        <h4 style="color: #fff; font-size: 0.95rem; font-weight: 700;">
          Write or Test a Movie Review (Analyzed Live by AI Classifier):
        </h4>

        <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.3rem;">
          <span style="font-size: 0.8rem; color: var(--text-dim); align-self: center;">Quick Test:</span>
          <button type="button" class="preset-chip pos-chip" id="m-preset-pos">👍 Positive Review</button>
          <button type="button" class="preset-chip neg-chip" id="m-preset-neg">👎 Negative Review</button>
        </div>

        <input type="text" id="modal-reviewer-name" placeholder="Your Name (e.g. Alex)" />
        <textarea id="modal-reviewer-text" rows="3" placeholder="Share your honest critique or review to test live sentiment..." required></textarea>

        <div style="display: flex; gap: 0.8rem; align-items: center; margin-top: 0.3rem;">
          <button type="submit" class="btn btn-primary btn-sm" id="btn-modal-submit">
            ⚡ POST REVIEW & PREDICT SENTIMENT
          </button>
          <button type="button" class="btn btn-outline btn-sm" id="btn-modal-test-only">
            Test AI Prediction Only
          </button>
        </div>
      </form>

      <!-- LIVE SENTIMENT INTELLIGENCE RESULT CARD (Requested by User) -->
      <div id="modal-sentiment-result-card" class="modal-sentiment-card" style="display: none; margin-top: 1.2rem;">
        <div class="modal-sentiment-header">
          <span class="tokens-caption" style="margin: 0;">PREDICTED SENTIMENT</span>
          <div class="sentiment-result-badge" id="m-sent-badge">
            <span id="m-sent-emoji">👍</span>
            <span id="m-sent-text">POSITIVE REVIEW</span>
          </div>
        </div>

        <div class="confidence-container">
          <div class="confidence-info">
            <span>Model Confidence</span>
            <span class="confidence-value" id="m-sent-conf">92.4%</span>
          </div>
          <div class="confidence-bar-track">
            <div class="confidence-bar-fill" id="m-sent-bar"></div>
          </div>
        </div>

        <div class="prob-dual-grid">
          <div class="prob-stat-card pos-stat">
            <span class="prob-stat-title">Positive Probability</span>
            <span class="prob-stat-num" id="m-sent-pos">0.92</span>
          </div>
          <div class="prob-stat-card neg-stat">
            <span class="prob-stat-title">Negative Probability</span>
            <span class="prob-stat-num" id="m-sent-neg">0.08</span>
          </div>
        </div>

        <div class="sanitized-tokens-box">
          <span class="tokens-caption">Preprocessed NLP Tokens:</span>
          <div class="tokens-string" id="m-sent-tokens">None</div>
        </div>
      </div>

      <!-- Reviews Stream List -->
      <h4 style="color: #fff; font-size: 1rem; margin: 1.8rem 0 0.8rem;">Audience Reviews</h4>
      <div id="modal-reviews-list">
        ${reviewsHtml}
      </div>
    </div>
  `;

  // Wire up presets, testing, and submission inside modal
  const form = document.getElementById("modal-review-form");
  const posPreset = document.getElementById("m-preset-pos");
  const negPreset = document.getElementById("m-preset-neg");
  const testOnlyBtn = document.getElementById("btn-modal-test-only");
  const nameInput = document.getElementById("modal-reviewer-name");
  const textInput = document.getElementById("modal-reviewer-text");

  function displaySentimentResult(prediction) {
    const card = document.getElementById("modal-sentiment-result-card");
    const badge = document.getElementById("m-sent-badge");
    const emoji = document.getElementById("m-sent-emoji");
    const textEl = document.getElementById("m-sent-text");
    const confEl = document.getElementById("m-sent-conf");
    const bar = document.getElementById("m-sent-bar");
    const posEl = document.getElementById("m-sent-pos");
    const negEl = document.getElementById("m-sent-neg");
    const tokensEl = document.getElementById("m-sent-tokens");

    if (!card) return;
    const isPos = prediction.sentiment === "positive";
    const confPct = prediction.score_percent || (prediction.confidence * 100).toFixed(1);
    const posProb = prediction.probabilities?.positive ?? (isPos ? prediction.confidence : 1 - prediction.confidence);
    const negProb = prediction.probabilities?.negative ?? (isPos ? 1 - prediction.confidence : prediction.confidence);

    badge.className = `sentiment-result-badge ${isPos ? 'positive' : 'negative'}`;
    emoji.textContent = isPos ? "👍" : "👎";
    textEl.textContent = isPos ? "POSITIVE REVIEW" : "NEGATIVE REVIEW";
    confEl.textContent = `${confPct}%`;
    bar.style.width = `${confPct}%`;
    bar.className = `confidence-bar-fill ${isPos ? '' : 'negative'}`;
    posEl.textContent = Number(posProb).toFixed(2);
    negEl.textContent = Number(negProb).toFixed(2);
    tokensEl.textContent = prediction.cleaned_text || "No alphanumeric tokens";
    card.style.display = "block";
    card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  if (posPreset) {
    posPreset.addEventListener("click", async () => {
      textInput.value = "An astonishing achievement in visual cinema! The direction, pacing, and emotional depth were phenomenal.";
      try {
        const res = await fetch(`${API_BASE}/api/sentiment`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ review: textInput.value })
        });
        if (res.ok) {
          const data = await res.json();
          displaySentimentResult(data);
        }
      } catch (e) {}
    });
  }

  if (negPreset) {
    negPreset.addEventListener("click", async () => {
      textInput.value = "Terrible script with laughable dialogue and sluggish pacing. A total waste of two hours.";
      try {
        const res = await fetch(`${API_BASE}/api/sentiment`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ review: textInput.value })
        });
        if (res.ok) {
          const data = await res.json();
          displaySentimentResult(data);
        }
      } catch (e) {}
    });
  }

  if (testOnlyBtn) {
    testOnlyBtn.addEventListener("click", async () => {
      const text = textInput.value.trim();
      if (!text) {
        alert("Please enter a review to test.");
        return;
      }
      try {
        testOnlyBtn.textContent = "Analyzing...";
        const res = await fetch(`${API_BASE}/api/sentiment`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ review: text })
        });
        if (res.ok) {
          const data = await res.json();
          displaySentimentResult(data);
        }
      } catch (e) {
        alert("Sentiment analysis error.");
      } finally {
        testOnlyBtn.textContent = "Test AI Prediction Only";
      }
    });
  }

  if (form) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const name = nameInput.value.trim() || "Anonymous";
      const text = textInput.value.trim();
      if (!text) return;

      const submitBtn = document.getElementById("btn-modal-submit");
      if (submitBtn) submitBtn.textContent = "Analyzing & Posting...";

      try {
        const res = await fetch(`${API_BASE}/api/movie/${movie.id}/reviews`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ review: text, user: name })
        });

        if (res.ok) {
          const saved = await res.json();
          displaySentimentResult(saved);

          const list = document.getElementById("modal-reviews-list");
          const isPos = saved.sentiment === "positive";
          const el = document.createElement("div");
          el.className = "review-item-card";
          el.innerHTML = `
            <div class="review-top-meta">
              <span class="reviewer-name">👤 ${escapeHtml(saved.user)} (Just Now)</span>
              <span class="sentiment-pill ${isPos ? 'positive' : 'negative'}">
                ${isPos ? '👍 Positive' : '👎 Negative'} (${Math.round((saved.confidence || 0.88) * 100)}%)
              </span>
            </div>
            <p class="review-text-content">${escapeHtml(saved.review)}</p>
          `;
          list.prepend(el);
          textInput.value = "";
        }
      } catch (err) {
        alert("Error submitting review.");
      } finally {
        if (submitBtn) submitBtn.textContent = "⚡ POST REVIEW & PREDICT SENTIMENT";
      }
    });
  }
}

function toggleModalWatchlist(movieId) {
  toggleWatchlist(movieId);
  const inList = state.watchlistIds.has(Number(movieId));
  const btn = document.getElementById("modal-watchlist-toggle-btn");
  if (btn) {
    btn.textContent = inList ? "✓ Saved in Watchlist" : "＋ Add to Watchlist";
  }
}

function closeMovieModal() {
  dom.movieModal.style.display = "none";
  document.body.style.overflow = "auto";
  state.activeModalMovieId = null;
}

// =========================================================================
// 10. EVENT LISTENERS
// =========================================================================
function initEventListeners() {
  // Hero Carousel Navigation
  dom.heroNextBtn.addEventListener("click", nextHeroSlide);
  dom.heroPrevBtn.addEventListener("click", prevHeroSlide);

  // Hero Play & Watchlist Buttons
  dom.heroPlayBtn.addEventListener("click", () => {
    if (state.featuredMovies.length) {
      const activeMovie = state.featuredMovies[state.currentSlideIndex];
      openMovieModal(activeMovie.id);
    }
  });

  dom.heroWatchlistToggle.addEventListener("click", () => {
    if (state.featuredMovies.length) {
      const activeMovie = state.featuredMovies[state.currentSlideIndex];
      toggleWatchlist(activeMovie.id);
    }
  });

  // Horizontal Channel Row Scroll Buttons
  document.querySelectorAll(".row-scroll-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const trackId = btn.dataset.target;
      const dir = parseInt(btn.dataset.dir || "1", 10);
      const track = document.getElementById(trackId);
      if (track) {
        track.scrollBy({ left: dir * 480, behavior: "smooth" });
      }
    });
  });

  // Catalog Search Debounce
  dom.catalogSearchInput.addEventListener("input", (e) => {
    const q = e.target.value;
    dom.catalogClearSearch.style.display = q ? "block" : "none";
    clearTimeout(state.searchDebounce);
    state.searchDebounce = setTimeout(() => {
      state.searchQuery = q.trim();
      state.catalogPage = 1;
      loadCatalogMovies();
    }, 350);
  });

  dom.catalogClearSearch.addEventListener("click", () => {
    dom.catalogSearchInput.value = "";
    dom.catalogClearSearch.style.display = "none";
    state.searchQuery = "";
    state.catalogPage = 1;
    loadCatalogMovies();
  });

  // Catalog Pagination
  dom.catalogPrevPage.addEventListener("click", () => {
    if (state.catalogPage > 1) {
      state.catalogPage--;
      loadCatalogMovies();
      window.scrollTo({ top: document.getElementById("explore-catalog-sec").offsetTop - 70, behavior: "smooth" });
    }
  });

  dom.catalogNextPage.addEventListener("click", () => {
    if (state.catalogPage < state.catalogTotalPages) {
      state.catalogPage++;
      loadCatalogMovies();
      window.scrollTo({ top: document.getElementById("explore-catalog-sec").offsetTop - 70, behavior: "smooth" });
    }
  });

  // Genre Pills
  initGenrePills();

  // Modal Close
  dom.modalCloseCross.addEventListener("click", closeMovieModal);
  dom.movieModal.addEventListener("click", (e) => {
    if (e.target === dom.movieModal) closeMovieModal();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && dom.movieModal.style.display === "flex") {
      closeMovieModal();
    }
  });

  // Quick Search Focus
  dom.navSearchTrigger.addEventListener("click", () => {
    const searchSection = document.getElementById("explore-catalog-sec");
    if (searchSection) {
      window.scrollTo({ top: searchSection.offsetTop - 70, behavior: "smooth" });
      dom.catalogSearchInput.focus();
    }
  });
}

async function initGenrePills() {
  try {
    const res = await fetch(`${API_BASE}/api/genres`);
    if (!res.ok) return;
    const genres = await res.json();

    dom.catalogGenrePills.innerHTML = `<button class="pill-chip active" data-genre="All">All Genres</button>`;
    genres.forEach(g => {
      const btn = document.createElement("button");
      btn.className = "pill-chip";
      btn.dataset.genre = g;
      btn.textContent = g;
      btn.addEventListener("click", () => {
        dom.catalogGenrePills.querySelectorAll(".pill-chip").forEach(c => c.classList.remove("active"));
        btn.classList.add("active");
        state.selectedGenre = g;
        state.catalogPage = 1;
        loadCatalogMovies();
      });
      dom.catalogGenrePills.appendChild(btn);
    });

    dom.catalogGenrePills.firstElementChild.addEventListener("click", (e) => {
      dom.catalogGenrePills.querySelectorAll(".pill-chip").forEach(c => c.classList.remove("active"));
      e.target.classList.add("active");
      state.selectedGenre = "All";
      state.catalogPage = 1;
      loadCatalogMovies();
    });
  } catch (e) {
    console.error("Genre pills error:", e);
  }
}

// Utility: Escape HTML
function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
