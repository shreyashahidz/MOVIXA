import re
import json
import ast

# Pre-compiled regex patterns for speed
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"http\S+|www\S+")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
WHITESPACE_RE = re.compile(r"\s+")

# Core english stopwords for clean token filtering
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
    "by", "can", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", 
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", 
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", 
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", 
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she", 
    "should", "so", "some", "such", "t", "than", "that", "the", "their", "theirs", "them", 
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", 
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", 
    "while", "who", "whom", "why", "will", "with", "you", "your", "yours", "yourself", "yourselves"
}

# Simple suffix lemmatization for common words
def simple_lemmatize(word):
    if len(word) > 4:
        if word.endswith("ing"):
            return word[:-3]
        elif word.endswith("ly"):
            return word[:-2]
        elif word.endswith("ed"):
            return word[:-2]
        elif word.endswith("es"):
            return word[:-2]
        elif word.endswith("s") and not word.endswith("ss"):
            return word[:-1]
    return word


def clean_text(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
    """
    Cleans raw review or overview text:
    - Strips HTML tags
    - Strips URLs
    - Strips non-alphabetic characters
    - Lowercases text
    - Removes common stopwords
    - Applies lemmatization
    """
    if not isinstance(text, str):
        return ""
        
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = text.lower()
    
    tokens = text.split()
    if remove_stopwords:
        tokens = [w for w in tokens if w not in STOPWORDS and len(w) > 1]
    if lemmatize:
        tokens = [simple_lemmatize(w) for w in tokens]
        
    return " ".join(tokens)


def extract_json_names(val) -> str:
    """
    Extracts name attributes from TMDB JSON strings e.g. [{"id": 28, "name": "Action"}] -> 'Action'
    """
    if not val or not isinstance(val, str) or val.strip() == "[]":
        return ""
    try:
        data = json.loads(val)
        if isinstance(data, list):
            return " ".join([item.get("name", "").replace(" ", "") for item in data if isinstance(item, dict)])
    except Exception:
        pass

    try:
        data = ast.literal_eval(val)
        if isinstance(data, list):
            return " ".join([item.get("name", "").replace(" ", "") for item in data if isinstance(item, dict)])
    except Exception:
        pass

    return ""


def create_tags(overview: str, genres: str, keywords: str) -> str:
    """
    Combines movie overview, parsed genres, and parsed keywords into unified tags string.
    """
    cleaned_overview = clean_text(str(overview or ""))
    genres_str = extract_json_names(genres)
    keywords_str = extract_json_names(keywords)
    
    parts = [genres_str, keywords_str, cleaned_overview]
    return " ".join([p for p in parts if p]).strip()
