from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Project directories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Dataset
RAW_DATA_PATH = DATA_DIR / "IMDBRecSys.csv"
PROCESSED_DATA_PATH = DATA_DIR / "IMDBRecSys_Processed.csv"

# Model artifacts
MOVIE_DATA_PATH = MODELS_DIR / "movie_data.pkl"
TFIDF_MATRIX_PATH = MODELS_DIR / "tfidf_matrix.pkl"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"