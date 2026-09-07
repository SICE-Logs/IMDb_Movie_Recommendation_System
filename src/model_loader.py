"""
Model and data artifact loading utilities.

This module is responsible only for locating, loading, and validating
the artifacts required by the movie recommendation engine.
"""

import joblib
import pandas as pd

from .config import (
    MOVIE_DATA_PATH,
    TFIDF_MATRIX_PATH,
    TFIDF_VECTORIZER_PATH,
)


def _validate_artifact_paths() -> None:
    """Ensure all required model artifacts exist."""
    artifact_paths = {
        "movie data": MOVIE_DATA_PATH,
        "TF-IDF matrix": TFIDF_MATRIX_PATH,
        "TF-IDF vectorizer": TFIDF_VECTORIZER_PATH,
    }

    missing = [
        f"{name}: {path}"
        for name, path in artifact_paths.items()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "The following model artifacts are missing:\n"
            + "\n".join(missing)
        )


def load_models() -> dict:
    """
    Load all artifacts required by the recommendation engine.

    Returns
    -------
    dict
        Dictionary containing:
        - movie_data
        - tfidf_matrix
        - tfidf_vectorizer

    Raises
    ------
    FileNotFoundError
        If any required artifact is missing.
    ValueError
        If the loaded artifacts are inconsistent.
    """

    _validate_artifact_paths()

    movie_data = pd.read_pickle(MOVIE_DATA_PATH)
    tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)
    tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)

    # Validate expected movie data structure.
    required_columns = {
        "Movie_Title",
        "Storyline_Processed",
    }

    missing_columns = required_columns.difference(movie_data.columns)

    if missing_columns:
        raise ValueError(
            "Movie data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # Validate alignment between movie records and TF-IDF vectors.
    if len(movie_data) != tfidf_matrix.shape[0]:
        raise ValueError(
            "Movie data and TF-IDF matrix are misaligned: "
            f"{len(movie_data)} movie records vs "
            f"{tfidf_matrix.shape[0]} TF-IDF vectors."
        )

    return {
        "movie_data": movie_data,
        "tfidf_matrix": tfidf_matrix,
        "tfidf_vectorizer": tfidf_vectorizer,
    }