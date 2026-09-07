"""
Text preprocessing utilities for the IMDb Movie Recommendation System.

This module contains reusable functions for cleaning movie titles and
storylines. The same preprocessing pipeline can be applied to both the
movie dataset and a user's input storyline.
"""

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# Reusable English stopword set.
STOP_WORDS = set(ENGLISH_STOP_WORDS)


def clean_movie_title(title: str) -> str:
    """
    Remove the numerical index prefix from a movie title.

    Example
    -------
    "1. The Fall Guy" -> "The Fall Guy"
    """
    if not isinstance(title, str):
        raise TypeError("Movie title must be a string.")

    title = re.sub(r"^\s*\d+\.\s*", "", title)

    return title.strip()


def normalize_text(text: str) -> str:
    """
    Normalize text for NLP processing.

    Operations performed:
    - Convert text to lowercase.
    - Remove non-alphabetic characters.
    - Normalize repeated whitespace.
    - Strip leading and trailing whitespace.
    """
    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize_text(text: str) -> list[str]:
    """
    Tokenize normalized text using whitespace-based tokenization.
    """
    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    return text.split()


def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Remove common English stopwords from a token list.
    """
    if not isinstance(tokens, list):
        raise TypeError("Tokens must be provided as a list.")

    return [
        token
        for token in tokens
        if token not in STOP_WORDS
    ]


def preprocess_storyline(text: str) -> str:
    """
    Apply the complete storyline preprocessing pipeline.

    Pipeline
    --------
    Raw text
        ↓
    Normalize
        ↓
    Tokenize
        ↓
    Remove stopwords
        ↓
    Reconstruct processed text

    Returns
    -------
    str
        Processed storyline ready for TF-IDF vectorization.
    """
    normalized_text = normalize_text(text)

    tokens = tokenize_text(normalized_text)

    filtered_tokens = remove_stopwords(tokens)

    return " ".join(filtered_tokens)