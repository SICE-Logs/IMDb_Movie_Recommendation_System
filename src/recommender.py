"""
Core recommendation engine for the IMDb Movie Recommendation System.

This module:
- preprocesses user-provided storylines,
- transforms them using the fitted TF-IDF vectorizer,
- calculates cosine similarity against movie vectors,
- returns the top-N most similar movies.
"""

from typing import Any

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import preprocess_storyline


class MovieRecommender:
    """
    Storyline-based content recommendation engine.

    Parameters
    ----------
    movie_data : pandas.DataFrame
        Movie metadata and processed storylines.

    tfidf_matrix : scipy.sparse matrix
        TF-IDF representations of all movie storylines.

    tfidf_vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer used to create the movie vectors.
    """

    def __init__(
        self,
        movie_data: pd.DataFrame,
        tfidf_matrix: Any,
        tfidf_vectorizer: Any,
    ) -> None:

        required_columns = {
            "Movie_Title",
            "Storyline_Raw",
            "Storyline_Processed",
        }

        missing_columns = required_columns.difference(
            movie_data.columns
        )

        if missing_columns:
            raise ValueError(
                "Movie data is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        if len(movie_data) != tfidf_matrix.shape[0]:
            raise ValueError(
                "Movie data and TF-IDF matrix are misaligned: "
                f"{len(movie_data)} movie records vs "
                f"{tfidf_matrix.shape[0]} TF-IDF vectors."
            )

        self.movie_data = movie_data.reset_index(drop=True)
        self.tfidf_matrix = tfidf_matrix
        self.tfidf_vectorizer = tfidf_vectorizer

    def recommend(
        self,
        storyline: str,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        Recommend movies based on a user's storyline.

        Parameters
        ----------
        storyline : str
            User-provided movie storyline, plot, or description.

        top_n : int, default=5
            Number of recommendations to return.

        Returns
        -------
        pandas.DataFrame
            Recommended movies ranked by cosine similarity.
        """

        if not isinstance(storyline, str):
            raise TypeError("Storyline must be a string.")

        if not storyline.strip():
            raise ValueError("Storyline cannot be empty.")

        if not isinstance(top_n, int):
            raise TypeError("top_n must be an integer.")

        if top_n < 1:
            raise ValueError("top_n must be at least 1.")

        top_n = min(top_n, len(self.movie_data))

        # Apply the exact same preprocessing used for movie storylines.
        processed_query = preprocess_storyline(storyline)

        if not processed_query:
            raise ValueError(
                "Storyline contains no meaningful words after preprocessing."
            )

        # Transform the processed user query into the fitted TF-IDF space.
        query_vector = self.tfidf_vectorizer.transform(
            [processed_query]
        )

        # Calculate similarity between the query and every movie.
        similarity_scores = cosine_similarity(
            query_vector,
            self.tfidf_matrix,
        ).flatten()

        # Rank movies from highest to lowest similarity.
        top_indices = similarity_scores.argsort()[::-1][:top_n]

        recommendations = self.movie_data.iloc[
            top_indices
        ][
            [
                "Movie_Title",
                "Storyline_Raw",
                "Storyline_Processed",
            ]
        ].copy()

        recommendations["similarity_score"] = (
            similarity_scores[top_indices]
        )

        return recommendations.reset_index(drop=True)

    def recommend_dicts(
        self,
        storyline: str,
        top_n: int = 5,
    ) -> list[dict]:
        """
        Return recommendations as a list of dictionaries.

        This format is convenient for application/UI layers.
        """

        recommendations = self.recommend(
            storyline,
            top_n,
        )

        return recommendations.to_dict(
            orient="records"
        )