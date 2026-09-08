"""
Streamlit application for the IMDb Movie Recommendation System.

The application accepts a movie storyline, plot, or short description
and recommends the five most similar movies using the trained
TF-IDF + cosine similarity recommendation engine.
"""

import streamlit as st

from src.model_loader import load_models
from src.recommender import MovieRecommender


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="IMDb Movie Recommender",
    page_icon="🎬",
    layout="wide",
)


# ---------------------------------------------------------------------
# Load recommendation engine
# ---------------------------------------------------------------------

@st.cache_resource
def load_recommender() -> MovieRecommender:
    """
    Load model artifacts once and initialize the recommender.

    Streamlit caches this object so that model files are not repeatedly
    loaded every time the user interacts with the application.
    """
    models = load_models()

    return MovieRecommender(
        movie_data=models["movie_data"],
        tfidf_matrix=models["tfidf_matrix"],
        tfidf_vectorizer=models["tfidf_vectorizer"],
    )


# ---------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------

def main() -> None:
    """Run the Streamlit application."""

    # Header
    st.title("🎬 IMDb Movie Recommendation System")

    st.markdown(
        """
        Enter a movie **storyline, plot, or short description** and
        discover five movies with the most similar storylines.
        """
    )

    st.divider()

    # -----------------------------------------------------------------
    # Sidebar
    # -----------------------------------------------------------------

    with st.sidebar:
        st.header("ℹ️ About")

        st.write(
            """
            This application uses a content-based recommendation
            approach.

            **Pipeline**

            Storyline  
            ↓  
            Text preprocessing  
            ↓  
            TF-IDF vectorization  
            ↓  
            Cosine similarity  
            ↓  
            Top 5 recommendations
            """
        )

        st.divider()

        st.caption(
            "Similarity scores represent textual similarity between "
            "the entered storyline and the movie storylines."
        )

    # -----------------------------------------------------------------
    # Input section
    # -----------------------------------------------------------------

    st.subheader("📝 Enter a Storyline")

    with st.form("recommendation_form"):

        storyline = st.text_area(
            "Movie storyline",
            placeholder=(
                "Example: A group of astronauts becomes stranded on "
                "a distant planet and must work together to survive "
                "an unknown threat."
            ),
            height=180,
            max_chars=2000,
            help=(
                "Describe the movie plot or story. "
                "Do not enter a movie title."
            ),
        )

        submitted = st.form_submit_button(
            "🎯 Recommend Movies",
            type="primary",
            use_container_width=True,
        )

    # -----------------------------------------------------------------
    # Recommendation logic
    # -----------------------------------------------------------------

    if submitted:

        if not storyline.strip():
            st.warning(
                "Please enter a movie storyline before requesting recommendations."
            )
            return

        with st.spinner("Finding similar movies..."):

            try:
                recommender = load_recommender()

                recommendations = recommender.recommend(
                    storyline=storyline,
                    top_n=5,
                )

            except ValueError as error:
                st.warning(str(error))
                return

            except FileNotFoundError as error:
                st.error(
                    "The recommendation model is not available.\n\n"
                    f"{error}"
                )
                st.info(
                    "Please generate the model artifacts by running "
                    "the recommender notebook first."
                )
                return

            except Exception as error:
                st.error(
                    "Something went wrong while generating recommendations."
                )
                st.caption(f"Technical details: {error}")
                return

        # -------------------------------------------------------------
        # Results
        # -------------------------------------------------------------

        st.subheader("🎬 Recommended Movies")

        st.caption(
            "Ranked from highest to lowest storyline similarity."
        )

        for index, row in recommendations.iterrows():

            similarity_percentage = (
                row["similarity_score"] * 100
            )

            with st.container(border=True):

                col1, col2 = st.columns(
                    [0.78, 0.22]
                )

                with col1:
                    st.markdown(
                        f"### {index + 1}. {row['Movie_Title']}"
                    )

                with col2:
                    st.metric(
                        "Similarity",
                        f"{similarity_percentage:.1f}%",
                    )

                st.markdown("**Storyline**")

                st.write(row["Storyline_Raw"])

        st.success(
            "Recommendation complete! 🍿"
        )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()