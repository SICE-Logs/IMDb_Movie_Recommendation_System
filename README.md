# 🎬 IMDb Movie Recommendation System

A storyline-based content recommendation system that recommends the **top 5 movies** based on a user's movie plot, storyline, or short description.

The project uses **Natural Language Processing (NLP)** with **TF-IDF vectorization** and **Cosine Similarity** to compare the user's storyline with movie storylines and rank the most similar results.

---

## 📌 Project Overview

Traditional movie recommendation systems often depend on ratings, popularity, or user history. This project takes a different approach: it recommends movies based on the **story itself**.

A user enters a storyline such as:

> A group of astronauts becomes stranded on a distant planet and must work together to survive an unknown threat.

The system preprocesses the text, converts it into a TF-IDF representation, compares it with the movie storyline dataset using cosine similarity, and returns the five most similar movies.

### Core Pipeline

```text
User Storyline
      ↓
Text Preprocessing
      ↓
TF-IDF Vectorization
      ↓
Cosine Similarity
      ↓
Similarity Ranking
      ↓
Top 5 Movie Recommendations
```

---

## 🎯 Problem Statement

The objective is to build an interactive movie recommendation system that can identify movies with storylines similar to a user's input storyline.

The project is based on the IMDb 2024 movie dataset and focuses on movie titles and their associated storylines. The recommended workflow includes data collection, NLP preprocessing, text vectorization, cosine similarity, and a Streamlit interface. Selenium-based scraping is optional because a dataset is provided for direct use. fileciteturn26file0L10-L30 fileciteturn26file5L291-L304

---

## ✨ Features

- 📝 Accepts a movie storyline, plot, or short description as input
- 🧹 Performs NLP-based text preprocessing
- 🔤 Removes unnecessary characters and stopwords
- 🔢 Converts storylines into numerical TF-IDF vectors
- 📐 Calculates cosine similarity between the input and movie storylines
- 🎯 Returns the top 5 most similar movies
- 📖 Displays the original readable storyline for each recommendation
- 📊 Displays a similarity score for each result
- 🌐 Provides an interactive Streamlit web interface
- 🧩 Uses modular source code for preprocessing, model loading, and recommendation logic
- ✅ Includes validation and testing of model/data alignment

The overall functionality follows the project requirement of accepting a storyline and displaying the top five recommended movies with their names and storylines. fileciteturn26file3L220-L245

---

## 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Data Manipulation | Pandas |
| NLP / Text Processing | Python Regular Expressions, Scikit-learn English stopwords |
| Text Vectorization | TF-IDF |
| Similarity Measure | Cosine Similarity |
| Machine Learning | Scikit-learn |
| Web Application | Streamlit |
| Data Collection | Selenium (optional) |
| Development Environment | VS Code / Python Environment |
| Version Control | Git & GitHub |

The original project specification identifies Selenium, Python, Pandas, Streamlit, NLP, TF-IDF, cosine similarity, machine learning, data cleaning, data analysis, visualization, and UI development as relevant project skills and tools. fileciteturn26file0L10-L20 fileciteturn26file3L237-L245

---

## 📂 Project Structure

```text
IMDb_Movie_Recommendation_System/
│
├── data/
│   ├── IMDBRecSys.csv
│   └── IMDBRecSys_Processed.csv
│
├── models/
│   ├── movie_data.pkl
│   ├── tfidf_matrix.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   ├── 1_EDA.ipynb
│   ├── 2_NLP_Preprocessing.ipynb
│   └── 3_Recommender_System.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── model_loader.py
│   ├── preprocessing.py
│   └── recommender.py
│
├── test/
│   └── test_reco.py
│
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
```

### Folder Responsibilities

**`data/`**  
Contains the raw IMDb dataset and the processed storyline dataset.

**`models/`**  
Contains generated model artifacts used by the application. These files are intentionally excluded from Git through `.gitignore` and can be regenerated locally.

**`notebooks/`**  
Contains the exploratory data analysis, NLP preprocessing, and recommender development workflow.

**`src/`**  
Contains reusable application logic separated by responsibility.

**`test/`**  
Contains automated tests for recommendation-related behaviour.

**`app.py`**  
Streamlit application entry point.

---

## 📊 Dataset

The project uses IMDb 2024 movie data containing:

- **Movie title**
- **Storyline**

The provided project specification describes the dataset as a CSV containing the movie name and storyline for 2024 movies. fileciteturn26file5L291-L304

The current working dataset contains:

- **6,021 movie records**
- **2 raw columns:** `Movie_Title`, `Storyline`
- A processed dataset with:
  - `Movie_Title`
  - `Storyline_Raw`
  - `Storyline_Processed`

Movie title prefixes from the source data are cleaned so that titles are displayed without their numerical index.

---

## 🧹 Data Preprocessing

The preprocessing pipeline is designed to be reusable for both the dataset and the user's input storyline.

### Processing steps

```text
Raw storyline
     ↓
Convert to lowercase
     ↓
Remove non-alphabetic characters
     ↓
Normalize whitespace
     ↓
Tokenize
     ↓
Remove English stopwords
     ↓
Reconstruct processed text
```

The same preprocessing function is reused inside the recommendation engine so that the user's input is transformed consistently with the movie storyline data.

This follows the project specification's requirement to clean storylines, remove stopwords and punctuation/unnecessary characters, tokenize the text, and prepare it for vectorization. fileciteturn26file5L305-L316

---

## 🔢 TF-IDF Vectorization

After preprocessing, the movie storylines are converted into numerical vectors using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

TF-IDF gives greater importance to terms that help distinguish one storyline from another while reducing the influence of very common words.

The trained vectorizer creates the feature space used by the recommendation engine.

Current model statistics:

```text
Movies:             6,021
TF-IDF dimensions:  18,700
Matrix type:        Sparse CSR matrix
```

Using a sparse matrix avoids unnecessarily converting the full feature matrix into a dense representation.

---

## 📐 Recommendation Method

The recommendation engine uses **Cosine Similarity**.

Given a user storyline:

1. The storyline is validated.
2. The same preprocessing pipeline is applied.
3. The text is transformed using the fitted TF-IDF vectorizer.
4. Cosine similarity is calculated between the query vector and all movie vectors.
5. Movies are ranked by similarity score.
6. The top 5 movies are returned.

The project specification explicitly calls for cosine similarity or another suitable machine-learning method to rank movies according to storyline similarity. fileciteturn26file0L21-L30

### Recommendation Formula

Cosine similarity measures the angle between two vectors:

```text
cosine_similarity(A, B) = (A · B) / (||A|| ||B||)
```

A higher score indicates stronger textual similarity between the two storyline representations.

---

## 🧩 Application Architecture

The application follows a modular design:

```text
                 ┌────────────────────┐
                 │      app.py        │
                 │   Streamlit UI     │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │  MovieRecommender  │
                 │ recommender.py     │
                 └─────────┬──────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
   ┌────────────────────┐    ┌────────────────────┐
   │ preprocessing.py    │    │  model_loader.py   │
   │ NLP preprocessing   │    │ Load data/models   │
   └────────────────────┘    └─────────┬──────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │ Model Artifacts │
                              │   TF-IDF + Data │
                              └─────────────────┘
```

This keeps the UI layer separate from the data processing and recommendation logic.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/SICE-Logs/IMDb_Movie_Recommendation_System.git
cd IMDb_Movie_Recommendation_System
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

---

## ⚙️ Generate Model Artifacts

The generated `.pkl` model files are intentionally ignored by Git.

After cloning the repository on a new machine, run the recommender/model-building notebook to recreate:

```text
models/
├── movie_data.pkl
├── tfidf_matrix.pkl
└── tfidf_vectorizer.pkl
```

This approach keeps the repository portable and avoids storing generated model artifacts directly in Git.

---

## ▶️ Run the Streamlit Application

After the model artifacts have been generated:

```bash
python -m streamlit run app.py
```

Then open the local URL shown by Streamlit, normally:

```text
http://localhost:8501
```

### Example Input

```text
A young scientist discovers a mysterious machine that allows
her to travel through time and must prevent a dangerous future
from becoming reality.
```

### Example Output

The application returns:

```text
1. Movie Title
   Similarity: XX.X%

   Storyline...

2. Movie Title
   Similarity: XX.X%

   Storyline...

...

5. Movie Title
   Similarity: XX.X%

   Storyline...
```

The project specification requires the Streamlit interface to allow storyline input and display the top five recommended movies with their names and storylines. fileciteturn26file2L203-L205

---

## 🧪 Testing

The project includes tests for the recommendation engine and its validation logic.

Run the test suite with:

```bash
python -m pytest
```

The recommender validates:

- Input storyline type
- Empty storyline input
- `top_n` type
- Valid recommendation count
- Required dataset columns
- Movie-data and TF-IDF matrix alignment
- Meaningful text remaining after preprocessing

The recommendation engine calculates similarity only between the user's query vector and the stored movie vectors, avoiding the unnecessary construction of a full movie-by-movie similarity matrix. This follows the project guideline to use cosine similarity efficiently and avoid performance bottlenecks. fileciteturn26file4L263-L269

---

## 📓 Notebooks

### `1_EDA.ipynb`

Explores:

- Dataset structure
- Missing values
- Duplicate checks
- Storyline length characteristics
- Text-related dataset observations
- Data quality findings

### `2_NLP_Preprocessing.ipynb`

Handles:

- Movie title cleaning
- Storyline normalization
- Tokenization
- Stopword removal
- Processed dataset generation

### `3_Recommender_System.ipynb`

Handles:

- TF-IDF vectorization
- Feature-space creation
- Similarity calculation
- Recommendation experiments
- Model artifact generation

---

## 💡 Why Content-Based Recommendation?

This system does not require user ratings, user profiles, or historical interaction data.

Instead, it relies on the textual content of the movie storyline.

That makes it useful when a user already knows **what kind of story they want to watch**, even when they do not know the movie title.

### Example

```text
User knows:
"I want something about astronauts surviving on another planet."

System:
→ converts the description into a TF-IDF vector
→ compares it with movie storylines
→ returns the five closest matches
```

---

## ✅ Strengths

- Simple and interpretable recommendation approach
- Works without user rating history
- Uses a lightweight classical NLP/ML pipeline
- Fast query-time similarity calculation
- Modular project architecture
- Reusable preprocessing pipeline
- Easy to run locally with Streamlit
- Suitable for an educational and portfolio project

---

## ⚠️ Limitations

### 1. Lexical Similarity

TF-IDF primarily captures word-level statistical relationships.

Two storylines with similar meanings but very different vocabulary may receive a lower similarity score than expected.

### 2. No User Personalization

Recommendations are based on storyline similarity only. The system does not currently consider:

- User ratings
- Watch history
- Genre preference
- Actors
- Directors
- Popularity

### 3. Dataset Scope

The system is limited to the movie records present in the dataset.

### 4. Similarity Is Not Quality

A higher similarity score does **not** mean that a movie is better. It only indicates stronger textual similarity between the user's input and the stored storyline representation.

---

## 🔮 Future Enhancements

Possible future improvements include:

- Semantic embeddings using transformer-based models
- Genre-aware recommendation
- Rating and popularity integration
- Hybrid content + collaborative filtering
- Movie posters and metadata integration
- Recommendation explanations
- Search and filtering options
- Persistent user preference profiles
- Cloud deployment
- Automated model/data refresh pipelines
- More robust evaluation of recommendation relevance and diversity

---

## 📌 Project Requirements Mapping

| Requirement | Implementation |
|---|---|
| IMDb movie data | IMDb 2024 dataset |
| Movie name + storyline | ✅ |
| NLP preprocessing | ✅ |
| Stopword removal | ✅ |
| Tokenization | ✅ |
| TF-IDF / Count Vectorizer | ✅ TF-IDF |
| Cosine Similarity | ✅ |
| Top 5 recommendations | ✅ |
| Streamlit application | ✅ |
| Modular Python code | ✅ |
| Model/data validation | ✅ |
| Documentation | ✅ |
| GitHub repository | ✅ |

The implementation aligns with the supplied project specification, which lists the CSV dataset, Python scripts, NLP/recommendation processing, Streamlit application, and project documentation as the expected deliverables. fileciteturn26file2L178-L188

---

## 🔗 Repository

**GitHub:**  
https://github.com/SICE-Logs/IMDb_Movie_Recommendation_System

---

## 👨‍💻 Author

**Logajit S**

B.Tech Artificial Intelligence and Data Science

GitHub: https://github.com/SICE-Logs

---

## 📄 Project Context

This repository was developed as part of a practical machine-learning project focused on text-based recommendation systems using IMDb movie storylines.

The project emphasizes:

- Data preprocessing
- NLP
- Feature engineering
- Similarity-based recommendation
- Modular Python development
- Streamlit deployment
- Reproducibility
- Git/GitHub workflow

---

## ⭐ Acknowledgement

The project structure and technical workflow are based on the supplied IMDb Movie Recommendation System project brief, which specifies storyline-based recommendation using NLP, TF-IDF/Count Vectorizer, cosine similarity, and Streamlit. fileciteturn26file0L21-L30
