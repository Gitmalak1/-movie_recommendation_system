# 🎬 Semantic Movie Recommender

A production-style **semantic movie search and recommendation app** built with Python and Streamlit. Instead of matching only keywords, it represents movie descriptions as vectors and retrieves titles based on their **meaning**.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red?logo=streamlit)
![Tests](https://img.shields.io/badge/tests-pytest-green?logo=pytest)

## ✨ Features

- 🔎 **Semantic search** — describe the kind of movie you want in natural language.
- 🧲 **More like this** — find movies similar to a selected title.
- ❤️ **Taste profile** — combine several favourites into one recommendation profile.
- 🎭 **Genre filtering** — narrow recommendations without changing the semantic ranking.
- 📁 **Custom datasets** — upload any CSV containing `title`, `genre`, and `description`.
- ⚡ **Embedding cache** — avoid recomputing embeddings between app launches.
- 🛟 **Offline fallback** — automatically uses TF-IDF if the sentence-transformer model cannot be loaded.
- 🧪 **Automated tests** — input validation, filtering, title lookup, and recommendation behaviour are covered by pytest.
- 🔄 **CI** — GitHub Actions runs the test suite on pushes and pull requests.

## 🧠 How it works

The default model is [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), a compact sentence-transformer model.

Each movie is represented using:

```text
Title + Genre + Description
```

The text is converted into an embedding vector. Search queries and user taste profiles are embedded in the same vector space, and **cosine similarity** is used to rank candidates.

For a favourite-based recommendation, the engine calculates the centroid of the selected movie embeddings:

```text
user_profile = mean(embedding(movie_1), ..., embedding(movie_n))
```

The selected favourites are then excluded from the result list.

If the transformer model is unavailable, the engine falls back to a TF-IDF representation with unigrams and bigrams.

## 🏗️ Project structure

```text
semantic-search-recommender/
├── .github/workflows/ci.yml   # Continuous integration
├── data/items.csv             # Small demo dataset
├── app.py                     # Streamlit interface
├── engine.py                  # Search and recommendation engine
├── test_engine.py             # Automated tests
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

## 🚀 Run locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/semantic-search-recommender.git
cd semantic-search-recommender
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the app

```bash
streamlit run app.py
```

The first run may download the sentence-transformer model. Subsequent runs reuse cached embeddings for the same dataset and model.

### 5. Run tests

```bash
pytest -q
```

## 📄 Custom dataset format

Upload a CSV with these required columns:

| Column | Description |
|---|---|
| `title` | Unique movie/item title |
| `genre` | Genre or category |
| `description` | Text used for semantic matching |

Example:

```csv
title,genre,description
Example Movie,Sci-Fi,A scientist discovers a portal to another dimension.
Another Movie,Drama,A family rebuilds its life after moving to a new city.
```

## ☁️ Deployment

The app can be deployed on Streamlit Community Cloud:

1. Push the project to GitHub.
2. Create a new app on Streamlit Community Cloud.
3. Select the repository and `main` branch.
4. Set the main file to `app.py`.
5. Deploy.

The application does not require API keys or application secrets.

## 🔬 Limitations

- The demo dataset is intentionally small, so recommendations are limited by the available movies.
- Semantic similarity is not the same as personal preference; two movies can be textually similar while appealing to different audiences.
- The default model is English-focused, so multilingual descriptions may perform differently.
- The TF-IDF fallback is keyword-based and will generally capture less semantic nuance than the transformer model.

## 🛠️ Future improvements

- Add movie posters and metadata from a public movie API.
- Add evaluation metrics such as Recall@K and NDCG@K.
- Support hybrid ranking that combines semantic similarity with genre and popularity signals.
- Add persistent vector storage for larger datasets.
- Add user accounts and persistent preference profiles.

## 📜 License

Add a license before publishing if you want others to reuse the code. MIT is a common choice for portfolio projects.
