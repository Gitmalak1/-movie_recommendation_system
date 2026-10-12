"""Recommendation engine for semantic movie search.

The engine uses sentence-transformers when available and automatically falls back
on TF-IDF when the embedding model cannot be loaded (for example, offline).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = Path(".cache")
REQUIRED_COLUMNS = {"title", "genre", "description"}


class SemanticIndex:
    """Build a searchable semantic index over a movie DataFrame."""

    def __init__(self, df: pd.DataFrame, model_name: str = MODEL_NAME) -> None:
        self.df = self._validate_dataframe(df)
        self.model_name = model_name
        self.texts = self._build_texts(self.df)
        self.backend = "sentence-transformers"

        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            self.embeddings = self._load_or_create_embeddings()
        except Exception as exc:  # pragma: no cover - depends on environment/network
            self.backend = "tfidf"
            self.model_error = str(exc)
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.model = TfidfVectorizer(
                stop_words="english", ngram_range=(1, 2), sublinear_tf=True
            )
            matrix = self.model.fit_transform(self.texts)
            self.embeddings = self._normalize(matrix.toarray())

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                "Dataset is missing required columns: " + ", ".join(sorted(missing))
            )

        clean = df.loc[:, ["title", "genre", "description"]].copy()
        for column in REQUIRED_COLUMNS:
            clean[column] = clean[column].fillna("").astype(str).str.strip()

        clean = clean[clean["title"] != ""].drop_duplicates(subset="title")
        if clean.empty:
            raise ValueError("Dataset must contain at least one movie with a title.")

        return clean.reset_index(drop=True)

    @staticmethod
    def _build_texts(df: pd.DataFrame) -> list[str]:
        return (
            df["title"]
            + ". Genre: "
            + df["genre"]
            + ". Description: "
            + df["description"]
        ).tolist()

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return values / np.maximum(norms, 1e-12)

    def _cache_path(self) -> Path:
        fingerprint = hashlib.sha256(
            (self.model_name + "\n" + "\n".join(self.texts)).encode("utf-8")
        ).hexdigest()[:20]
        return CACHE_DIR / f"embeddings_{fingerprint}.npy"

    def _load_or_create_embeddings(self) -> np.ndarray:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = self._cache_path()

        if cache_file.exists():
            cached = np.load(cache_file)
            if cached.shape == (len(self.texts), cached.shape[1]):
                return cached.astype(np.float32)

        embeddings = self.model.encode(
            self.texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        np.save(cache_file, embeddings)
        return embeddings

    def _encode(self, text: str) -> np.ndarray:
        text = str(text).strip()
        if not text:
            raise ValueError("Search text cannot be empty.")

        if self.backend == "tfidf":
            vector = self.model.transform([text]).toarray()
            return self._normalize(vector)[0]

        return np.asarray(
            self.model.encode(
                [text], convert_to_numpy=True, normalize_embeddings=True
            )[0],
            dtype=np.float32,
        )

    def _resolve_title(self, title: str) -> int:
        matches = np.flatnonzero(self.df["title"].str.casefold().eq(title.strip().casefold()))
        if len(matches) == 0:
            raise ValueError(f"Movie not found: {title}")
        return int(matches[0])

    def _rank(
        self,
        scores: np.ndarray,
        k: int,
        genres: Iterable[str] | None = None,
        exclude: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        if k < 1:
            raise ValueError("k must be at least 1")

        scores = np.asarray(scores, dtype=np.float32).copy()
        if exclude:
            scores[list(exclude)] = -np.inf

        selected_genres = {str(g) for g in (genres or []) if str(g).strip()}
        if selected_genres:
            mask = ~self.df["genre"].isin(selected_genres).to_numpy()
            scores[mask] = -np.inf

        valid = np.flatnonzero(np.isfinite(scores))
        if len(valid) == 0:
            return self.df.iloc[0:0].assign(score=pd.Series(dtype=float))

        k = min(k, len(valid))
        top = valid[np.argsort(scores[valid])[::-1][:k]]
        return self.df.iloc[top].assign(score=scores[top]).reset_index(drop=True)

    def search(
        self, query: str, k: int = 5, genres: Iterable[str] | None = None
    ) -> pd.DataFrame:
        """Return movies semantically similar to a natural-language query."""
        query_vector = self._encode(query)
        scores = self.embeddings @ query_vector
        return self._rank(scores, k, genres)

    def similar(
        self, title: str, k: int = 5, genres: Iterable[str] | None = None
    ) -> pd.DataFrame:
        """Return movies similar to a selected movie, excluding the movie itself."""
        index = self._resolve_title(title)
        scores = self.embeddings @ self.embeddings[index]
        return self._rank(scores, k, genres, exclude=[index])

    def recommend_from_liked(
        self,
        titles: Sequence[str],
        k: int = 5,
        genres: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Recommend movies from the centroid of the user's liked movies."""
        if not titles:
            raise ValueError("Select at least one favourite movie.")

        indices = [self._resolve_title(title) for title in titles]
        profile = self._normalize(
            self.embeddings[indices].mean(axis=0, keepdims=True)
        )[0]
        scores = self.embeddings @ profile
        return self._rank(scores, k, genres, exclude=indices)
