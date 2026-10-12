import pandas as pd
import pytest

from engine import SemanticIndex


@pytest.fixture(scope="module")
def index():
    return SemanticIndex(pd.read_csv("data/items.csv"), model_name="invalid-model-for-test")


def test_dataset_is_loaded(index):
    assert len(index.df) == 33
    assert set(index.df.columns) == {"title", "genre", "description"}


def test_search_returns_requested_number(index):
    results = index.search("space travel", 3)
    assert len(results) == 3
    assert results["score"].between(-1, 1).all()


def test_similar_excludes_self(index):
    results = index.similar("The Matrix", 5)
    assert "The Matrix" not in results.title.values


def test_title_lookup_is_case_insensitive(index):
    results = index.similar("the matrix", 3)
    assert len(results) == 3


def test_genre_filter(index):
    results = index.search("love story", 5, ["Romance"])
    assert set(results.genre) == {"Romance"}


def test_liked_excludes_liked_movies(index):
    results = index.recommend_from_liked(["Toy Story"], 5)
    assert "Toy Story" not in results.title.values


def test_unknown_title_has_clear_error(index):
    with pytest.raises(ValueError, match="Movie not found"):
        index.similar("Not A Real Movie")


def test_empty_query_has_clear_error(index):
    with pytest.raises(ValueError, match="cannot be empty"):
        index.search("   ")


def test_invalid_dataset_is_rejected():
    with pytest.raises(ValueError, match="missing required columns"):
        SemanticIndex(pd.DataFrame({"title": ["Only Title"]}), model_name="invalid")
