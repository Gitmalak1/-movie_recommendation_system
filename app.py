from pathlib import Path

import pandas as pd
import streamlit as st

from engine import SemanticIndex

DATA_PATH = Path(__file__).parent / "data" / "items.csv"

st.set_page_config(
    page_title="Semantic Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎬 Semantic Movie Recommender")
st.markdown(
    "Find movies by **meaning**, discover similar titles, or build recommendations "
    "from your favourites — powered by sentence embeddings."
)


def load_data(uploaded_file):
    source = pd.read_csv(uploaded_file) if uploaded_file else pd.read_csv(DATA_PATH)
    try:
        return SemanticIndex._validate_dataframe(source)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()


with st.sidebar:
    st.header("⚙️ Settings")
    uploaded = st.file_uploader(
        "Upload your own CSV", type="csv", help="Required columns: title, genre, description"
    )
    df = load_data(uploaded)

    available_genres = sorted(g for g in df["genre"].unique() if g)
    genres = st.multiselect("Genre filter", available_genres)
    result_count = st.slider("Number of results", 1, min(15, len(df)), min(5, len(df)))

    st.divider()
    st.caption(f"{len(df)} movies · {df['genre'].nunique()} genres")


@st.cache_resource(show_spinner="Building recommendation index...")
def get_index(df_json: str) -> SemanticIndex:
    return SemanticIndex(pd.read_json(df_json))


index = get_index(df.to_json())

with st.sidebar:
    backend_label = "Semantic embeddings" if index.backend == "sentence-transformers" else "TF-IDF fallback"
    st.success(f"Search engine: {backend_label}")


def show_results(results: pd.DataFrame) -> None:
    if results.empty:
        st.warning("No movies match the selected filters.")
        return

    for _, movie in results.iterrows():
        with st.container(border=True):
            left, right = st.columns([4, 1])
            with left:
                st.subheader(movie["title"])
                st.caption(movie["genre"])
                st.write(movie["description"])
            with right:
                st.metric("Match", f"{movie['score']:.2f}")


st.caption("Scores are cosine-similarity values; higher means more semantically similar.")

search_tab, similar_tab, profile_tab = st.tabs(
    ["🔎 Search by description", "🧲 Similar movies", "❤️ My taste"]
)

with search_tab:
    query = st.text_input(
        "What kind of movie are you looking for?",
        placeholder="A lonely robot discovers friendship and love",
    )
    if query.strip():
        try:
            show_results(index.search(query, result_count, genres))
        except ValueError as exc:
            st.warning(str(exc))
    else:
        st.info("Try: *mind-bending dreams*, *survival on another planet*, or *heartwarming family adventure*.")

with similar_tab:
    selected_title = st.selectbox("Choose a movie", df["title"].tolist(), key="similar_title")
    show_results(index.similar(selected_title, result_count, genres))

with profile_tab:
    liked = st.multiselect(
        "Choose movies you like",
        df["title"].tolist(),
        placeholder="Select two or more favourites...",
    )
    if liked:
        st.caption(f"Building a taste profile from {len(liked)} favourite(s).")
        show_results(index.recommend_from_liked(liked, result_count, genres))
    else:
        st.info("Select a few favourites and the recommender will find movies with a similar semantic profile.")

st.divider()
st.caption("Built with Python · Sentence Transformers · scikit-learn · Streamlit")
