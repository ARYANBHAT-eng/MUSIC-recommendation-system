import os
import pickle
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import streamlit as st

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
except ImportError:  # pragma: no cover - runtime environment specific
    spotipy = None
    SpotifyClientCredentials = None

DEFAULT_COVER_URL = "https://i.postimg.cc/0QNxYz4V/social.png"
RECOMMENDATION_COUNT = 5
MUSIC_PICKLE_PATH = Path("df.pkl")
SIMILARITY_PICKLE_PATH = Path("similarity.pkl")


def load_pickle_file(file_path: Path) -> Optional[Any]:
    """Load a pickle file safely and return None if unavailable or invalid."""
    if not file_path.exists():
        return None

    try:
        with file_path.open("rb") as file_pointer:
            return pickle.load(file_pointer)
    except (OSError, pickle.UnpicklingError):
        return None


def create_spotify_client() -> Optional[Any]:
    """Create Spotify client from environment variables if available."""
    if spotipy is None or SpotifyClientCredentials is None:
        return None

    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

    if not client_id or not client_secret:
        return None

    try:
        credentials = SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret,
        )
        return spotipy.Spotify(client_credentials_manager=credentials)
    except Exception:
        return None


def fetch_album_cover_url(
    spotify_client: Optional[Any],
    song_name: str,
    artist_name: str,
) -> str:
    """Fetch album cover from Spotify, or fall back to default image."""
    if spotify_client is None:
        return DEFAULT_COVER_URL

    search_query = f"track:{song_name} artist:{artist_name}"
    try:
        results = spotify_client.search(q=search_query, type="track", limit=1)
    except Exception:
        return DEFAULT_COVER_URL

    tracks = results.get("tracks", {}).get("items", []) if results else []
    if not tracks:
        return DEFAULT_COVER_URL

    images = tracks[0].get("album", {}).get("images", [])
    if not images:
        return DEFAULT_COVER_URL

    return images[0].get("url", DEFAULT_COVER_URL)


def generate_recommendations(
    selected_song: str,
    music_data: Any,
    similarity_matrix: Sequence[Sequence[float]],
    spotify_client: Optional[Any],
    top_k: int = RECOMMENDATION_COUNT,
) -> List[Tuple[str, str]]:
    """Return recommended songs with album covers."""
    matching_indexes = music_data[music_data["song"] == selected_song].index
    if len(matching_indexes) == 0:
        return []

    selected_index = matching_indexes[0]
    similarity_scores = list(enumerate(similarity_matrix[selected_index]))
    ranked_scores = sorted(
        similarity_scores,
        key=lambda item: item[1],
        reverse=True,
    )

    recommendations: List[Tuple[str, str]] = []
    for song_index, _ in ranked_scores[1 : top_k + 1]:
        song_title = music_data.iloc[song_index].get("song", "Unknown Song")
        artist_name = music_data.iloc[song_index].get("artist", "Unknown Artist")
        cover_url = fetch_album_cover_url(spotify_client, song_title, artist_name)
        recommendations.append((song_title, cover_url))

    return recommendations


def render_recommendations(recommendations: List[Tuple[str, str]]) -> None:
    """Render recommendation cards in responsive Streamlit columns."""
    if not recommendations:
        st.info("No recommendations were found for the selected song.")
        return

    column_count = min(RECOMMENDATION_COUNT, len(recommendations))
    columns = st.columns(column_count)

    for column, (song_title, cover_url) in zip(columns, recommendations):
        with column:
            st.caption(song_title)
            st.image(cover_url, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Music Recommender", page_icon="🎵", layout="wide")
    st.title("🎵 Music Recommender System")
    st.write("Select a song to get similar music recommendations.")

    spotify_client = create_spotify_client()

    if spotify_client is None:
        st.warning(
            "Spotify API credentials are missing or invalid. "
            "Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET for album artwork."
        )

    music_data = load_pickle_file(MUSIC_PICKLE_PATH)
    similarity_matrix = load_pickle_file(SIMILARITY_PICKLE_PATH)

    if music_data is None or similarity_matrix is None:
        st.error(
            "Recommendation assets are not available yet. "
            "Please add `df.pkl` and `similarity.pkl` to enable recommendations."
        )
        st.info(
            "Limited mode is active: the UI is available, but recommendations are "
            "disabled until model artifacts are added."
        )
        # TODO: Replace local pickle loading with a versioned model registry.
        # TODO: Integrate the training/data pipeline output artifacts here.
        st.stop()

    if "song" not in music_data.columns:
        st.error("`df.pkl` is missing the required `song` column.")
        st.stop()

    if "artist" not in music_data.columns:
        st.warning(
            "`df.pkl` has no `artist` column. Fallback artist names will be used."
        )
        music_data["artist"] = "Unknown Artist"

    song_options = music_data["song"].dropna().astype(str).unique().tolist()
    if not song_options:
        st.error("No songs are available in `df.pkl`.")
        st.stop()

    selected_song = st.selectbox(
        "Choose a song",
        options=song_options,
        index=0,
        placeholder="Start typing a song name...",
    )

    if st.button("Recommend Songs", use_container_width=True):
        with st.spinner("Finding similar songs..."):
            recommendations = generate_recommendations(
                selected_song=selected_song,
                music_data=music_data,
                similarity_matrix=similarity_matrix,
                spotify_client=spotify_client,
            )
        render_recommendations(recommendations)


if __name__ == "__main__":
    main()





