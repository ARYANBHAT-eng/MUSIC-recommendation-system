import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MAX_ROWS = 20000


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["song", "artist", "text"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in dataset: {missing_columns}")

    processed_df = df[required_columns].dropna().copy()
    processed_df["text"] = processed_df["text"].astype(str).str.lower()
    processed_df = processed_df.drop_duplicates(subset=["song", "artist"])
    processed_df = processed_df[required_columns].reset_index(drop=True)
    return processed_df


def build_similarity(df: pd.DataFrame) -> np.ndarray:
    if len(df) > MAX_ROWS:
        raise ValueError(
            f"Dataset too large ({len(df)} rows). "
            "Full similarity matrix will exceed memory limits."
        )

    if df["text"].str.strip().eq("").all():
        raise ValueError(
            "All text data is empty after preprocessing. Cannot build vocabulary."
        )

    vectorizer = CountVectorizer(max_features=5000, stop_words="english")
    try:
        vectors = vectorizer.fit_transform(df["text"]).toarray()
    except ValueError as error:
        raise ValueError(
            "Vectorization failed: likely due to empty vocabulary after preprocessing"
        ) from error

    similarity = cosine_similarity(vectors)
    return similarity


def validate(df: pd.DataFrame, similarity) -> None:
    if df.empty:
        raise ValueError("Dataframe is empty after preprocessing")

    required_columns = {"song", "artist", "text"}
    if not required_columns.issubset(df.columns):
        missing_columns = sorted(required_columns - set(df.columns))
        raise ValueError(f"Missing required columns: {missing_columns}")

    expected_shape = (len(df), len(df))
    if similarity.shape != expected_shape:
        raise ValueError(
            f"Similarity matrix shape mismatch: expected {expected_shape}, got {similarity.shape}"
        )

    print(f"Dataset size: {len(df)}")
    print(f"Similarity shape: {similarity.shape}")


def save_artifacts(df: pd.DataFrame, similarity, output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)

    df_path = output_path / "df.pkl"
    similarity_path = output_path / "similarity.pkl"

    with df_path.open("wb") as df_file:
        pickle.dump(df, df_file, protocol=pickle.HIGHEST_PROTOCOL)

    with similarity_path.open("wb") as similarity_file:
        pickle.dump(similarity, similarity_file, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    start = time.time()
    try:
        parser = argparse.ArgumentParser(
            description="Train music recommendation artifacts from CSV data."
        )
        parser.add_argument(
            "--input",
            type=Path,
            default=Path("spotify_millsongdata.csv"),
            help="Path to input CSV dataset.",
        )
        parser.add_argument(
            "--output",
            type=Path,
            default=Path("."),
            help="Directory to save df.pkl and similarity.pkl.",
        )
        args = parser.parse_args()

        df = load_data(args.input)
        df = preprocess(df)
        similarity = build_similarity(df)
        validate(df, similarity)
        similarity = similarity.astype("float32")
        save_artifacts(df, similarity, args.output)
        print("Training pipeline completed successfully ✅")
        print(f"Completed in {time.time() - start:.2f}s")
    except Exception as error:
        print(f"Error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
