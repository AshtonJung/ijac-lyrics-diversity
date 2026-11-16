"""
01_inventory.py
Reproduces the data inventory reported in outputs/data_inventory.md.
Reads only from data/raw/ (never touches the original external project folders).
"""
import pandas as pd
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def main():
    matched = pd.read_parquet(RAW / "mgd_plus_study2_source" / "us_lyrics_matched.parquet")
    genre = pd.read_parquet(RAW / "mgd_plus_study2_source" / "track_market_year_lyrics.parquet")
    study1 = pd.read_parquet(RAW / "mgd_plus_study2_source" / "us_study1_lyric_features.parquet")

    print("us_lyrics_matched.parquet:", matched.shape)
    print("track_market_year_lyrics.parquet:", genre.shape)
    print("us_study1_lyric_features.parquet:", study1.shape)

    print("\nlyric_source_status counts:")
    print(matched["lyric_source_status"].value_counts())

    print("\nmatch_confidence counts:")
    print(matched["match_confidence"].value_counts())

    genre_us = genre[genre["market"] == "us"][["song_id", "main_genre"]].drop_duplicates("song_id")
    joined = matched.merge(genre_us, on="song_id", how="left")
    print("\nrows with resolvable genre:", joined["main_genre"].notna().sum(), "/", len(joined))


if __name__ == "__main__":
    main()
