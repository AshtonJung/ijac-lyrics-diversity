"""
02_clean_human_lyrics.py

Implements the inclusion/exclusion rules locked in outputs/preanalysis_decisions.md
(Section 1). Reads only from data/raw/, writes only to data/processed/ and
outputs/exclusions.csv. Never modifies data/raw/ in place.

Stages (applied in this fixed order, matching the pre-analysis decision record):
  0. duplicate_song_id_row  : the raw source file itself contains literal duplicate rows
                             for the same Spotify song_id (295 rows / 280 unique song_id,
                             discovered during Phase 3 QA -- not caught at inventory time).
                             Dedup first on song_id (keep highest match_confidence, then
                             highest duration_agrees, then first) before any other stage.
  1. manual_review_status : drop lyric_source_status != "matched"
  2. genre_unresolved      : drop rows where main_genre does not join via song_id
  3. empty_lyrics          : drop null/empty lyric text
  4. duplicate_lyrics_text : dedup exact-duplicate lyric text, keep highest
                             match_confidence then highest duration_agrees
  5. unmapped_genre_bucket : drop rows whose main_genre does not map to one of the
                             six controlled broad-genre buckets
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

CONF_RANK = {"high": 3, "medium": 2, "low": 1}

GENRE_RULES = [
    (("hip hop", "rap", "trap", "drill"), "hip-hop/rap"),
    (("country",), "country"),
    (("r&b", "soul"), "r&b/soul"),
    (("rock", "punk", "metal", "grunge"), "rock/alternative"),
    (("edm", "dance", "electro", "house", "techno", "trance"), "electronic/dance"),
]


def bucket_genre(raw_genre: str) -> str:
    g = str(raw_genre).lower()
    if g in ("boy band", "girl group"):
        return "pop"
    for keywords, label in GENRE_RULES:
        if any(k in g for k in keywords):
            return label
    if "pop" in g:
        return "pop"
    return "other"


def normalize_text(text: str) -> str:
    """Lowercase, strip bracketed annotations e.g. [Chorus], collapse whitespace.
    Keeps words intact (no stemming) for lexical-diversity metrics."""
    t = re.sub(r"\[[^\]]*\]", " ", text)  # remove [Verse]/[Chorus] tags
    t = t.lower()
    t = re.sub(r"[^a-z0-9'\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    matched = pd.read_parquet(RAW / "mgd_plus_study2_source" / "us_lyrics_matched.parquet")
    genre = pd.read_parquet(RAW / "mgd_plus_study2_source" / "track_market_year_lyrics.parquet")
    genre_us = genre[genre["market"] == "us"][["song_id", "main_genre"]].drop_duplicates("song_id")

    exclusions = []

    # Stage 0: duplicate_song_id_row (raw-source artifact: same song_id repeated).
    # Priority when picking which duplicate row to keep: "matched" status first
    # (status is decided in Stage 1 below, but we must not let a manual_review
    # duplicate crowd out a matched duplicate of the same song_id here), then
    # higher match_confidence, then higher duration_agrees, then row order.
    matched = matched.copy().reset_index(drop=True)
    matched["_status_rank"] = (matched["lyric_source_status"] == "matched").astype(int)
    matched["_conf_rank"] = matched["match_confidence"].map(CONF_RANK).fillna(0)
    matched["_orig_order"] = matched.index
    matched_sorted = matched.sort_values(
        ["_status_rank", "_conf_rank", "duration_agrees", "_orig_order"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    row_dup_mask = matched_sorted.duplicated(subset="song_id", keep="first")
    for _, r in matched_sorted[row_dup_mask].iterrows():
        exclusions.append((r["song_id"], "duplicate_song_id_row", "00_dedup_song_id",
                            "raw source file contained >1 row for this song_id"))
    matched = matched_sorted[~row_dup_mask].drop(columns=["_status_rank", "_conf_rank", "_orig_order"])

    df = matched.merge(genre_us, on="song_id", how="left")
    df["stage"] = "retained"
    df["exclusion_reason"] = ""

    # Stage 1: manual_review_status
    mask = df["lyric_source_status"] != "matched"
    for _, r in df[mask].iterrows():
        exclusions.append((r["song_id"], "manual_review_status", "01_status_filter",
                            f"lyric_source_status={r['lyric_source_status']}"))
    df = df[~mask].copy()

    # Stage 2: genre_unresolved
    mask = df["main_genre"].isna()
    for _, r in df[mask].iterrows():
        exclusions.append((r["song_id"], "genre_unresolved", "02_genre_join", "no main_genre via song_id join"))
    df = df[~mask].copy()

    # Stage 3: empty_lyrics
    mask = df["lyrics"].isna() | (df["lyrics"].astype(str).str.strip() == "")
    for _, r in df[mask].iterrows():
        exclusions.append((r["song_id"], "empty_lyrics", "03_empty_lyrics", "null/empty lyric text"))
    df = df[~mask].copy()

    # Stage 4: duplicate_lyrics_text
    df["conf_rank"] = df["match_confidence"].map(CONF_RANK).fillna(0)
    df = df.sort_values(["conf_rank", "duration_agrees"], ascending=False)
    dup_mask = df.duplicated(subset="lyrics", keep="first")
    kept_by_text = df[~dup_mask].set_index("lyrics")["song_id"]
    for _, r in df[dup_mask].iterrows():
        kept_id = kept_by_text.get(r["lyrics"], "unknown")
        exclusions.append((r["song_id"], "duplicate_lyrics_text", "04_dedup_exact_text",
                            f"exact-duplicate of retained song_id={kept_id}"))
    df = df[~dup_mask].copy()

    # Broad genre bucketing
    df["genre_broad"] = df["main_genre"].apply(bucket_genre)

    # Stage 5: unmapped_genre_bucket
    mask = df["genre_broad"] == "other"
    for _, r in df[mask].iterrows():
        exclusions.append((r["song_id"], "unmapped_genre_bucket", "05_genre_bucket_map",
                            f"main_genre={r['main_genre']} did not map to a controlled bucket"))
    df = df[~mask].copy()

    # Build analysis text (preserve structure/words) and normalized text (for lexical metrics)
    df["analysis_text"] = df["lyrics"].astype(str)
    df["normalized_text"] = df["analysis_text"].apply(normalize_text)
    df["word_count"] = df["analysis_text"].apply(lambda x: len(x.split()))
    df["word_count_normalized"] = df["normalized_text"].apply(lambda x: len(x.split()) if x else 0)

    keep_cols = [
        "song_id", "song_name", "artist_name", "market", "main_genre", "genre_broad",
        "match_confidence", "match_score", "duration_s", "word_count",
        "word_count_normalized", "analysis_text", "normalized_text",
    ]
    clean = df[keep_cols].reset_index(drop=True)
    clean.to_parquet(PROCESSED / "human_lyrics_clean.parquet", index=False)

    exclusions_df = pd.DataFrame(
        exclusions, columns=["song_id", "reason", "stage", "detail"]
    )
    exclusions_df.to_csv(OUTPUTS / "exclusions.csv", index=False)

    print(f"Final clean sample: {len(clean)} songs")
    print(clean["genre_broad"].value_counts())
    print(f"\nTotal exclusions logged: {len(exclusions_df)}")
    print(exclusions_df["reason"].value_counts())


if __name__ == "__main__":
    main()
