"""
06_features.py

Feature engineering (Section 12 of CLAUDE.md), Primary Outcomes A and B.

Currently runs on the HUMAN corpus only (data/processed/human_lyrics_labeled.parquet),
because AI generations do not exist yet (Phase 5 blocked on ANTHROPIC_API_KEY). This
script is written so that once data/generated/ai_lyrics_raw.jsonl exists, it can be
re-run against the combined human+AI corpus with no logic changes -- see
`load_corpus()`, which already checks for and would include the AI file if present.

Outcome A -- semantic dispersion:
  Model: sentence-transformers/all-mpnet-base-v2 (768-dim), chosen because it is the
  general-purpose default recommended by the sentence-transformers library docs, not
  fine-tuned or selected after looking at outcome differences (there are none yet --
  AI data doesn't exist).
  Long lyrics are chunked into ~120-word segments (model max sequence length is smaller
  than the longest lyrics in this corpus), each chunk embedded, and the song-level
  embedding is the mean of chunk embeddings, L2-normalized. This aggregation choice is
  logged here explicitly per Section 24 (robustness: "different reasonable embedding
  aggregation choice" is a named robustness check for later, e.g. embed-first-512-tokens
  only, or embed-and-concatenate-verse-chorus).
  Dispersion = cosine distance from each song's embedding to its GENRE centroid
  (distance-to-centroid, PERMDISP-style, per Section 12). No pairwise lyric-to-lyric
  distances are used as independent observations anywhere in this script.

Outcome B -- lexical diversity:
  MTLD (primary) and HD-D (named robustness alternative) via the `lexical_diversity`
  package, computed on normalized_text (lowercased, punctuation-stripped, but not
  stemmed). Raw TTR also reported, descriptive only.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
GENERATED = ROOT / "data" / "generated"
OUTPUTS = ROOT / "outputs"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CHUNK_WORDS = 120


PRIMARY_GENERATION_METHOD = "anthropic_api"  # see outputs/preanalysis_decisions_v5.md


def load_corpus():
    """Load human corpus; include AI generations if they exist yet. The primary AI arm
    is restricted to generation_method == "anthropic_api" (real Messages API calls,
    outputs/preanalysis_decisions_v5.md) -- the two earlier pilot cohorts
    (claude_code_cli_headless, claude_in_session_authored; 56 rows total) are real data
    but are deliberately excluded from the confirmatory feature set and discussed
    separately in the manuscript as a methods/transparency case study, not blended into
    the primary result."""
    human = pd.read_parquet(PROCESSED / "human_lyrics_labeled.parquet")
    human = human.rename(columns={"analysis_text": "text", "normalized_text": "norm_text"})
    human["source"] = "human"
    human["row_id"] = "H_" + human["song_id"].astype(str)
    human["prompt_id"] = "P" + human["song_id"].astype(str)
    human["condition"] = "human"

    ai_path = GENERATED / "ai_lyrics_raw.jsonl"
    if ai_path.exists():
        rows = [json.loads(l) for l in open(ai_path) if l.strip()]
        ai = pd.DataFrame(rows)
        n_total = len(ai)
        ai = ai[ai["generation_status"] == "ok"].copy()
        n_ok = len(ai)
        ai = ai[ai["generation_method"] == PRIMARY_GENERATION_METHOD].copy()
        print(f"Loaded {n_total} AI generation rows total ({n_ok} status=ok); "
              f"{len(ai)} are primary-cohort ({PRIMARY_GENERATION_METHOD}) rows used below.")
        ai["text"] = ai["generation_text"]
        ai["source"] = "ai"
        ai["row_id"] = ai["prompt_id"] + "_" + ai["condition"]
        ai["word_count"] = ai["actual_word_count"]
        combined_cols = ["row_id", "source", "genre", "text", "prompt_id", "condition", "word_count"]
        human_small = human.rename(columns={"genre_broad": "genre"})[combined_cols]
        ai_small = ai[combined_cols]
        return pd.concat([human_small, ai_small], ignore_index=True), human
    else:
        print("No AI generations found yet (data/generated/ai_lyrics_raw.jsonl absent). "
              "Running on human corpus only -- semantic/lexical features are still valid "
              "and reusable once AI data is added; no primary human-vs-AI comparison yet.")
        cols = ["row_id", "source", "genre", "text", "prompt_id", "condition", "word_count"]
        human_small = human.rename(columns={"genre_broad": "genre"})[cols]
        return human_small, human


def chunk_text(text: str, chunk_words: int = CHUNK_WORDS):
    words = text.split()
    if not words:
        return [""]
    return [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), chunk_words)]


def compute_embeddings(df: pd.DataFrame) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    song_embeddings = []
    for text in df["text"]:
        chunks = chunk_text(text)
        chunk_vecs = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
        song_vec = np.mean(chunk_vecs, axis=0)
        norm = np.linalg.norm(song_vec)
        if norm > 0:
            song_vec = song_vec / norm
        song_embeddings.append(song_vec)
    return np.vstack(song_embeddings)


def distance_to_centroid(embeddings: np.ndarray, groups: pd.Series) -> np.ndarray:
    """Cosine distance of each row's embedding to its own group's centroid embedding
    (group = genre here; group = genre x source once AI data exists)."""
    dists = np.zeros(len(groups))
    for g in groups.unique():
        idx = np.where(groups.values == g)[0]
        centroid = embeddings[idx].mean(axis=0)
        centroid_norm = centroid / np.linalg.norm(centroid) if np.linalg.norm(centroid) > 0 else centroid
        cos_sim = embeddings[idx] @ centroid_norm
        dists[idx] = 1 - cos_sim
    return dists


def compute_lexical_metrics(text: str) -> dict:
    from lexical_diversity import lex_div as ld

    tokens = text.split()
    if len(tokens) < 10:
        return {"mtld": np.nan, "hdd": np.nan, "ttr": np.nan}
    flt = ld.flemmatize(text)  # simple flemmatized token list (no aggressive stemming)
    mtld = ld.mtld(flt)
    try:
        hdd = ld.hdd(flt)
    except Exception:
        hdd = np.nan
    ttr = len(set(flt)) / len(flt)
    return {"mtld": mtld, "hdd": hdd, "ttr": ttr}


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    combined, human_full = load_corpus()

    print(f"Computing embeddings for {len(combined)} songs with {EMBEDDING_MODEL_NAME} "
          f"(chunk size = {CHUNK_WORDS} words, mean-pooled, L2-normalized)...")
    embeddings = compute_embeddings(combined)
    combined["semantic_dispersion_genre"] = distance_to_centroid(embeddings, combined["genre"])

    print("Computing lexical diversity (MTLD primary, HD-D robustness, TTR descriptive)...")
    lex_rows = combined["text"].apply(compute_lexical_metrics)
    lex_df = pd.DataFrame(list(lex_rows))
    combined = pd.concat([combined.reset_index(drop=True), lex_df.reset_index(drop=True)], axis=1)

    out_path = PROCESSED / "features_song_level.parquet"
    combined.drop(columns=["text"]).to_parquet(out_path, index=False)
    # embeddings saved separately (not tabular-friendly, and to avoid bloating the
    # feature table); row order matches `combined`
    np.save(PROCESSED / "embeddings_all-mpnet-base-v2.npy", embeddings)

    print(f"\nWrote {out_path} ({len(combined)} rows) and embeddings .npy")
    print("\nSemantic dispersion (distance to genre centroid) by source:")
    print(combined.groupby("source")["semantic_dispersion_genre"].describe())
    print("\nMTLD by source:")
    print(combined.groupby("source")["mtld"].describe())
    print("\nEmbedding model:", EMBEDDING_MODEL_NAME)


if __name__ == "__main__":
    main()
