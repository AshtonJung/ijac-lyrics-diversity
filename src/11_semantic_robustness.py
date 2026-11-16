"""
11_semantic_robustness.py

Semantic-dispersion robustness checks (requested by researcher, 2026-10-07), per
Section 15 of CLAUDE.md and the researcher's explicit audit instructions.

The PRIMARY semantic-dispersion measure (pooled human+AI genre centroid, unweighted
mean of 120-word chunk embeddings from sentence-transformers/all-mpnet-base-v2) is NOT
replaced -- outputs/preanalysis_decisions.md Section 4 specifies "distance-to-
source-centroid" and src/06_features.py has always computed this as a centroid pooled
over the genre group (human+AI combined), consistent with the manuscript's own Section
III.D text ("a common centroid was calculated using all human and AI lyric vectors
combined"). This script tests whether the headline result is an artifact of that
specific choice -- it does not change the choice itself.

Checks implemented:
  A. Group-specific (genre x source) centroid: each lyric's distance to its OWN
     source's centroid within its genre, instead of the pooled human+AI centroid.
  B. Balanced-sample / equal-size resampling: genre-stratified downsampling to equal
     human/AI counts per genre, centroid recomputed fresh within each resample
     (fixed seed, 2000 iterations, matching the existing bootstrap convention in
     src/08_robustness.py).
  C. Alternative chunk-aggregation: length-weighted mean of chunk embeddings (weighted
     by each chunk's word count) before L2 normalization, instead of the primary
     script's unweighted mean across chunks -- requires re-embedding from raw text
     with the SAME model (all-mpnet-base-v2).
  D. Second embedding model (sentence-transformers/all-MiniLM-L6-v2), primary
     unweighted-mean aggregation, re-embedded from raw text.

For C and D, the corrected (genre-weighted, interaction-model-based) overall contrast
is computed the same way as src/10_corrected_contrasts.py, for direct comparability to
the corrected primary result.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
GENERATED = ROOT / "data" / "generated"
OUTPUTS = ROOT / "outputs"

GENRES = ["country", "electronic/dance", "hip-hop/rap", "pop", "r&b/soul", "rock/alternative"]
REFERENCE_GENRE = "country"
CHUNK_WORDS = 120
PRIMARY_GENERATION_METHOD = "anthropic_api"
N_RESAMPLE = 2000
RESAMPLE_SEED = 20261007  # distinct, documented seed for this new check


# ---------------------------------------------------------------------------
# Shared helpers (mirrors src/06_features.py and src/10_corrected_contrasts.py
# exactly, duplicated here rather than imported so this script has no side effect
# on, and no dependency on mutable state in, the locked pipeline scripts)
# ---------------------------------------------------------------------------

def load_text_corpus():
    human = pd.read_parquet(PROCESSED / "human_lyrics_labeled.parquet")
    human = human.rename(columns={"analysis_text": "text"})
    human["source"] = "human"
    human["prompt_id"] = "P" + human["song_id"].astype(str)
    human_small = human.rename(columns={"genre_broad": "genre"})[
        ["source", "genre", "text", "prompt_id"]
    ].copy()
    human_small["word_count"] = human_small["text"].str.split().apply(len)

    rows = [json.loads(l) for l in open(GENERATED / "ai_lyrics_raw.jsonl") if l.strip()]
    ai = pd.DataFrame(rows)
    ai = ai[ai["generation_status"] == "ok"]
    ai = ai[ai["generation_method"] == PRIMARY_GENERATION_METHOD].copy()
    ai["text"] = ai["generation_text"]
    ai["source"] = "ai"
    ai["word_count"] = ai["actual_word_count"]
    ai_small = ai[["source", "genre", "text", "prompt_id", "word_count"]].copy()

    combined = pd.concat([human_small, ai_small], ignore_index=True)
    return combined


def chunk_text(text: str, chunk_words: int = CHUNK_WORDS):
    words = text.split()
    if not words:
        return [""]
    return [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), chunk_words)]


def embed_corpus(texts, model_name: str, weighted: bool):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    song_embeddings = []
    for text in texts:
        chunks = chunk_text(text)
        chunk_word_counts = np.array([max(len(c.split()), 1) for c in chunks], dtype=float)
        chunk_vecs = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
        if weighted:
            w = chunk_word_counts / chunk_word_counts.sum()
            song_vec = (chunk_vecs * w[:, None]).sum(axis=0)
        else:
            song_vec = np.mean(chunk_vecs, axis=0)
        norm = np.linalg.norm(song_vec)
        if norm > 0:
            song_vec = song_vec / norm
        song_embeddings.append(song_vec)
    return np.vstack(song_embeddings)


def distance_to_group_centroid(embeddings: np.ndarray, groups: pd.Series) -> np.ndarray:
    dists = np.zeros(len(groups))
    for g in groups.unique():
        idx = np.where(groups.values == g)[0]
        centroid = embeddings[idx].mean(axis=0)
        cn = centroid / np.linalg.norm(centroid) if np.linalg.norm(centroid) > 0 else centroid
        dists[idx] = 1 - embeddings[idx] @ cn
    return dists


def genre_weights(df: pd.DataFrame) -> dict:
    human = df[df["source"] == "human"]
    counts = human["genre"].value_counts()
    n = len(human)
    return {g: counts.get(g, 0) / n for g in GENRES}


def fit_interaction_model(df: pd.DataFrame, outcome_col: str):
    d = df.copy()
    d["AI"] = (d["source"] == "ai").astype(int)
    d["log_word_count"] = np.log(d["word_count"].clip(lower=1))
    formula = f"{outcome_col} ~ AI + log_word_count + C(genre) + AI:C(genre)"
    return smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["prompt_id"]})


def contrast_vector(model, genre: str) -> pd.Series:
    names = model.params.index
    c = pd.Series(0.0, index=names)
    c["AI"] = 1.0
    if genre != REFERENCE_GENRE:
        c[f"AI:C(genre)[T.{genre}]"] = 1.0
    return c


def weighted_overall_contrast(df: pd.DataFrame, dispersion_col: str):
    w = genre_weights(df)
    model = fit_interaction_model(df, dispersion_col)
    c_overall = sum(w[g] * contrast_vector(model, g) for g in GENRES)
    b = model.params.reindex(c_overall.index)
    V = model.cov_params().reindex(index=c_overall.index, columns=c_overall.index)
    est = float(c_overall.values @ b.values)
    se = float(np.sqrt(c_overall.values @ V.values @ c_overall.values))
    z = est / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    crit = stats.norm.ppf(0.975)
    return {"estimate": est, "se": se, "ci_low": est - crit * se, "ci_high": est + crit * se, "p_value": p}


# ---------------------------------------------------------------------------
# Check A: group-specific (genre x source) centroid
# ---------------------------------------------------------------------------

def check_A_group_specific_centroid(feat: pd.DataFrame, emb: np.ndarray, results: list):
    group_key = feat["genre"].astype(str) + "||" + feat["source"].astype(str)
    dists = distance_to_group_centroid(emb, group_key)
    df = feat.copy()
    df["semantic_dispersion_group_specific"] = dists
    out = weighted_overall_contrast(df, "semantic_dispersion_group_specific")
    results.append({
        "check": "A_group_specific_genre_x_source_centroid",
        "description": "Each lyric's distance to its OWN source's (human or AI) "
                        "centroid within its genre, instead of the pooled centroid.",
        **out, "n": len(df),
    })
    print(f"[A] group-specific centroid, weighted overall contrast: "
          f"{out['estimate']:.4f} [{out['ci_low']:.4f},{out['ci_high']:.4f}] p={out['p_value']:.2e}")


# ---------------------------------------------------------------------------
# Check B: balanced-sample / equal-size resampling (genre-stratified)
# ---------------------------------------------------------------------------

def check_B_balanced_resample(feat: pd.DataFrame, emb: np.ndarray, results: list):
    rng = np.random.default_rng(RESAMPLE_SEED)
    diffs = []
    idx_by_group = {}
    for g in GENRES:
        h_idx = np.where((feat["genre"].values == g) & (feat["source"].values == "human"))[0]
        a_idx = np.where((feat["genre"].values == g) & (feat["source"].values == "ai"))[0]
        idx_by_group[g] = (h_idx, a_idx)

    for _ in range(N_RESAMPLE):
        sel_idx = []
        for g in GENRES:
            h_idx, a_idx = idx_by_group[g]
            k = min(len(h_idx), len(a_idx))
            if k == 0:
                continue
            sel_idx.append(rng.choice(h_idx, size=k, replace=False))
            sel_idx.append(rng.choice(a_idx, size=k, replace=False))
        sel_idx = np.concatenate(sel_idx)
        sub_emb = emb[sel_idx]
        sub_genre = feat["genre"].values[sel_idx]
        sub_source = feat["source"].values[sel_idx]
        dists = distance_to_group_centroid(sub_emb, pd.Series(sub_genre))
        human_mean = dists[sub_source == "human"].mean()
        ai_mean = dists[sub_source == "ai"].mean()
        diffs.append(ai_mean - human_mean)

    diffs = np.array(diffs)
    balanced_n_per_genre = {g: min(len(idx_by_group[g][0]), len(idx_by_group[g][1])) for g in GENRES}
    results.append({
        "check": "B_balanced_equal_size_genre_stratified_resample",
        "description": (
            f"{N_RESAMPLE}x resampling, each genre downsampled to equal human/AI n "
            f"(seed={RESAMPLE_SEED}); per-genre balanced n: {balanced_n_per_genre}; "
            "centroid recomputed fresh within each resample (pooled per genre, same "
            "definition as primary, just on the balanced subsample); unadjusted "
            "AI-human mean difference reported, matching the existing bootstrap "
            "check's convention in src/08_robustness.py."
        ),
        "estimate": float(np.mean(diffs)), "se": float(np.std(diffs, ddof=1)),
        "ci_low": float(np.percentile(diffs, 2.5)), "ci_high": float(np.percentile(diffs, 97.5)),
        "p_value": np.nan, "n": int(sum(2 * v for v in balanced_n_per_genre.values())),
    })
    print(f"[B] balanced resample ({N_RESAMPLE}x), AI-human diff: "
          f"{np.mean(diffs):.4f} [{np.percentile(diffs,2.5):.4f},{np.percentile(diffs,97.5):.4f}]")


# ---------------------------------------------------------------------------
# Check C: alternative chunk-aggregation (length-weighted mean)
# ---------------------------------------------------------------------------

def check_C_weighted_aggregation(feat: pd.DataFrame, text_corpus: pd.DataFrame, results: list):
    emb_weighted = embed_corpus(text_corpus["text"].tolist(),
                                 "sentence-transformers/all-mpnet-base-v2", weighted=True)
    df = feat.copy()
    df["semantic_dispersion_weighted_chunks"] = distance_to_group_centroid(emb_weighted, df["genre"])
    out = weighted_overall_contrast(df, "semantic_dispersion_weighted_chunks")
    results.append({
        "check": "C_length_weighted_chunk_aggregation",
        "description": "Chunk embeddings combined via a word-count-weighted mean "
                        "(instead of the primary unweighted mean) before L2 "
                        "normalization, same embedding model (all-mpnet-base-v2), "
                        "pooled genre centroid as in the primary definition.",
        **out, "n": len(df),
    })
    print(f"[C] length-weighted chunk aggregation, weighted overall contrast: "
          f"{out['estimate']:.4f} [{out['ci_low']:.4f},{out['ci_high']:.4f}] p={out['p_value']:.2e}")
    return emb_weighted


# ---------------------------------------------------------------------------
# Check D: second embedding model
# ---------------------------------------------------------------------------

def check_D_second_embedding_model(feat: pd.DataFrame, text_corpus: pd.DataFrame, results: list):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    try:
        emb2 = embed_corpus(text_corpus["text"].tolist(), model_name, weighted=False)
    except Exception as e:
        results.append({
            "check": "D_second_embedding_model", "description": f"NOT RUN: {e}",
            "estimate": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan,
            "p_value": np.nan, "n": np.nan,
        })
        print(f"[D] second embedding model FAILED: {e}")
        return
    df = feat.copy()
    df["semantic_dispersion_minilm"] = distance_to_group_centroid(emb2, df["genre"])
    out = weighted_overall_contrast(df, "semantic_dispersion_minilm")
    results.append({
        "check": "D_second_embedding_model_all-MiniLM-L6-v2",
        "description": "Same pooled-centroid, unweighted-chunk-mean definition as "
                        "primary, but embedded with sentence-transformers/"
                        "all-MiniLM-L6-v2 instead of all-mpnet-base-v2.",
        **out, "n": len(df),
    })
    print(f"[D] second embedding model (MiniLM-L6-v2), weighted overall contrast: "
          f"{out['estimate']:.4f} [{out['ci_low']:.4f},{out['ci_high']:.4f}] p={out['p_value']:.2e}")


def main():
    feat = pd.read_parquet(PROCESSED / "features_song_level.parquet")
    emb = np.load(PROCESSED / "embeddings_all-mpnet-base-v2.npy")
    assert len(feat) == emb.shape[0], "feature/embedding row-count mismatch"

    results = []
    check_A_group_specific_centroid(feat, emb, results)
    check_B_balanced_resample(feat, emb, results)

    print("\nLoading raw text corpus for re-embedding checks C and D "
          "(this re-derives text identically to src/06_features.py's load_corpus, "
          "restricted to the primary anthropic_api cohort)...")
    text_corpus = load_text_corpus()
    assert len(text_corpus) == len(feat), (
        f"text corpus n={len(text_corpus)} != feature table n={len(feat)} -- "
        "row-order alignment cannot be assumed; aborting C/D rather than silently "
        "misaligning rows."
    )
    # Confirm row-level alignment on genre + word_count before trusting positional
    # alignment for checks C/D.
    genre_match = (text_corpus["genre"].values == feat["genre"].values).mean()
    wc_match = (text_corpus["word_count"].values == feat["word_count"].values).mean()
    print(f"Row-alignment check: genre match={genre_match:.4f}, word_count match={wc_match:.4f}")
    if genre_match < 0.999 or wc_match < 0.999:
        print("WARNING: text_corpus row order does not match features_song_level.parquet "
              "row order closely enough to trust positional alignment. Checks C/D skipped.")
        results.append({
            "check": "C_and_D_SKIPPED_row_alignment_failed", "description":
                f"genre_match={genre_match}, wc_match={wc_match}",
            "estimate": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan,
            "p_value": np.nan, "n": np.nan,
        })
    else:
        check_C_weighted_aggregation(feat, text_corpus, results)
        check_D_second_embedding_model(feat, text_corpus, results)

    out_df = pd.DataFrame(results)
    out_path = OUTPUTS / "tables" / "semantic_centroid_robustness.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(out_df)} rows)")
    print("\nFor reference, the CORRECTED primary (pooled centroid, unweighted mean, "
          "all-mpnet-base-v2) overall weighted contrast was -0.1116 "
          "[-0.1232,-0.1000] (from src/10_corrected_contrasts.py / "
          "outputs/tables/corrected_primary_contrasts.csv).")


if __name__ == "__main__":
    main()
