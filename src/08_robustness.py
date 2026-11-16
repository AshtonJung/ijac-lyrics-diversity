"""
08_robustness.py

Robustness suite (Section 15 of CLAUDE.md), run against
data/processed/features_song_level.parquet + data/processed/ai_lyrics_qc.parquet once
both exist with a real AI arm. Writes outputs/tables/robustness_summary.csv with one row
per check: analysis name, estimate, CI, p-value (if applicable), n, interpretation.

Checks implemented here (numbered to match CLAUDE.md Section 15):
  1. Genre-stratified analyses         -- refit primary model within each genre_broad
  2. Length-restricted sample          -- drop songs outside the central word-count band
  3. Exclude memorization/QC-flagged   -- refit excluding qc_any_flag rows
  4. Alternative lexical metric        -- HD-D instead of MTLD as the outcome
  5. Bootstrap CIs                     -- song-level (not pairwise-distance) bootstrap
                                          on the primary AI-vs-human contrast
  6. With/without repeated-chorus norm -- primary outcome recomputed on chorus-deduped
                                          text vs raw text (see Section 7 of
                                          preanalysis_decisions_v2.md)
  7. Alternative embedding aggregation -- e.g. first-chunk-only vs mean-pooled

Not runnable end-to-end until Phase 5 generation + Phase 6 QC + Phase 7 (combined
features) all exist. Exits with a clear blocked message otherwise, per the same pattern
as src/07_primary_analysis.py.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
GENERATED = ROOT / "data" / "generated"
OUTPUTS = ROOT / "outputs"
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260808
LENGTH_TRIM_PCTL = 10  # symmetric trim: drop below p10 and above p90 of word_count


def require_combined_features():
    feat_path = PROCESSED / "features_song_level.parquet"
    if not feat_path.exists():
        return None
    df = pd.read_parquet(feat_path)
    if "ai" not in set(df["source"].unique()):
        return None
    return df


def check_genre_stratified(df, outcome, results):
    for g in sorted(df["genre"].unique()):
        sub = df[df["genre"] == g]
        if sub["source"].nunique() < 2 or len(sub) < 10:
            results.append({"analysis": f"genre_stratified[{g}]_{outcome}", "estimate": np.nan,
                             "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan,
                             "n": len(sub), "interpretation": "insufficient n or one group missing"})
            continue
        m = smf.ols(f"{outcome} ~ AI", data=sub).fit(cov_type="HC1")
        ci = m.conf_int().loc["AI"]
        results.append({"analysis": f"genre_stratified[{g}]_{outcome}", "estimate": m.params["AI"],
                         "ci_low": ci[0], "ci_high": ci[1], "p_value": m.pvalues["AI"],
                         "n": len(sub), "interpretation": f"AI-human diff in {outcome} within {g}"})


def check_bootstrap_contrast(df, outcome, results, cluster_col="prompt_id"):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    clusters = df[cluster_col].unique()
    diffs = []
    for _ in range(N_BOOTSTRAP):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        boot = pd.concat([df[df[cluster_col] == c] for c in sampled_clusters])
        human_mean = boot.loc[boot["source"] == "human", outcome].mean()
        ai_mean = boot.loc[boot["source"] == "ai", outcome].mean()
        diffs.append(ai_mean - human_mean)
    diffs = np.array(diffs)
    results.append({
        "analysis": f"song_level_cluster_bootstrap_{outcome}",
        "estimate": np.mean(diffs), "ci_low": np.percentile(diffs, 2.5),
        "ci_high": np.percentile(diffs, 97.5), "p_value": np.nan,
        "n": len(df), "interpretation": (
            f"AI-human mean diff in {outcome}, {N_BOOTSTRAP}x cluster bootstrap "
            f"resampling matched-prompt clusters (not naive pairwise-distance resampling)"),
    })


def check_length_restricted(df, outcome, results):
    lo, hi = np.percentile(df["word_count"], [LENGTH_TRIM_PCTL, 100 - LENGTH_TRIM_PCTL])
    sub = df[(df["word_count"] >= lo) & (df["word_count"] <= hi)]
    if sub["source"].nunique() < 2 or len(sub) < 10:
        results.append({"analysis": f"length_restricted_{outcome}", "estimate": np.nan,
                         "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan,
                         "n": len(sub), "interpretation": "insufficient n after trim"})
        return
    m = smf.ols(f"{outcome} ~ AI", data=sub).fit(
        cov_type="cluster", cov_kwds={"groups": sub["prompt_id"]})
    ci = m.conf_int().loc["AI"]
    results.append({
        "analysis": f"length_restricted_{outcome}", "estimate": m.params["AI"],
        "ci_low": ci[0], "ci_high": ci[1], "p_value": m.pvalues["AI"], "n": len(sub),
        "interpretation": (f"AI-human diff in {outcome}, word_count restricted to "
                            f"[{lo:.0f}, {hi:.0f}] (p{LENGTH_TRIM_PCTL}-p{100-LENGTH_TRIM_PCTL} "
                            f"of the combined sample), cluster-robust SE"),
    })


def check_qc_flag_exclusion(df, outcome, results):
    qc_path = PROCESSED / "ai_lyrics_qc.parquet"
    if not qc_path.exists():
        results.append({"analysis": f"exclude_qc_flagged_{outcome}", "estimate": np.nan,
                         "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan, "n": np.nan,
                         "interpretation": "ai_lyrics_qc.parquet not found"})
        return
    qc = pd.read_parquet(qc_path)
    qc["row_id"] = qc["prompt_id"] + "_" + qc["condition"]
    flagged_ids = set(qc.loc[qc["qc_any_flag"], "row_id"])
    sub = df[~((df["source"] == "ai") & (df["row_id"].isin(flagged_ids)))]
    n_dropped = (df["source"] == "ai").sum() - (sub["source"] == "ai").sum()
    if sub.loc[sub["source"] == "ai"].empty:
        results.append({"analysis": f"exclude_qc_flagged_{outcome}", "estimate": np.nan,
                         "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan, "n": len(sub),
                         "interpretation": f"all AI rows flagged, none left after exclusion ({n_dropped} dropped)"})
        return
    m = smf.ols(f"{outcome} ~ AI", data=sub).fit(
        cov_type="cluster", cov_kwds={"groups": sub["prompt_id"]})
    ci = m.conf_int().loc["AI"]
    results.append({
        "analysis": f"exclude_qc_flagged_{outcome}", "estimate": m.params["AI"],
        "ci_low": ci[0], "ci_high": ci[1], "p_value": m.pvalues["AI"], "n": len(sub),
        "interpretation": (f"AI-human diff in {outcome} excluding {n_dropped} QC-flagged "
                            f"AI rows (length/refusal/duplicate/chorus/memorization flags "
                            f"from src/05_quality_control.py), cluster-robust SE"),
    })


def check_hdd_alternative(df, results):
    if "hdd" not in df.columns or df["hdd"].isna().all():
        results.append({"analysis": "alternative_lexical_metric_hdd", "estimate": np.nan,
                         "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan, "n": np.nan,
                         "interpretation": "hdd column missing or all-NaN"})
        return
    sub = df.dropna(subset=["hdd"])
    m = smf.ols("hdd ~ AI", data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["prompt_id"]})
    ci = m.conf_int().loc["AI"]
    results.append({
        "analysis": "alternative_lexical_metric_hdd", "estimate": m.params["AI"],
        "ci_low": ci[0], "ci_high": ci[1], "p_value": m.pvalues["AI"], "n": len(sub),
        "interpretation": "AI-human diff using HD-D instead of MTLD as the lexical-diversity outcome",
    })


def _dedup_repeated_lines(text: str, max_repeats: int = 2) -> str:
    """Keep only the first `max_repeats` occurrences of any exactly-repeated line.
    Simplified chorus-normalization: operates on lines, not semantic chorus detection."""
    lines = text.split("\n")
    counts = {}
    kept = []
    for line in lines:
        key = line.strip().lower()
        if not key:
            kept.append(line)
            continue
        counts[key] = counts.get(key, 0) + 1
        if counts[key] <= max_repeats:
            kept.append(line)
    return "\n".join(kept)


def check_chorus_dedup_mtld(results):
    """Simplified version of check 6: recompute MTLD (lexical outcome only -- this
    variant does not re-run the embedding pipeline) on chorus-deduplicated text versus
    the original raw-text MTLD already in features_song_level.parquet."""
    from lexical_diversity import lex_div as ld

    human = pd.read_parquet(PROCESSED / "human_lyrics_labeled.parquet")
    ai_path = GENERATED / "ai_lyrics_raw.jsonl"
    pilot_subset_path = PROCESSED / "generation_prompts_pilot_subset.csv"
    if not (ai_path.exists() and pilot_subset_path.exists()):
        results.append({"analysis": "chorus_dedup_mtld", "estimate": np.nan, "ci_low": np.nan,
                         "ci_high": np.nan, "p_value": np.nan, "n": np.nan,
                         "interpretation": "AI data not available"})
        return
    import json
    rows = [json.loads(l) for l in open(ai_path) if l.strip()]
    ai = pd.DataFrame(rows)
    ai = ai[ai["generation_status"] == "ok"]
    pilot_ids = set(pd.read_csv(pilot_subset_path)["prompt_id"])
    ai = ai[ai["prompt_id"].isin(pilot_ids)].copy()

    def mtld_of(text):
        norm = re.sub(r"[^a-z0-9'\s]", " ", _dedup_repeated_lines(text).lower())
        norm = re.sub(r"\s+", " ", norm).strip()
        toks = norm.split()
        if len(toks) < 10:
            return np.nan
        return ld.mtld(ld.flemmatize(norm))

    human_mtld = human["analysis_text"].apply(mtld_of)
    ai_mtld = ai["generation_text"].apply(mtld_of)
    human_df = pd.DataFrame({"mtld_dedup": human_mtld, "AI": 0,
                              "prompt_id": "P" + human["song_id"].astype(str)})
    ai_df = pd.DataFrame({"mtld_dedup": ai_mtld.values, "AI": 1, "prompt_id": ai["prompt_id"].values})
    sub = pd.concat([human_df, ai_df], ignore_index=True).dropna(subset=["mtld_dedup"])
    m = smf.ols("mtld_dedup ~ AI", data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["prompt_id"]})
    ci = m.conf_int().loc["AI"]
    results.append({
        "analysis": "chorus_dedup_mtld", "estimate": m.params["AI"], "ci_low": ci[0],
        "ci_high": ci[1], "p_value": m.pvalues["AI"], "n": len(sub),
        "interpretation": ("AI-human diff in MTLD after collapsing lines repeated >2x "
                            "(simplified chorus normalization; line-exact, not semantic) "
                            "-- lexical outcome only, embeddings not recomputed for this variant"),
    })


def main():
    df = require_combined_features()
    if df is None:
        print("BLOCKED: data/processed/features_song_level.parquet does not yet contain "
              "an AI arm. Robustness checks require Phase 5 (generation), Phase 6 (QC), "
              "and a Phase-7 re-run on the combined human+AI corpus first.")
        return

    df = df.copy()
    df["AI"] = (df["source"] == "ai").astype(int)
    results = []

    for outcome in ["semantic_dispersion_genre", "mtld"]:
        check_genre_stratified(df, outcome, results)
        check_bootstrap_contrast(df, outcome, results)
        check_length_restricted(df, outcome, results)
        check_qc_flag_exclusion(df, outcome, results)

    check_hdd_alternative(df, results)
    check_chorus_dedup_mtld(results)

    # Not yet implemented: check 7 (alternative embedding aggregation -- would need a
    # second embeddings pass with a different chunking/pooling choice, not run here).

    out = pd.DataFrame(results)
    OUTPUTS.joinpath("tables").mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUTS / "tables" / "robustness_summary.csv", index=False)
    print(out)
    print(f"\nWrote {len(out)} robustness rows to outputs/tables/robustness_summary.csv")


if __name__ == "__main__":
    main()
