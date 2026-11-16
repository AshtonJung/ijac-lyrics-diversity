"""
07_primary_analysis.py

Confirmatory statistical models (Section 13 of CLAUDE.md), applied to
data/processed/features_song_level.parquet (produced by src/06_features.py).

Primary model (locked in outputs/preanalysis_decisions_v2.md Section 5):

    Diversity_i = b0 + b1*AI_i + b2*log(WordCount_i) + Genre_broad FE + Theme FE
                  + b3*AI_i x Genre_broad_i + e_i

  fit separately for:
    - Outcome A: semantic_dispersion_genre (distance to genre centroid)
    - Outcome B: mtld

  Standard errors clustered at matched-prompt level (`prompt_id`, i.e. the human song a
  set of AI generations was matched to -- up to 3 AI rows per human song, one per
  temperature condition), because those AI rows are not independent draws.

  Genre/Theme fixed effects: only Theme is available for AI rows once labels are carried
  over from the matched human prompt (Section 3 of preanalysis_decisions_v2.md); this
  script pulls theme/genre through from the prompt-level metadata rather than
  re-deriving it from AI-generated text (the AI text was generated FROM these labels, so
  re-classifying it would be circular).

This script will not produce a valid comparison until Phase 5 (AI generation) has run
and Phase 6/7 have been re-run on the combined corpus -- as of this writing,
data/processed/features_song_level.parquet contains ONLY human rows (source == "human"
for every row), so this script exits after reporting that the AI arm is missing rather
than fitting a one-group "model" that would misrepresent the confirmatory design.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"


def fit_clustered_model(df: pd.DataFrame, outcome: str, cluster_col: str):
    formula = f"{outcome} ~ AI + log_word_count + C(genre) + AI:C(genre)"
    model = smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df[cluster_col]}
    )
    return model


def main():
    feat_path = PROCESSED / "features_song_level.parquet"
    if not feat_path.exists():
        print(f"BLOCKED: {feat_path} not found. Run src/06_features.py first.")
        return

    df = pd.read_parquet(feat_path)
    if "ai" not in set(df["source"].unique()):
        print("BLOCKED: features_song_level.parquet contains only "
              f"{sorted(df['source'].unique())} -- no AI arm yet. The primary "
              "human-vs-AI confirmatory model cannot be fit until Phase 5 (generation) "
              "completes and src/06_features.py is re-run on the combined corpus. "
              "Refusing to fit a placeholder model rather than report a misleading "
              "single-group result.")
        print("\nDescriptive human-only baseline (for reference, NOT a confirmatory "
              "result):")
        print(df.groupby("genre")[["semantic_dispersion_genre", "mtld"]].agg(["mean", "std", "count"]))
        return

    df = df.copy()
    df["AI"] = (df["source"] == "ai").astype(int)
    df["log_word_count"] = np.log(df["word_count"].clip(lower=1))
    # prompt_id needed as cluster variable -- must be carried through by Phase 7 once
    # AI rows exist (row_id currently encodes prompt_id + temperature for AI rows,
    # song_id for human rows); left as TODO to wire up exactly once real data exists.

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "tables").mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "model_results").mkdir(parents=True, exist_ok=True)
    results = {}
    coef_rows = []
    for outcome in ["semantic_dispersion_genre", "mtld"]:
        model = fit_clustered_model(df, outcome, cluster_col="prompt_id")
        results[outcome] = model
        print(model.summary())
        with open(OUTPUTS / "model_results" / f"model_{outcome}.txt", "w") as f:
            f.write(str(model.summary()))
        ci = model.conf_int()
        n_clusters = df["prompt_id"].nunique()
        for term in model.params.index:
            coef_rows.append({
                "outcome": outcome, "term": term, "coef": model.params[term],
                "std_err": model.bse[term], "ci_low": ci.loc[term, 0],
                "ci_high": ci.loc[term, 1], "p_value": model.pvalues[term],
                "n": int(model.nobs), "n_clusters": n_clusters,
            })
    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(OUTPUTS / "tables" / "table2_primary_models.csv", index=False)
    print(f"\nWrote outputs/tables/table2_primary_models.csv ({len(coef_df)} rows)")

    # AI-arm-only: condition comparison. If the AI arm's condition values are the
    # low/medium/high temperature labels (real API data, preanalysis_decisions_v5.md),
    # this is the RQ3/H3 temperature-manipulation test. Reports EVERY non-intercept
    # dummy term (not just the first), since with 3 levels there are 2 contrasts.
    ai_only = df[df["source"] == "ai"].copy()
    cond_rows = []
    for outcome in ["semantic_dispersion_genre", "mtld"]:
        m = smf.ols(f"{outcome} ~ C(condition)", data=ai_only).fit(
            cov_type="cluster", cov_kwds={"groups": ai_only["prompt_id"]})
        ci = m.conf_int()
        for term in [t for t in m.params.index if t != "Intercept"]:
            cond_rows.append({
                "outcome": outcome, "term": term, "coef": m.params[term],
                "std_err": m.bse[term], "ci_low": ci.loc[term, 0], "ci_high": ci.loc[term, 1],
                "p_value": m.pvalues[term], "n": int(m.nobs),
                "n_clusters": ai_only["prompt_id"].nunique(),
            })
    cond_df = pd.DataFrame(cond_rows)
    cond_df.to_csv(OUTPUTS / "tables" / "table_condition_comparison.csv", index=False)
    print(f"Wrote outputs/tables/table_condition_comparison.csv ({len(cond_df)} rows)")


if __name__ == "__main__":
    main()
