"""
10_corrected_contrasts.py

Statistical audit correction (requested by researcher, 2026-10-07).

The prespecified primary model (outputs/preanalysis_decisions.md Section 5, locked
2026-08-08, never amended on this point) is:

    Diversity_i = b0 + b1*AI_i + b2*log(WordCount_i) + Genre_broad FE + Theme FE
                  + b3*AI_i x Genre_broad_i + cluster(prompt_id) + e_i

(Theme FE was later dropped -- disclosed deviation, documented in research_audit.md and
final_statistical_audit.md, NOT re-added here.) The AI x Genre interaction term WAS
prespecified and IS what src/07_primary_analysis.py actually fits (confirmed by reading
the script and outputs/model_results/*.txt). This is kept as the primary model -- it is
NOT being changed.

What was wrong in the manuscript was the INTERPRETATION of the raw "AI" coefficient
from this model as an "adjusted average difference...after accounting for genre." Under
patsy's default treatment coding, C(genre) drops the alphabetically-first level
(country) as reference, so the raw "AI" coefficient is the AI-human contrast
specifically for country, not an average over genres. This script:

  1. Refits the exact same interaction model (reproducing table2_primary_models.csv
     as a check -- does NOT overwrite that file).
  2. Computes all six genre-specific adjusted AI-human contrasts from the fitted
     model's coefficients (not by refitting six separate unadjusted models -- those
     already exist as the genre-stratified ROBUSTNESS check in robustness_summary.csv
     and are a different, intentionally simpler specification used for the
     cross-genre-consistency check in Section IV.C).
  3. Computes ONE overall adjusted human-AI contrast as a human-genre-weighted average
     of the six genre-specific contrasts, with correct SE/CI/p-value obtained as a
     linear combination of the fitted model's coefficients using the model's own
     cluster-robust covariance matrix (delta-method-free since the contrast is linear
     in the parameters: Var(c'b) = c' V c).
  4. Runs a joint Wald test that all five AI:C(genre) interaction coefficients are
     jointly zero (cluster-robust), so the one nominally-significant interaction term
     (AI:electronic/dance, p=.044 uncorrected) is not overinterpreted in isolation.

Weights = human-sample genre distribution (N=213): the natural choice given the
estimand is "the adjusted AI-human gap an AI system would produce if asked to write
lyrics across the SAME genre mix as the real human corpus used in this study."
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

GENRES = ["country", "electronic/dance", "hip-hop/rap", "pop", "r&b/soul", "rock/alternative"]
REFERENCE_GENRE = "country"  # confirmed: alphabetically first, dropped by patsy C(genre)


def genre_weights(df: pd.DataFrame) -> dict:
    human = df[df["source"] == "human"]
    counts = human["genre"].value_counts()
    n = len(human)
    w = {g: counts.get(g, 0) / n for g in GENRES}
    assert abs(sum(w.values()) - 1.0) < 1e-9
    return w


def fit_model(df: pd.DataFrame, outcome: str):
    df = df.copy()
    df["AI"] = (df["source"] == "ai").astype(int)
    df["log_word_count"] = np.log(df["word_count"].clip(lower=1))
    formula = f"{outcome} ~ AI + log_word_count + C(genre) + AI:C(genre)"
    model = smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["prompt_id"]}
    )
    return model


def contrast_vector(model, genre: str) -> pd.Series:
    """Linear contrast selecting the AI-human adjusted effect for a single genre."""
    names = model.params.index
    c = pd.Series(0.0, index=names)
    c["AI"] = 1.0
    if genre != REFERENCE_GENRE:
        term = f"AI:C(genre)[T.{genre}]"
        assert term in names, f"missing term {term}"
        c[term] = 1.0
    return c


def linear_contrast_stats(model, c: pd.Series, alpha=0.05):
    b = model.params.reindex(c.index)
    V = model.cov_params().reindex(index=c.index, columns=c.index)
    est = float(c.values @ b.values)
    var = float(c.values @ V.values @ c.values)
    se = np.sqrt(var)
    z = est / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    crit = stats.norm.ppf(1 - alpha / 2)
    return {
        "estimate": est, "se": se, "ci_low": est - crit * se, "ci_high": est + crit * se,
        "z": z, "p_value": p,
    }


def joint_interaction_test(model):
    """Wald test that all AI:C(genre)[...] terms are jointly zero, using the model's
    cluster-robust covariance (model was fit with cov_type='cluster')."""
    interaction_terms = [n for n in model.params.index if n.startswith("AI:C(genre)")]
    hyp = ", ".join(f"{t} = 0" for t in interaction_terms)
    wald = model.wald_test(hyp, scalar=True)
    return {
        "terms_tested": interaction_terms,
        "n_restrictions": len(interaction_terms),
        "statistic": float(np.asarray(wald.statistic).squeeze()),
        "df_denom": getattr(wald, "df_denom", None),
        "df_num": getattr(wald, "df_num", None),
        "p_value": float(np.asarray(wald.pvalue).squeeze()),
    }


def main():
    feat_path = PROCESSED / "features_song_level.parquet"
    df = pd.read_parquet(feat_path)
    w = genre_weights(df)
    print("Genre weights (human N=213 distribution):")
    for g in GENRES:
        print(f"  {g}: {w[g]:.4f} (n={round(w[g]*213)})")

    rows = []
    reproduction_check = []

    for outcome in ["semantic_dispersion_genre", "mtld"]:
        model = fit_model(df, outcome)

        # --- reproduction check against existing table2_primary_models.csv ---
        old = pd.read_csv(OUTPUTS / "tables" / "table2_primary_models.csv")
        old_sub = old[old["outcome"] == outcome].set_index("term")
        max_abs_diff = 0.0
        for term in model.params.index:
            if term in old_sub.index:
                diff = abs(model.params[term] - old_sub.loc[term, "coef"])
                max_abs_diff = max(max_abs_diff, diff)
        reproduction_check.append({"outcome": outcome, "max_abs_coef_diff_vs_existing_table2": max_abs_diff})

        # --- per-genre adjusted contrasts ---
        genre_contrast_vecs = {}
        for g in GENRES:
            c = contrast_vector(model, g)
            genre_contrast_vecs[g] = c
            stats_g = linear_contrast_stats(model, c)
            rows.append({
                "outcome": outcome, "contrast": f"genre[{g}]_adjusted_AI_minus_human",
                "weight": w[g], **stats_g,
            })

        # --- overall human-genre-weighted average contrast ---
        c_overall = sum(w[g] * genre_contrast_vecs[g] for g in GENRES)
        stats_overall = linear_contrast_stats(model, c_overall)
        rows.append({
            "outcome": outcome, "contrast": "overall_weighted_adjusted_AI_minus_human",
            "weight": 1.0, **stats_overall,
        })

        # --- old (uncorrected) reference-category reading, kept for the audit doc ---
        old_ai_row = old_sub.loc["AI"]
        rows.append({
            "outcome": outcome, "contrast": "OLD_reading_raw_AI_coef_mislabeled_as_average",
            "weight": np.nan, "estimate": old_ai_row["coef"], "se": old_ai_row["std_err"],
            "ci_low": old_ai_row["ci_low"], "ci_high": old_ai_row["ci_high"],
            "z": np.nan, "p_value": old_ai_row["p_value"],
        })

        # --- joint test of AI x genre interactions ---
        jt = joint_interaction_test(model)
        rows.append({
            "outcome": outcome, "contrast": "joint_wald_test_AIxGenre_interactions",
            "weight": np.nan, "estimate": jt["statistic"], "se": np.nan,
            "ci_low": np.nan, "ci_high": np.nan, "z": np.nan, "p_value": jt["p_value"],
        })
        print(f"\n[{outcome}] Joint Wald test (H0: all 5 AI:genre interactions = 0): "
              f"stat={jt['statistic']:.4f}, df_num={jt['df_num']}, p={jt['p_value']:.4f}")

    out_df = pd.DataFrame(rows)
    out_path = OUTPUTS / "tables" / "corrected_primary_contrasts.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(out_df)} rows)")

    repro_df = pd.DataFrame(reproduction_check)
    print("\nReproduction check vs existing table2_primary_models.csv "
          "(should be ~0, confirms this script refits the SAME model, not a new one):")
    print(repro_df.to_string(index=False))

    # --- corrected_table2.csv: same shape/purpose as table2_primary_models.csv, but
    #     with the headline "AI" row correctly labeled and the weighted-average row
    #     added; temperature-condition rows are untouched by this issue and are
    #     copied through unchanged from table_condition_comparison.csv for a single
    #     combined reference table. ---
    cond = pd.read_csv(OUTPUTS / "tables" / "table_condition_comparison.csv")
    corrected_rows = []
    for outcome in ["semantic_dispersion_genre", "mtld"]:
        sub = out_df[out_df["outcome"] == outcome]
        overall = sub[sub["contrast"] == "overall_weighted_adjusted_AI_minus_human"].iloc[0]
        country = sub[sub["contrast"] == f"genre[country]_adjusted_AI_minus_human"].iloc[0]
        corrected_rows.append({
            "outcome": outcome, "term": "AI (human-genre-weighted average contrast)",
            "coef": overall["estimate"], "std_err": overall["se"],
            "ci_low": overall["ci_low"], "ci_high": overall["ci_high"],
            "p_value": overall["p_value"], "note": "CORRECTED headline estimate",
        })
        corrected_rows.append({
            "outcome": outcome, "term": "AI (country, reference genre only -- OLD mislabel)",
            "coef": country["estimate"], "std_err": country["se"],
            "ci_low": country["ci_low"], "ci_high": country["ci_high"],
            "p_value": country["p_value"],
            "note": "This is what table2_primary_models.csv's raw 'AI' row actually is",
        })
    for _, r in cond.iterrows():
        corrected_rows.append({
            "outcome": r["outcome"], "term": f"Temperature: {r['term']}",
            "coef": r["coef"], "std_err": r["std_err"], "ci_low": r["ci_low"],
            "ci_high": r["ci_high"], "p_value": r["p_value"], "note": "unchanged from original",
        })
    corrected_table2 = pd.DataFrame(corrected_rows)
    corrected_table2.to_csv(OUTPUTS / "tables" / "corrected_table2.csv", index=False)
    print(f"Wrote {OUTPUTS / 'tables' / 'corrected_table2.csv'} ({len(corrected_table2)} rows)")


if __name__ == "__main__":
    main()
