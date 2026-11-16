"""
09_figures_tables.py

Publication-quality figures and tables (Sections 17-18 of CLAUDE.md). Reads from
data/processed/features_song_level.parquet (+ ai_lyrics_qc.parquet once it exists) and
outputs/tables/robustness_summary.csv. Writes PNG (+ SVG where feasible) to
outputs/figures/ and CSV to outputs/tables/.

Figures:
  1. Research design schematic       -- static diagram, no data dependency, can be built now
  2. Primary semantic diversity      -- point estimates + 95% CI, human vs AI
  3. Genre heterogeneity             -- AI-human difference by genre, with CIs
  4. Technology-management result    -- standard vs diversity-oriented condition
  5. 2D embedding visualization      -- PCA/UMAP, descriptive only (not inferential)

Tables:
  1. Sample characteristics by source/condition
  2. Primary statistical models (semantic + lexical)
  3. Genre interactions / heterogeneity
  4. Robustness summary (already produced by src/08_robustness.py; this script only
     reformats it for the manuscript-facing version)

Figure 1 has no data dependency and is built unconditionally every run. Figures 2-5 and
Tables 1-3 require the combined human+AI feature set and exit with a clear blocked
message otherwise (same pattern as the rest of the pipeline).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"

# dataviz palette placeholder -- swap for validated palette per project dataviz skill
# before final figure export (not yet applied here; these are structural drafts).
COLOR_HUMAN = "#4C72B0"
COLOR_AI = "#DD8452"


def figure1_design_schematic():
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis("off")
    steps = [
        "Human lyric corpus\n(N=213, 6 genres)",
        "Matched metadata prompts\n(genre, theme, mood,\nperspective, length --\nno lyric text)",
        "AI generation\n(standard vs\ndiversity-oriented)",
        "Diversity analysis\n(semantic + lexical,\nclustered by prompt)",
    ]
    x_positions = [0.05, 0.30, 0.55, 0.80]
    for x, label in zip(x_positions, steps):
        ax.text(x, 0.5, label, ha="center", va="center", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#EAEAF2", edgecolor="#333333"))
    for x0, x1 in zip(x_positions[:-1], x_positions[1:]):
        ax.annotate("", xy=(x1 - 0.06, 0.5), xytext=(x0 + 0.06, 0.5),
                     arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.suptitle("Figure 1. Research design", fontsize=11)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "figure1_design_schematic.png", dpi=200)
    fig.savefig(FIGURES / "figure1_design_schematic.svg")
    plt.close(fig)
    print("Wrote figure1_design_schematic.{png,svg}")


def require_combined_features():
    feat_path = PROCESSED / "features_song_level.parquet"
    if not feat_path.exists():
        return None
    df = pd.read_parquet(feat_path)
    if "ai" not in set(df["source"].unique()):
        return None
    return df


def table1_sample_characteristics(df: pd.DataFrame):
    tbl = df.groupby(["source"]).agg(
        n=("row_id", "count"),
        word_count_mean=("word_count", "mean") if "word_count" in df.columns else ("mtld", "count"),
        mtld_mean=("mtld", "mean"),
        mtld_sd=("mtld", "std"),
        semantic_dispersion_mean=("semantic_dispersion_genre", "mean"),
        semantic_dispersion_sd=("semantic_dispersion_genre", "std"),
    ).reset_index()
    TABLES.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(TABLES / "table1_sample_characteristics.csv", index=False)
    print(tbl)


def figure2_semantic_diversity(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, outcome, ylabel in zip(
        axes, ["semantic_dispersion_genre", "mtld"],
        ["Semantic dispersion\n(cosine distance to genre centroid)", "MTLD (lexical diversity)"],
    ):
        stats = df.groupby("source")[outcome].agg(["mean", "std", "count"])
        stats["se"] = stats["std"] / stats["count"] ** 0.5
        stats["ci95"] = 1.96 * stats["se"]
        order = ["human", "ai"]
        stats = stats.reindex(order)
        ax.errorbar(order, stats["mean"], yerr=stats["ci95"], fmt="o", capsize=5,
                     color=COLOR_HUMAN, ecolor="#555555", markersize=8)
        for i, src in enumerate(order):
            ax.scatter([i], [stats.loc[src, "mean"]],
                       color=COLOR_HUMAN if src == "human" else COLOR_AI, s=80, zorder=3)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticks(range(len(order)))
        n_human = int(stats.loc["human", "count"])
        n_ai = int(stats.loc["ai", "count"])
        ax.set_xticklabels([f"Human\n(N={n_human})", f"AI\n(N={n_ai})"], fontsize=8)
    fig.suptitle("Figure 2. Semantic and lexical diversity, human vs. AI (95% CI)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure2_semantic_lexical_diversity.png", dpi=200)
    fig.savefig(FIGURES / "figure2_semantic_lexical_diversity.svg")
    plt.close(fig)
    print("Wrote figure2_semantic_lexical_diversity.{png,svg}")


def figure3_genre_heterogeneity():
    robust_path = TABLES / "robustness_summary.csv"
    if not robust_path.exists():
        print("Skipping Figure 3: outputs/tables/robustness_summary.csv not found "
              "(run src/08_robustness.py first).")
        return
    r = pd.read_csv(robust_path)
    r = r[r["analysis"].str.startswith("genre_stratified") & r["analysis"].str.endswith("semantic_dispersion_genre")]
    r = r.dropna(subset=["estimate"])
    if r.empty:
        print("Skipping Figure 3: no valid genre-stratified semantic-dispersion rows.")
        return
    r["genre"] = r["analysis"].str.extract(r"\[(.*?)\]")
    r = r.sort_values("estimate")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(r["estimate"], r["genre"], xerr=[r["estimate"] - r["ci_low"], r["ci_high"] - r["estimate"]],
                 fmt="o", capsize=4, color=COLOR_AI, ecolor="#555555")
    ax.axvline(0, color="#333333", linewidth=1, linestyle="--")
    ax.set_xlabel("AI - human difference in semantic dispersion (95% CI)", fontsize=9)
    fig.suptitle("Figure 3. Genre heterogeneity in the AI-human contrast", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure3_genre_heterogeneity.png", dpi=200)
    fig.savefig(FIGURES / "figure3_genre_heterogeneity.svg")
    plt.close(fig)
    print("Wrote figure3_genre_heterogeneity.{png,svg}")


def figure4_condition_comparison(df: pd.DataFrame):
    """CLAUDE.md Section 16 'Managing Technology' figure: diversity across temperature
    conditions with the human benchmark shown as a reference line. Falls back to a
    generic condition-comparison plot if the AI arm's conditions are not the
    low/medium/high temperature labels (e.g. if only pilot data is loaded)."""
    ai_only = df[df["source"] == "ai"]
    temp_order = ["low", "medium", "high"]
    if ai_only.empty:
        print("Skipping Figure 4: no AI arm present.")
        return
    if not set(ai_only["condition"].unique()) >= set(temp_order):
        print("Skipping Figure 4: AI arm conditions are not low/medium/high temperature "
              f"labels (found {sorted(ai_only['condition'].unique())}) -- likely pilot-only data.")
        return

    human_only = df[df["source"] == "human"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5))
    for ax, outcome, ylabel in zip(
        axes, ["semantic_dispersion_genre", "mtld"],
        ["Semantic dispersion\n(distance to genre centroid)", "MTLD (lexical diversity)"],
    ):
        stats = ai_only.groupby("condition")[outcome].agg(["mean", "std", "count"]).reindex(temp_order)
        stats["ci95"] = 1.96 * stats["std"] / stats["count"] ** 0.5
        x = range(len(temp_order))
        ax.errorbar(x, stats["mean"], yerr=stats["ci95"], fmt="o-", capsize=5,
                     color=COLOR_AI, ecolor="#555555", markersize=8, label="AI (by temperature)")
        h_mean = human_only[outcome].mean()
        h_ci = 1.96 * human_only[outcome].std() / len(human_only) ** 0.5
        ax.axhline(h_mean, color=COLOR_HUMAN, linestyle="--", linewidth=1.5, label="Human mean")
        ax.axhspan(h_mean - h_ci, h_mean + h_ci, color=COLOR_HUMAN, alpha=0.12)
        ax.set_xticks(list(x))
        ax.set_xticklabels(["low\n(0.2)", "medium\n(0.7)", "high\n(1.0)"], fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle("Figure 4. AI diversity by sampling temperature, vs. human benchmark", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure4_condition_comparison.png", dpi=200)
    fig.savefig(FIGURES / "figure4_condition_comparison.svg")
    plt.close(fig)
    print("Wrote figure4_condition_comparison.{png,svg}")


def figure5_embedding_viz(df: pd.DataFrame):
    """Descriptive only, per CLAUDE.md Section 17 -- not used as inferential evidence."""
    emb_path = PROCESSED / "embeddings_all-mpnet-base-v2.npy"
    if not emb_path.exists():
        print("Skipping Figure 5: embeddings .npy not found.")
        return
    import numpy as np
    from sklearn.decomposition import PCA

    emb = np.load(emb_path)
    if len(emb) != len(df):
        print(f"Skipping Figure 5: embeddings ({len(emb)}) and features ({len(df)}) "
              "row counts do not match -- re-run src/06_features.py to regenerate both together.")
        return
    coords = PCA(n_components=2, random_state=20260808).fit_transform(emb)
    fig, ax = plt.subplots(figsize=(6, 5))
    for src, color, marker in [("human", COLOR_HUMAN, "o"), ("ai", COLOR_AI, "^")]:
        mask = (df["source"] == src).values
        ax.scatter(coords[mask, 0], coords[mask, 1], c=color, marker=marker,
                   alpha=0.6, s=35, label=f"{src} (N={mask.sum()})", edgecolors="none")
    ax.set_xlabel("PCA dimension 1", fontsize=9)
    ax.set_ylabel("PCA dimension 2", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Figure 5. 2D PCA of song embeddings (descriptive only,\n"
                  "not used as inferential evidence -- see Methods)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure5_embedding_pca.png", dpi=200)
    fig.savefig(FIGURES / "figure5_embedding_pca.svg")
    plt.close(fig)
    print("Wrote figure5_embedding_pca.{png,svg}")


def main():
    figure1_design_schematic()

    df = require_combined_features()
    if df is None:
        print("\nBLOCKED (figures 2-5, tables 1-3): features_song_level.parquet has no "
              "AI arm yet. Only Figure 1 (no data dependency) was produced this run. "
              "Re-run this script after Phase 5/6/7 complete.")
        return

    table1_sample_characteristics(df)
    figure2_semantic_diversity(df)
    figure3_genre_heterogeneity()
    figure4_condition_comparison(df)
    figure5_embedding_viz(df)
    # Table 3 (genre interactions) is already present as the AI:C(genre) rows in
    # table2_primary_models.csv; not separately re-exported.


if __name__ == "__main__":
    main()
