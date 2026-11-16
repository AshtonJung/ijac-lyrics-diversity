"""
05_quality_control.py

QC on AI generations (Section 11 of CLAUDE.md). Reads data/generated/ai_lyrics_raw.jsonl
(written by src/04_generate_ai_lyrics.py) and data/processed/human_lyrics_labeled.parquet
(for the memorization screen against the human corpus actually used in this study).

Flags (per-row, non-exclusive):
  - api_error            : generation_status != "ok"
  - refusal_or_meta      : contains refusal/meta-commentary markers
  - length_deviation     : actual_word_count outside target +/- 40% (documented threshold,
                            wider than the +/-10-15% *targeting* band used at prompt
                            construction, because generation is expected to have more
                            variance than the target -- this is a QC flag, not a resample
                            trigger)
  - duplicate_output      : exact-duplicate generation_text within the AI corpus
  - dominant_chorus       : any single line repeated so often it exceeds a fixed share of
                            total lines
  - memorization_risk     : high n-gram overlap (8-gram Jaccard) against ANY human song in
                            data/processed/human_lyrics_labeled.parquet (not just the
                            matched one), computed on normalized text

This script does not delete rows. It writes qc_flags per row so Phase 8 can run the
primary model both with and without flagged rows, per Section 11's explicit
instruction ("rerun primary models both with and without them").

Not runnable end-to-end yet: depends on data/generated/ai_lyrics_raw.jsonl, which does
not exist until Phase 5 (AI generation) completes (blocked on ANTHROPIC_API_KEY as of
this writing). Written and ready in advance per project rules ("prepare the exact script
and continue every other task that can be completed").
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
GENERATED = ROOT / "data" / "generated"
OUTPUTS = ROOT / "outputs"

REFUSAL_MARKERS = [
    "i cannot help with that", "i cannot generate", "i can't help with that",
    "i can't generate", "i'm not able to help with", "i'm not able to write",
    "as an ai language model", "i'm sorry, but i", "i won't generate",
    "i will not generate",
]
META_MARKERS = [
    "here are the lyrics", "here's a song", "sure, here", "[note:", "disclaimer:",
]
# Real refusals/meta-commentary front-load the response (there is no lyric content yet
# to interrupt); genuine song lyrics can innocuously contain phrases like "I can't stop"
# or "I cannot lie" anywhere in the text. Only check the opening of the response, not
# the full text, to avoid flagging ordinary lyric content as a refusal.
REFUSAL_CHECK_WINDOW_CHARS = 150
LENGTH_DEVIATION_PCT = 0.40
DOMINANT_LINE_SHARE = 0.30
NGRAM_N = 8
MEMORIZATION_JACCARD_THRESHOLD = 0.5


def normalize(text: str) -> str:
    t = re.sub(r"[^a-z0-9'\s]", " ", text.lower())
    return re.sub(r"\s+", " ", t).strip()


def ngram_set(text: str, n: int) -> set:
    tokens = text.split()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)} if len(tokens) >= n else set()


def flag_refusal_meta(text: str) -> bool:
    low = text.lower()[:REFUSAL_CHECK_WINDOW_CHARS]
    return any(m in low for m in REFUSAL_MARKERS) or any(m in low for m in META_MARKERS)


def flag_dominant_chorus(text: str) -> bool:
    lines = [l.strip().lower() for l in text.split("\n") if l.strip()]
    if not lines:
        return False
    counts = pd.Series(lines).value_counts()
    return (counts.max() / len(lines)) > DOMINANT_LINE_SHARE if len(lines) > 0 else False


def main():
    ai_path = GENERATED / "ai_lyrics_raw.jsonl"
    if not ai_path.exists():
        print(f"BLOCKED: {ai_path} does not exist yet -- Phase 5 (generation) has not "
              "run (ANTHROPIC_API_KEY not set as of this writing). Nothing to QC. "
              "This script is ready to run as soon as generation completes.")
        return

    rows = [json.loads(l) for l in open(ai_path) if l.strip()]
    ai = pd.DataFrame(rows)

    human = pd.read_parquet(PROCESSED / "human_lyrics_labeled.parquet")
    human_ngrams = [ngram_set(t, NGRAM_N) for t in human["normalized_text"]]

    ai["qc_api_error"] = ai["generation_status"] != "ok"
    ai["qc_refusal_or_meta"] = ai["generation_text"].fillna("").apply(flag_refusal_meta)
    ai["qc_length_deviation"] = (
        (ai["actual_word_count"] - ai["target_word_count"]).abs()
        > LENGTH_DEVIATION_PCT * ai["target_word_count"]
    )
    ai["qc_duplicate_output"] = ai["generation_text"].duplicated(keep=False) & ai["generation_text"].notna()
    ai["qc_dominant_chorus"] = ai["generation_text"].fillna("").apply(flag_dominant_chorus)

    mem_flags = []
    for text in ai["generation_text"].fillna(""):
        norm = normalize(text)
        gset = ngram_set(norm, NGRAM_N)
        if not gset:
            mem_flags.append(False)
            continue
        max_jaccard = 0.0
        for hset in human_ngrams:
            if not hset:
                continue
            inter = len(gset & hset)
            union = len(gset | hset)
            if union:
                max_jaccard = max(max_jaccard, inter / union)
        mem_flags.append(max_jaccard > MEMORIZATION_JACCARD_THRESHOLD)
    ai["qc_memorization_risk"] = mem_flags

    flag_cols = [c for c in ai.columns if c.startswith("qc_")]
    ai["qc_any_flag"] = ai[flag_cols].any(axis=1)

    out_path = PROCESSED / "ai_lyrics_qc.parquet"
    ai.to_parquet(out_path, index=False)

    print(f"QC'd {len(ai)} generations.")
    for c in flag_cols:
        print(f"  {c}: {ai[c].sum()} flagged")
    print(f"  any flag: {ai['qc_any_flag'].sum()} / {len(ai)}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
