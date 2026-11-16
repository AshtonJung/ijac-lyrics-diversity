# Pre-Analysis Decision Record — v5 (real, full-scale API generation)

Locked: 2026-08-09. Amends v1-v4, all kept unmodified. **This is the operative version
for the primary confirmatory analysis from this point forward.**

## What changed

The researcher obtained a real `ANTHROPIC_API_KEY` from console.anthropic.com and
provided it directly. `src/04_generate_ai_lyrics.py` was rewritten a third time (v3 of
the script, distinct from the v2/v3 decision-record numbering) to call the Anthropic
Messages API directly via the official Python SDK, restoring the **original v1 design**:
one generation per matched prompt at each of three temperatures (low=0.2, medium=0.7,
high=1.0), with real per-call model version, token usage, and status logged.

## What actually completed

213 prompts x 3 temperatures = 639 planned calls. **628 succeeded (98.3%), 11 failed**
with `invalid_request_error: Your credit balance is too low to access the Anthropic
API` -- the researcher's newly funded API credit balance was exhausted near the end of
the run, not a code or design failure. The 11 failures are logged with
`generation_status = "error"` and their exact error text in
`data/generated/ai_lyrics_raw.jsonl` -- not silently dropped. `src/04_generate_ai_lyrics.py`
is resumable and will pick up exactly these 11 (and only these 11) if re-run after the
balance is topped up.

## Sample composition for the primary confirmatory analysis

`data/generated/ai_lyrics_raw.jsonl` now contains three disclosed cohorts, distinguished
by `generation_method`:
- `anthropic_api`: 628 rows (**primary AI arm for this analysis**), real Messages API
  calls, `model = claude-sonnet-4-5-20250929`, real `temperature` per condition
  (low/medium/high), real token usage.
- `claude_code_cli_headless`: 10 rows (v3 fallback pilot, kept as historical record).
- `claude_in_session_authored`: 46 rows (v4 reduced-scope pilot, kept as historical
  record, now clearly superseded).

**The primary analysis in this version uses only the 628 `anthropic_api` rows**, joined
to the full N=213 human corpus. The two earlier pilot cohorts (56 rows total) are
excluded from the confirmatory feature set and are instead discussed in the manuscript
as a methods/transparency case study (Section 3.3/Appendix), not blended into the
primary result. This mirrors the same logic already used in v4 (excluding leftover
CLI rows outside the locked pilot subset) applied to the new, much larger real cohort.

## Consequence for design completeness

This restores the originally planned three-level temperature manipulation (RQ3/H3) in
full, superseding the "standard vs. diversity-oriented" prompt-wording fallback used in
v3/v4. Per-genre coverage: every one of the 213 human songs now has up to 3 real AI
counterparts (2-3 depending on which of the 11 failures landed where); exact per-song,
per-temperature completeness is enumerated in the rebuilt feature set.

## Everything else unchanged

Human corpus (N=213), theme/mood/perspective labeling, outcome metrics (semantic
dispersion via distance-to-genre-centroid, MTLD/HD-D), confirmatory model form
(now restoring the Temperature term dropped in v4), and multiple-comparison strategy
remain as locked in v1/v2.
