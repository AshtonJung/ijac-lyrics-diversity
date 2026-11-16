"""
04_generate_ai_lyrics.py (v3 -- real API, restores the original design)

Reproducible AI-lyric generation pipeline using the real Anthropic Messages API,
now that ANTHROPIC_API_KEY is available. This restores the originally planned design
from outputs/preanalysis_decisions.md (v1): one generation per matched prompt at each
of three temperatures (low/medium/high), logging model version, timestamp, temperature,
prompt_id, and generation status. Supersedes the v2 CLI-subprocess fallback
(outputs/preanalysis_decisions_v3.md) and the v4 in-session-authored pilot
(outputs/preanalysis_decisions_v4.md), both of which remain in
data/generated/ai_lyrics_raw.jsonl as a disclosed, separate cohort
(generation_method != "anthropic_api") rather than being deleted.

The key is read ONLY from the environment variable, is never printed, and is never
written to any file in this project.

Usage:
    python3 src/04_generate_ai_lyrics.py [--limit N] [--temperatures low,medium,high]
                                          [--concurrency 8]
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
GENERATED = ROOT / "data" / "generated"
LOGS = ROOT / "outputs" / "logs"

MODEL_NAME = "claude-sonnet-4-5-20250929"
TEMPERATURES = {"low": 0.2, "medium": 0.7, "high": 1.0}
MAX_RETRIES = 3
MAX_TOKENS = 1200

SYSTEM_PROMPT = (
    "You are a songwriting assistant helping with an academic research study on "
    "AI-generated song lyrics. Follow the user's structural constraints exactly. "
    "Do not add commentary, explanations, or disclaimers -- return only the lyrics."
)

write_lock = threading.Lock()


def load_existing(output_path: Path) -> set:
    done = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if row.get("generation_status") == "ok":
                        done.add((row["prompt_id"], row["condition"]))
                except json.JSONDecodeError:
                    continue
    return done


def generate_one(client, prompt_text: str, temperature: float):
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.messages.create(
                model=MODEL_NAME,
                max_tokens=MAX_TOKENS,
                temperature=temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt_text}],
            )
            text = "".join(b.text for b in resp.content if b.type == "text")
            return {
                "status": "ok", "text": text, "model_version": resp.model,
                "retry_count": attempt, "stop_reason": resp.stop_reason,
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            }
        except Exception as e:  # noqa: BLE001 - log and retry deliberately
            last_err = str(e)
            time.sleep(2 ** attempt)
    return {"status": "error", "text": None, "model_version": None,
             "retry_count": MAX_RETRIES, "error": last_err,
             "input_tokens": None, "output_tokens": None}


def process_one(client, row, t_label, out_f):
    result = generate_one(client, row["generation_prompt"], TEMPERATURES[t_label])
    record = {
        "prompt_id": row["prompt_id"],
        "matched_human_id": row["song_id"],
        "model": result.get("model_version") or MODEL_NAME,
        "generation_method": "anthropic_api",
        "condition": t_label,
        "temperature_value": TEMPERATURES[t_label],
        "session_id": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "genre": row["genre_broad"],
        "theme": row["theme"],
        "mood": row["mood"],
        "target_word_count": int(row["target_word_count"]),
        "actual_word_count": len(result["text"].split()) if result.get("text") else None,
        "generation_text": result.get("text"),
        "retry_count": result["retry_count"],
        "generation_status": result["status"],
        "error": result.get("error"),
        "cost_usd": None,  # computed post hoc from input/output token counts + published rates
        "input_tokens": result.get("input_tokens"),
        "output_tokens": result.get("output_tokens"),
    }
    with write_lock:
        out_f.write(json.dumps(record) + "\n")
        out_f.flush()
    return row["prompt_id"], t_label, result["status"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperatures", type=str, default="low,medium,high")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY is not set. This script will not invent access.")
        sys.exit(1)

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    prompts = pd.read_csv(PROCESSED / "generation_prompts.csv")
    if args.limit:
        prompts = prompts.head(args.limit)

    temp_labels = [t.strip() for t in args.temperatures.split(",") if t.strip()]
    for t in temp_labels:
        assert t in TEMPERATURES, f"unknown temperature label {t}"

    GENERATED.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    out_path = GENERATED / "ai_lyrics_raw.jsonl"
    done = load_existing(out_path)

    jobs = [(row, t) for _, row in prompts.iterrows() for t in temp_labels
            if (row["prompt_id"], t) not in done]
    print(f"Planned calls: {len(prompts)} prompts x {len(temp_labels)} temperatures "
          f"= {len(prompts) * len(temp_labels)}; already done: {len(done)}; "
          f"remaining this run: {len(jobs)}")

    n_ok, n_err = 0, 0
    with open(out_path, "a") as out_f:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(process_one, client, row, t, out_f) for row, t in jobs]
            for i, fut in enumerate(as_completed(futures), 1):
                pid, t_label, status = fut.result()
                if status == "ok":
                    n_ok += 1
                else:
                    n_err += 1
                if i % 20 == 0 or i == len(jobs):
                    print(f"[{i}/{len(jobs)}] ok={n_ok} err={n_err}")

    print(f"Done. ok={n_ok} err={n_err}. Appended to {out_path}")


if __name__ == "__main__":
    main()
