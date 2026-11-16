"""
03_build_prompt_metadata.py

Assigns theme, mood, and narrative perspective to each cleaned human song using a
documented, reproducible lexicon/rule-based classifier (no ML training, no manual
per-song judgment calls beyond a post-hoc QA sample). Then builds the high-level,
lyric-leakage-free AI generation prompt for each song, per
outputs/preanalysis_decisions_v2.md Section 2-3.

Inputs : data/processed/human_lyrics_clean.parquet
Outputs: data/processed/human_lyrics_labeled.parquet
         data/processed/generation_prompts.csv
         outputs/theme_mood_manual_qa_sample.csv  (20-song sample for researcher review)
         outputs/logs/03_build_prompt_metadata.log
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
RANDOM_SEED = 20260808

# --- Controlled vocabularies (Section 9 of CLAUDE.md) -----------------------------------
THEME_KEYWORDS = {
    "romantic love": ["love", "kiss", "heart", "darling", "baby i", "in love", "romance",
                       "sweetheart", "adore", "forever yours"],
    "breakup/loss": ["goodbye", "broke my heart", "breakup", "break up", "left me", "over now",
                      "used to love", "walked away", "empty bed", "miss you", "let you go"],
    "friendship": ["my friend", "friends forever", "homie", "ride or die", "got my back",
                    "brotherhood", "sisterhood", "squad"],
    "ambition/success": ["grind", "hustle", "made it", "on top", "money", "success", "dream big",
                          "rich", "ceo", "empire", "the top"],
    "identity/self-reflection": ["who i am", "myself", "mirror", "looking inside", "my own skin",
                                  "self", "identity", "who i really am", "figure myself out"],
    "nostalgia": ["remember when", "back then", "those days", "childhood", "old photograph",
                  "throwback", "used to be", "way back"],
    "family/home": ["mama", "papa", "father", "mother", "family", "hometown", "back home",
                     "my brother", "my sister", "grew up"],
    "social conflict": ["fight the power", "system", "protest", "injustice", "the streets",
                         "war", "enemy", "battle", "struggle against"],
    "celebration": ["party", "celebrate", "dance floor", "turn up", "let's go", "tonight we",
                     "champagne", "club", "all night long"],
    "loneliness": ["alone", "lonely", "by myself", "empty room", "nobody here", "isolated",
                    "no one calls"],
    "hope/recovery": ["get better", "heal", "rise up", "keep going", "brighter days", "survive",
                       "recovery", "stronger now", "will be okay", "hope"],
}

MOOD_KEYWORDS = {
    "joyful": ["happy", "joy", "smile", "laughing", "sunshine", "great day"],
    "sad": ["cry", "tears", "sad", "hurt", "pain", "broken"],
    "angry": ["mad", "angry", "rage", "hate you", "furious", "damn you"],
    "reflective": ["think about", "wonder", "used to", "looking back", "realize"],
    "hopeful": ["hope", "believe", "will be fine", "someday", "future", "better days"],
    "anxious": ["afraid", "scared", "worry", "anxious", "nervous", "can't sleep"],
    "energetic": ["let's go", "turn up", "jump", "run", "loud", "fast", "up all night"],
    "calm": ["calm", "quiet", "peace", "slow down", "still", "gentle"],
}

FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours"}
SECOND_PERSON = {"you", "your", "yours", "yourself"}
THIRD_PERSON = {"he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs"}


def score_lexicon(text: str, lexicon: dict) -> dict:
    scores = {}
    for label, keywords in lexicon.items():
        scores[label] = sum(text.count(k) for k in keywords)
    return scores


def top_label(scores: dict, priority_order: list, fallback: str) -> str:
    max_val = max(scores.values()) if scores else 0
    if max_val == 0:
        return fallback
    tied = [k for k, v in scores.items() if v == max_val]
    for label in priority_order:
        if label in tied:
            return label
    return tied[0]


def narrative_perspective(text: str) -> str:
    tokens = re.findall(r"[a-z']+", text)
    c1 = sum(1 for t in tokens if t in FIRST_PERSON)
    c2 = sum(1 for t in tokens if t in SECOND_PERSON)
    c3 = sum(1 for t in tokens if t in THIRD_PERSON)
    counts = {"first person": c1, "second person": c2, "third person": c3}
    return top_label(counts, ["first person", "second person", "third person"], "first person")


def build_prompt(row: pd.Series) -> str:
    return (
        "Write original song lyrics with the following characteristics:\n"
        f"Genre: {row['genre_broad']}\n"
        f"Broad theme: {row['theme']}\n"
        f"Mood: {row['mood']}\n"
        f"Narrative perspective: {row['narrative_perspective']}\n"
        f"Approximate length: {row['target_word_count']} words\n"
        "Use a conventional verse/chorus structure.\n"
        "Do not imitate any named artist or existing song.\n"
        "Avoid quoting or paraphrasing known lyrics.\n"
        "Return lyrics only."
    )


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROCESSED / "human_lyrics_clean.parquet")

    theme_priority = list(THEME_KEYWORDS.keys())
    mood_priority = list(MOOD_KEYWORDS.keys())

    themes, moods, perspectives = [], [], []
    theme_scores_all, mood_scores_all = [], []
    for text in df["normalized_text"]:
        t_scores = score_lexicon(text, THEME_KEYWORDS)
        m_scores = score_lexicon(text, MOOD_KEYWORDS)
        themes.append(top_label(t_scores, theme_priority, fallback="identity/self-reflection"))
        moods.append(top_label(m_scores, mood_priority, fallback="reflective"))
        perspectives.append(narrative_perspective(text))
        theme_scores_all.append(max(t_scores.values()) if t_scores else 0)
        mood_scores_all.append(max(m_scores.values()) if m_scores else 0)

    df["theme"] = themes
    df["theme_top_score"] = theme_scores_all
    df["mood"] = moods
    df["mood_top_score"] = mood_scores_all
    df["narrative_perspective"] = perspectives

    # target_word_count = human word count, per Section 9 length-matching rule
    df["target_word_count"] = df["word_count"]
    df["prompt_id"] = "P" + df["song_id"].astype(str)

    df["generation_prompt"] = df.apply(build_prompt, axis=1)

    # Save labeled dataset (still contains analysis_text/normalized_text — internal only,
    # never to be exported into paper/figures/logs verbatim)
    df.to_parquet(PROCESSED / "human_lyrics_labeled.parquet", index=False)

    # Save the prompt table WITHOUT any lyric text -- this is what actually leaves the
    # human-corpus boundary and goes to the generation step.
    prompt_cols = ["prompt_id", "song_id", "genre_broad", "theme", "mood",
                    "narrative_perspective", "target_word_count", "generation_prompt"]
    df[prompt_cols].to_csv(PROCESSED / "generation_prompts.csv", index=False)

    # 20-song random QA sample for the researcher to manually check theme/mood assignment
    # against the actual song (title/artist only, no lyric text, so this file is safe to
    # open without exposing copyrighted text).
    qa_sample = df.sample(n=min(20, len(df)), random_state=RANDOM_SEED)[
        ["song_id", "song_name", "artist_name", "genre_broad", "theme", "mood",
         "narrative_perspective", "theme_top_score", "mood_top_score"]
    ].copy()
    qa_sample["researcher_agrees_theme_y_n"] = ""
    qa_sample["researcher_agrees_mood_y_n"] = ""
    qa_sample["researcher_notes"] = ""
    qa_sample.to_csv(OUTPUTS / "theme_mood_manual_qa_sample.csv", index=False)

    print("Theme distribution:\n", df["theme"].value_counts())
    print("\nMood distribution:\n", df["mood"].value_counts())
    print("\nNarrative perspective distribution:\n", df["narrative_perspective"].value_counts())
    print("\nZero-score (fallback-label) rate: theme=%.1f%%, mood=%.1f%%" % (
        100 * (df["theme_top_score"] == 0).mean(), 100 * (df["mood_top_score"] == 0).mean()))
    print(f"\nWrote {len(df)} labeled rows, {len(df)} prompts, 20-row QA sample.")


if __name__ == "__main__":
    main()
