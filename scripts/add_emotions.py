"""Step 5: add both emotion takes to a saved run and compare them.

1. LLM emotion — re-calls the (now emotion-extended) Step 1 prompt over the
   same reviews. Also re-checks sentiment: since temperature=0, it should
   reproduce the original Step 2 sentiment exactly — a good consistency
   check on the model, worth reporting either way.
2. Word-list emotion — nrc_emotion.py, no model calls, runs over the text
   already saved in the run.

Adds emotion_llm / emotion_wordlist (+ raw word-list scores) to each record,
and an "emotion_comparison" block to the summary.
"""
import json
import sys
import time
from pathlib import Path

from nrc_emotion import primary_emotion
from prompt import classify_review_full

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
EMOTIONS = ("anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust")


def llm_emotion_with_retry(title: str, text: str, attempts: int = 4) -> tuple:
    last_error = None
    for i in range(attempts):
        try:
            result = classify_review_full(title, text)
            return result["sentiment"], result["emotion"], None
        except Exception as e:  # noqa: BLE001
            last_error = str(e)
            time.sleep(2 * (i + 1))  # backoff: transient server load, not a token-budget problem
    return None, None, last_error


def enrich(run_path: Path) -> dict:
    data = json.loads(run_path.read_text())
    records = data["records"]
    n = len(records)
    sentiment_mismatches = 0

    for i, r in enumerate(records):
        if r.get("emotion_llm") and not r.get("emotion_llm_error"):
            sentiment_recheck = r.get("sentiment_llm_recheck")  # already enriched — skip the API call
        else:
            sentiment_recheck, emotion_llm, error = llm_emotion_with_retry(r["title"], r["text"])
            r["sentiment_llm_recheck"] = sentiment_recheck
            r["emotion_llm"] = emotion_llm
            r["emotion_llm_error"] = error

        emotion_wl, scores_wl = primary_emotion(r["title"], r["text"])
        r["emotion_wordlist"] = emotion_wl
        r["emotion_wordlist_scores"] = scores_wl

        if sentiment_recheck is not None and r.get("predicted_label") is not None and sentiment_recheck != r["predicted_label"]:
            sentiment_mismatches += 1

        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"  {i + 1}/{n}", file=sys.stderr)

    both_present = [r for r in records if r["emotion_llm"] and r["emotion_wordlist"]]
    agree = sum(1 for r in both_present if r["emotion_llm"] == r["emotion_wordlist"])
    no_wordlist_match = sum(1 for r in records if r["emotion_wordlist"] is None)

    confusion = {a: {b: 0 for b in EMOTIONS} for a in EMOTIONS}
    for r in both_present:
        confusion[r["emotion_llm"]][r["emotion_wordlist"]] += 1

    llm_dist = {e: sum(1 for r in records if r["emotion_llm"] == e) for e in EMOTIONS}
    wl_dist = {e: sum(1 for r in records if r["emotion_wordlist"] == e) for e in EMOTIONS}

    data["summary"]["emotion_comparison"] = {
        "n": n,
        "n_both_present": len(both_present),
        "n_wordlist_no_match": no_wordlist_match,
        "agreement_rate": agree / len(both_present) if both_present else None,
        "sentiment_recheck_mismatches": sentiment_mismatches,
        "llm_emotion_distribution": llm_dist,
        "wordlist_emotion_distribution": wl_dist,
        "llm_vs_wordlist_confusion": confusion,
    }
    return data


if __name__ == "__main__":
    run_name = sys.argv[1] if len(sys.argv) > 1 else "batch100.json"
    path = RUNS_DIR / run_name
    print(f"Adding emotions to {path}...", file=sys.stderr)
    data = enrich(path)
    path.write_text(json.dumps(data, indent=2))
    print(json.dumps(data["summary"]["emotion_comparison"], indent=2))
    print(f"\nUpdated {path}", file=sys.stderr)
