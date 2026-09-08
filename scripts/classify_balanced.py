"""Step 6: three-class scoring with a balanced sample.

Reading the first N rows in order under-represents NEUTRAL (3-star is only
~2% of the whole file) and NEGATIVE, so this pulls a fixed-seed balanced
sample from the WHOLE file instead - PER_CLASS reviews from each of
POSITIVE/NEUTRAL/NEGATIVE, same set every run.
"""
import json
import random
import sys
import time
from pathlib import Path

from data import iter_reviews
from labels import true_label_three
from prompt import classify_review_three

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
CLASSES = ("POSITIVE", "NEUTRAL", "NEGATIVE")
SEED = 42
PER_CLASS = 50


def build_balanced_sample(per_class: int = PER_CLASS, seed: int = SEED) -> list[dict]:
    buckets = {c: [] for c in CLASSES}
    for idx, r in enumerate(iter_reviews()):
        buckets[true_label_three(r["rating"])].append((idx, r))

    rng = random.Random(seed)
    sample = []
    for c in CLASSES:
        pool = list(buckets[c])
        rng.shuffle(pool)
        sample.extend(pool[:per_class])
    sample.sort(key=lambda pair: pair[0])  # stable, readable order
    return [{"orig_index": idx, **r} for idx, r in sample]


def classify_with_retry(title: str, text: str, attempts: int = 3) -> tuple:
    last_error = None
    for i in range(attempts):
        try:
            return classify_review_three(title, text), None
        except Exception as e:  # noqa: BLE001
            last_error = str(e)
            time.sleep(2 * (i + 1))
    return None, last_error


def run(per_class: int = PER_CLASS, seed: int = SEED) -> list[dict]:
    sample = build_balanced_sample(per_class, seed)
    n = len(sample)
    records = []
    for i, r in enumerate(sample):
        true = true_label_three(r["rating"])
        pred, error = classify_with_retry(r["title"], r["text"])
        records.append({
            "index": i,
            "orig_index": r["orig_index"],
            "asin": r.get("asin"),
            "rating": r["rating"],
            "title": r["title"],
            "text": r["text"],
            "true_label": true,
            "predicted_label": pred,
            "correct": pred == true,
            "error": error,
        })
        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"  {i + 1}/{n}", file=sys.stderr)
    return records


def summarize(records: list[dict]) -> dict:
    n = len(records)
    scored = [r for r in records if r["predicted_label"] is not None]
    correct = sum(r["correct"] for r in scored)
    class_counts = {l: sum(1 for r in records if r["true_label"] == l) for l in CLASSES}
    confusion = {t: {p: 0 for p in CLASSES} for t in CLASSES}
    for r in scored:
        confusion[r["true_label"]][r["predicted_label"]] += 1
    per_class_recall = {
        l: (confusion[l][l] / class_counts[l]) if class_counts[l] else None
        for l in CLASSES
    }
    return {
        "n": n,
        "n_scored": len(scored),
        "n_errors": n - len(scored),
        "seed": SEED,
        "per_class_target": PER_CLASS,
        "accuracy": correct / len(scored) if scored else None,
        "class_distribution_true": class_counts,
        "confusion_matrix": confusion,
        "per_class_recall": per_class_recall,
    }


if __name__ == "__main__":
    out_name = sys.argv[1] if len(sys.argv) > 1 else "balanced150.json"
    print(f"Building balanced sample (seed={SEED}, {PER_CLASS}/class) and classifying...", file=sys.stderr)
    records = run()
    summary = summarize(records)

    RUNS_DIR.mkdir(exist_ok=True)
    out_path = RUNS_DIR / out_name
    out_path.write_text(json.dumps({"summary": summary, "records": records}, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {out_path}", file=sys.stderr)
