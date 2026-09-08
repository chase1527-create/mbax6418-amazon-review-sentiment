"""Step 2: classify a batch of reviews and score against the star rating.

The rating is only ever used here, after the fact, to check the model —
classify_review() (Step 1) never sees it.
"""
import json
import sys
import time
from pathlib import Path

from data import first_n
from labels import true_label_binary
from prompt import classify_review

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
LABELS = ["POSITIVE", "NEGATIVE"]


def classify_with_retry(title: str, text: str, attempts: int = 2) -> tuple[str | None, str | None]:
    last_error = None
    for _ in range(attempts):
        try:
            return classify_review(title, text), None
        except Exception as e:  # noqa: BLE001 - want to record and move on, not crash a 100-row run
            last_error = str(e)
            time.sleep(1)
    return None, last_error


def run_batch(n: int) -> list[dict]:
    reviews = first_n(n)
    records = []
    for i, r in enumerate(reviews):
        true = true_label_binary(r["rating"])
        pred, error = classify_with_retry(r["title"], r["text"])
        records.append({
            "index": i,
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
    class_counts = {l: sum(1 for r in records if r["true_label"] == l) for l in LABELS}
    confusion = {t: {p: 0 for p in LABELS} for t in LABELS}
    for r in scored:
        confusion[r["true_label"]][r["predicted_label"]] += 1
    per_class_recall = {
        l: (confusion[l][l] / class_counts[l]) if class_counts[l] else None
        for l in LABELS
    }
    return {
        "n": n,
        "n_scored": len(scored),
        "n_errors": n - len(scored),
        "accuracy": correct / len(scored) if scored else None,
        "class_distribution_true": class_counts,
        "confusion_matrix": confusion,
        "per_class_recall": per_class_recall,
    }


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    out_name = sys.argv[2] if len(sys.argv) > 2 else "batch100.json"

    print(f"Classifying first {n} reviews...", file=sys.stderr)
    records = run_batch(n)
    summary = summarize(records)

    RUNS_DIR.mkdir(exist_ok=True)
    out_path = RUNS_DIR / out_name
    out_path.write_text(json.dumps({"summary": summary, "records": records}, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {out_path}", file=sys.stderr)
