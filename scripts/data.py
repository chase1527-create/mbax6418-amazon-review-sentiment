"""Loads the Gift Cards review file (gzipped JSON Lines)."""
import gzip
import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "Gift_Cards.jsonl.gz"


def iter_reviews():
    with gzip.open(DATA_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def first_n(n: int) -> list[dict]:
    reviews = []
    for i, r in enumerate(iter_reviews()):
        if i >= n:
            break
        reviews.append(r)
    return reviews
