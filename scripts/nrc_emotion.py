"""Step 5b: NRC word-list emotion scoring — no model calls.

Uses the NRC Word-Emotion Association Lexicon (Mohammad & Turney, 2013,
"Crowdsourcing a Word-Emotion Association Lexicon"), vendored at
data/nrc_lexicon.json (word -> list of associated emotions/sentiments).
Free for research use; cite the paper if this is reused elsewhere.

For each review: tokenize title+text, look up each word's emotions in the
lexicon, sum counts per emotion across all matched words, and take the
highest-scoring emotion as the primary one.
"""
import json
import re
from pathlib import Path

LEXICON_PATH = Path(__file__).resolve().parent.parent / "data" / "nrc_lexicon.json"
EMOTIONS = ("anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust")
TOKEN_RE = re.compile(r"[a-z']+")

_lexicon = None


def _load_lexicon() -> dict:
    global _lexicon
    if _lexicon is None:
        raw = json.loads(LEXICON_PATH.read_text())
        _lexicon = {word: [e for e in emos if e in EMOTIONS] for word, emos in raw.items()}
    return _lexicon


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def score_emotions(title: str, text: str) -> dict:
    lexicon = _load_lexicon()
    scores = {e: 0 for e in EMOTIONS}
    for word in tokenize(f"{title} {text}"):
        for emo in lexicon.get(word, ()):
            scores[emo] += 1
    return scores


def primary_emotion(title: str, text: str) -> tuple:
    """Returns (emotion_or_None, scores). None means no lexicon word matched."""
    scores = score_emotions(title, text)
    total = sum(scores.values())
    if total == 0:
        return None, scores
    top = max(scores.values())
    winners = sorted(e for e, s in scores.items() if s == top)
    return winners[0], scores  # deterministic tie-break: alphabetical


if __name__ == "__main__":
    cases = [
        ("Great gift", "Having Amazon money is always good."),
        ("Terrible", "Card did not work, waste of money, I am furious."),
        ("Scam", "Charged me and the code was already used by someone else."),
    ]
    for title, text in cases:
        emo, scores = primary_emotion(title, text)
        print(f"{title!r:20} -> {emo} {scores}")
