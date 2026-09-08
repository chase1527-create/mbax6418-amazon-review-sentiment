"""Step 1: reusable sentiment-classification prompt.

Given a review's title and text (never its rating), asks the model for a
POSITIVE / NEGATIVE verdict and returns a clean, programmatic answer.
"""
import json

from llm_client import client, MODEL

SENTIMENT_SYSTEM_PROMPT = """You are classifying Amazon product reviews by sentiment.

You will be given a review's TITLE and TEXT. Decide whether the reviewer's \
overall sentiment toward their purchase was POSITIVE or NEGATIVE.

Rules:
- Judge only from the title and text given. No star rating is provided, and \
none should factor into your answer.
- If the title and text seem to disagree (e.g. an angry title on an \
otherwise satisfied review, or vice versa), trust the text over the title — \
it carries more detail about the reviewer's actual experience.
- Short or terse reviews still get a real answer: judge tone and word choice \
as best you can. A curt "works" or "fine" with no complaint leans POSITIVE; \
a curt complaint or profanity leans NEGATIVE.
- Judge the reviewer's overall experience with the purchase — a minor gripe \
inside an otherwise satisfied review is still POSITIVE.
- There is no NEUTRAL option here. Every review gets one answer. If it is \
genuinely a toss-up, make your best judgment call rather than refusing to \
decide.

Also identify the PRIMARY emotion the reviewer is expressing, from exactly
this list: anger, anticipation, disgust, fear, joy, sadness, surprise, trust.
Pick the single strongest one present, even if it's mild — every review gets
exactly one, lowercase.

Respond with JSON only, in exactly this shape, no other text:
{"sentiment": "POSITIVE", "emotion": "joy"}
or
{"sentiment": "NEGATIVE", "emotion": "anger"}
(sentiment is always POSITIVE or NEGATIVE; emotion is always one of the eight
listed above.)
"""

EMOTIONS = ("anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust")


def build_messages(title: str, text: str) -> list[dict]:
    return [
        {"role": "system", "content": SENTIMENT_SYSTEM_PROMPT},
        {"role": "user", "content": f"TITLE: {title}\nTEXT: {text}"},
    ]


def parse_sentiment(raw: str) -> str:
    """Extract POSITIVE/NEGATIVE from a model response. Raises ValueError if unparseable."""
    return parse_full(raw)["sentiment"]


def parse_full(raw: str | None) -> dict:
    """Extract {sentiment, emotion} from a model response. Raises ValueError if unparseable."""
    if not raw:
        raise ValueError("Model returned no content (empty or truncated response)")
    try:
        obj = json.loads(raw)
        sentiment = obj["sentiment"].strip().upper()
        emotion = obj["emotion"].strip().lower()
    except (json.JSONDecodeError, KeyError, AttributeError, TypeError):
        upper = raw.upper()
        if "POSITIVE" in upper and "NEGATIVE" not in upper:
            sentiment = "POSITIVE"
        elif "NEGATIVE" in upper and "POSITIVE" not in upper:
            sentiment = "NEGATIVE"
        else:
            raise ValueError(f"Could not parse sentiment from: {raw!r}")
        emotion = next((e for e in EMOTIONS if e in raw.lower()), None)
        if emotion is None:
            raise ValueError(f"Could not parse emotion from: {raw!r}")
    if sentiment not in ("POSITIVE", "NEGATIVE"):
        raise ValueError(f"Unexpected sentiment value: {sentiment!r}")
    if emotion not in EMOTIONS:
        raise ValueError(f"Unexpected emotion value: {emotion!r}")
    return {"sentiment": sentiment, "emotion": emotion}


def _call_with_retry(messages: list[dict], max_tokens: int = 300) -> str | None:
    """Calls the model in JSON mode, escalating max_tokens if the model's
    internal reasoning runs long enough to get truncated before it can
    answer (observed: one review needed >900 tokens though most need <200,
    and the exact length isn't perfectly stable run-to-run at temperature=0
    - continuous-batching float non-determinism on the shared server).
    Escalates well past the common case rather than special-casing outliers."""
    ladder = (max_tokens, max_tokens * 3, max_tokens * 12)
    for attempt_tokens in ladder:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,
            max_tokens=attempt_tokens,
            response_format={"type": "json_object"},
        )
        choice = resp.choices[0]
        if choice.finish_reason == "length" and attempt_tokens != ladder[-1]:
            continue  # retry with more room instead of parsing a truncated answer
        return choice.message.content
    return None  # exhausted the ladder; caller's parser raises a clear error on None


def classify_review_full(title: str, text: str, max_tokens: int = 300) -> dict:
    """Classify one review's sentiment + primary emotion in a single call."""
    raw = _call_with_retry(build_messages(title, text), max_tokens)
    return parse_full(raw)


def classify_review(title: str, text: str, max_tokens: int = 300) -> str:
    """Sentiment-only convenience wrapper (Steps 1-2)."""
    return classify_review_full(title, text, max_tokens)["sentiment"]


# --- Step 6: three-class (POSITIVE / NEUTRAL / NEGATIVE) ---

THREE_CLASS_SYSTEM_PROMPT = """You are classifying Amazon product reviews by sentiment.

You will be given a review's TITLE and TEXT. Decide whether the reviewer's \
overall sentiment toward their purchase was POSITIVE, NEUTRAL, or NEGATIVE.

Rules:
- Judge only from the title and text given. No star rating is provided, and \
none should factor into your answer.
- If the title and text seem to disagree, trust the text over the title — \
it carries more detail about the reviewer's actual experience.
- POSITIVE: the reviewer was satisfied or happy with their purchase, even if \
there's a minor gripe inside an otherwise positive review.
- NEGATIVE: the reviewer was dissatisfied or upset; a real complaint \
dominates the review.
- NEUTRAL: reserve this for reviews that are genuinely mixed (real pros AND \
real cons, roughly balanced) or plainly lukewarm/indifferent - "it's fine," \
"does what it says," nothing enthusiastic or upset. NEUTRAL is a real, \
deliberate answer for a middling review - not a shortcut for "I'm not sure." \
Only use it when the review itself reads as middling, not whenever a review \
is hard to judge; a hard-to-judge review still leans POSITIVE or NEGATIVE \
based on your best reading.
- Short or terse reviews still get a real answer.
- Every review gets exactly one of the three answers.

Respond with JSON only, in exactly this shape, no other text:
{"sentiment": "POSITIVE"}
or
{"sentiment": "NEUTRAL"}
or
{"sentiment": "NEGATIVE"}
"""

THREE_CLASSES = ("POSITIVE", "NEUTRAL", "NEGATIVE")


def build_messages_three(title: str, text: str) -> list[dict]:
    return [
        {"role": "system", "content": THREE_CLASS_SYSTEM_PROMPT},
        {"role": "user", "content": f"TITLE: {title}\nTEXT: {text}"},
    ]


def parse_sentiment_three(raw: str | None) -> str:
    if not raw:
        raise ValueError("Model returned no content (empty or truncated response)")
    try:
        label = json.loads(raw)["sentiment"].strip().upper()
    except (json.JSONDecodeError, KeyError, AttributeError, TypeError):
        upper = raw.upper()
        hits = [c for c in THREE_CLASSES if c in upper]
        if len(hits) != 1:
            raise ValueError(f"Could not parse sentiment from: {raw!r}")
        label = hits[0]
    if label not in THREE_CLASSES:
        raise ValueError(f"Unexpected sentiment value: {label!r}")
    return label


def classify_review_three(title: str, text: str, max_tokens: int = 300) -> str:
    """Classify one review as POSITIVE / NEUTRAL / NEGATIVE (Step 6)."""
    raw = _call_with_retry(build_messages_three(title, text), max_tokens)
    return parse_sentiment_three(raw)


if __name__ == "__main__":
    # Quick spot-check: obviously positive and obviously negative reviews.
    cases = [
        ("Great gift", "Having Amazon money is always good.", "POSITIVE"),
        ("Terrible", "Card did not work, waste of money.", "NEGATIVE"),
        ("Love it!", "Exactly what I wanted, arrived fast, five stars.", "POSITIVE"),
        ("Scam", "Charged me and the code was already used by someone else.", "NEGATIVE"),
        ("meh", "works", "POSITIVE"),
        ("Disappointed", "Not what I expected but I guess it's fine, whatever.", None),
    ]
    for title, text, expected in cases:
        got = classify_review(title, text)
        mark = "?" if expected is None else ("OK" if got == expected else "MISMATCH")
        print(f"[{mark}] {title!r:35} -> {got:9} (expected: {expected})")
