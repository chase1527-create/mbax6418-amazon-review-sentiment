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

Respond with JSON only, in exactly this shape, no other text:
{"sentiment": "POSITIVE"}
or
{"sentiment": "NEGATIVE"}
"""


def build_messages(title: str, text: str) -> list[dict]:
    return [
        {"role": "system", "content": SENTIMENT_SYSTEM_PROMPT},
        {"role": "user", "content": f"TITLE: {title}\nTEXT: {text}"},
    ]


def parse_sentiment(raw: str) -> str:
    """Extract POSITIVE/NEGATIVE from a model response. Raises ValueError if unparseable."""
    try:
        label = json.loads(raw)["sentiment"].strip().upper()
    except (json.JSONDecodeError, KeyError, AttributeError, TypeError):
        # Fallback for a model that ignores the JSON instruction.
        upper = raw.upper()
        if "POSITIVE" in upper and "NEGATIVE" not in upper:
            label = "POSITIVE"
        elif "NEGATIVE" in upper and "POSITIVE" not in upper:
            label = "NEGATIVE"
        else:
            raise ValueError(f"Could not parse sentiment from: {raw!r}")
    if label not in ("POSITIVE", "NEGATIVE"):
        raise ValueError(f"Unexpected sentiment value: {label!r}")
    return label


def classify_review(title: str, text: str, max_tokens: int = 300) -> str:
    """Classify one review. Retries once with more headroom if the model got
    cut off mid-reasoning before it could answer (finish_reason == 'length')."""
    messages = build_messages(title, text)
    for attempt_tokens in (max_tokens, max_tokens * 3):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,
            max_tokens=attempt_tokens,
            response_format={"type": "json_object"},
        )
        choice = resp.choices[0]
        if choice.finish_reason == "length" and attempt_tokens == max_tokens:
            continue  # retry with more room instead of parsing a truncated answer
        return parse_sentiment(choice.message.content)
    raise ValueError(f"Model never finished answering for: {title!r}")


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
