"""Turns a star rating into the 'correct answer' the model is scored against.

The model never sees this — it's derived from the rating purely for scoring.
"""


def true_label_binary(rating: float) -> str:
    return "POSITIVE" if rating >= 4 else "NEGATIVE"


def true_label_three(rating: float) -> str:
    if rating >= 4:
        return "POSITIVE"
    if rating == 3:
        return "NEUTRAL"
    return "NEGATIVE"
