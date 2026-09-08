# Amazon Gift Cards Review — Emotion Classification

MBAX 6418, Assignment 1. A working classifier for Amazon Gift Cards reviews: sentiment
(binary, then three-class), scored against the star rating; primary emotion, detected two
independent ways (positive or negative); and a results dashboard tying it all together.

## Data source

Reviews come from [Amazon Reviews '23](https://amazon-reviews-2023.github.io) (McAuley Lab,
UC San Diego), **Gift Cards** category — 152,410 reviews, gzipped JSON Lines, downloaded
directly from
[mcauleylab.ucsd.edu/.../Gift_Cards.jsonl.gz](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Gift_Cards.jsonl.gz).
Not committed to this repo (large, and re-downloadable from that link); see `scripts/data.py`
for how it's read.

## Repo layout

| File | What it is |
|---|---|
| `scripts/prompt.py` | The classification prompts — binary (Step 1) and three-class (Step 6), plus the emotion extension (Step 5) |
| `scripts/classify_batch.py` | Scores the first 100 reviews (in file order) against the rating — Step 2 |
| `scripts/classify_balanced.py` | Builds the fixed-seed, 50/class balanced sample and scores it three-class — Step 6 |
| `scripts/nrc_emotion.py` | Word-list emotion scoring against the NRC lexicon — no model calls — Step 5 |
| `scripts/add_emotions.py` | Adds both emotion methods to a saved run and compares them — Step 5 |
| `scripts/generate_dashboard.py` | The dashboard generator — injects `runs/*.json` into `dashboard/template.html` |
| `dashboard/index.html` | The final dashboard — self-contained, works offline, no server needed |
| `runs/batch100.json` | Raw output: the first-100 binary run (with emotion enrichment) |
| `runs/balanced150.json` | Raw output: the balanced 150-review three-class run |
| `data/nrc_lexicon.json` | Vendored NRC Word-Emotion Association Lexicon (word → emotions) |

## Reproducing this

```
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
cp .env.example .env   # fill in your OpenAI-compatible endpoint + key
cd scripts
python3 classify_batch.py 100 batch100.json
python3 add_emotions.py batch100.json
python3 classify_balanced.py balanced150.json
python3 generate_dashboard.py
```

`classify_balanced.py` uses a fixed random seed (42) over the *whole* dataset, so it pulls
the same 150 reviews every time — see "reproducibility" note below.

## Results

### Binary, first 100 reviews (Step 2)

| | |
|---|---|
| Accuracy | **99%** (99/100) |
| True split | 93 POSITIVE / 7 NEGATIVE |
| POSITIVE recall | 100% (93/93) |
| NEGATIVE recall | 85.7% (6/7) |

### Three-class, balanced sample (Step 6)

50 reviews per class, pulled with a fixed seed from all 152,410 reviews (not the first N —
see Q1 below for why that matters).

| | |
|---|---|
| Accuracy | **71.3%** (107/150) |
| POSITIVE recall | 96% (48/50) |
| NEUTRAL recall | **22%** (11/50) |
| NEGATIVE recall | 96% (48/50) |

Confusion matrix (rows = true, columns = predicted):

| True \ Predicted | POSITIVE | NEUTRAL | NEGATIVE |
|---|---|---|---|
| **POSITIVE** | 48 | 2 | 0 |
| **NEUTRAL** | 13 | 11 | 26 |
| **NEGATIVE** | 1 | 1 | 48 |

### Emotion: LLM vs. NRC word list (Step 5, on the 100-row binary set)

| | |
|---|---|
| Reviews where both methods gave an answer | 85 / 100 (15 had no lexicon words at all) |
| Agreement | **23.5%** (20 / 85) |
| LLM's top emotion | joy — 89/100 reviews |
| Word list's top emotion | anticipation — 59/100 reviews |

## Screenshots

**First 100, binary run:**

![Dashboard — binary run](screenshots/dashboard-binary.png)

**Balanced 150, three-class run:**

![Dashboard — balanced 3-class run](screenshots/dashboard-balanced.png)

## Answers

### 1. Why did the lopsided run look very accurate, and what did sampling equal amounts of each class change?

The first test looked great — 99% accuracy — mostly just because of luck in the ordering.
93 of the first 100 reviews happened to be positive. So if the model had just guessed
positive every single time, it would have already scored 93% without learning anything. The
real model only beat that lazy guess by 6 points.

When you look across the whole dataset, it's skewed the same way: about 88% positive, 9%
negative, and 2% neutral.

Once I pulled a balanced sample (50 reviews from each category, randomly picked but seeded
for consistency), accuracy dropped to 71%. The positive and negative scores barely moved —
the entire drop came from neutral, which the model struggled with. Balancing didn't make the
model worse; it just stopped letting the easy majority class hide a weakness.

### 2. Look at where the model's mistakes go — which classes get confused with which, and in what direction?

Most of the model's mistakes are concentrated in one place: the neutral reviews. Out of the
50 reviews that were truly neutral, the model only got 11 correct. Of the ones it missed, 26
got called negative and only 13 got called positive — so when the model is wrong about
neutral, it leans toward calling it negative about twice as often as positive.

In the other direction, the model is actually pretty reliable on negative reviews — only 1
out of 50 true negative reviews got mistaken for neutral. This confirms the biggest error is
specifically neutral reviews getting framed as negative.

### 3. How do the LLM's emotions and the word list's emotions differ, and why?

They only agree about 1 in 4 times (23.5%). The AI mostly said the reviews felt like "joy"
(89 out of 100). The word-list method mostly said "anticipation" instead (59 out of 100).

There's a quirk in the language being used: common words in these reviews — "gift," "money,"
"good" — are tagged as both "anticipation" and "joy" at the same time. That tie has to be
broken, since the model is required to give one answer; my code broke ties alphabetically, so
"anticipation" beat "joy" almost every time.

Looking only at the reviews where the AI said "joy," the word list agreed on "joy" just 20
times, but called the same reviews "anticipation" 51 times. Bottom line: with the same
reviews and the same words, the disagreement is really about how the tie gets broken, not
about the two methods disagreeing on how the review actually feels.

### 4. What bugs and/or issues did you hit along the way, and how did you work around them?

1. **One specific review kept causing problems**, and it took two passes to actually fix.
   The first symptom was a crash: the AI's response for that review came back completely
   empty, and the code choked on it with a confusing error instead of a clear one. That got
   patched so it would fail with a readable error instead of crashing.

   But the review kept failing even after that fix — the readable error was just showing the
   same problem more clearly. The real cause turned out to be that this review needed a lot
   more "thinking" time from the model before it could answer, using way more tokens than a
   typical review. Even with the settings locked to make the model consistent every time
   (temperature = 0), the amount of "thinking" it did on this review wasn't perfectly steady
   between runs — a known quirk of how shared AI inference servers handle everyone's
   requests at once. The actual fix was giving the model a lot more room to think before
   giving up, rather than special-casing this one review.

   The specific review, for reference: a 5-star review titled "No note attached to sent gift
   card" — a genuinely mixed piece of text (glowing overall, but with a detailed embedded
   complaint about a missing note feature), which is likely why it needed more reasoning than
   a straightforwardly happy or unhappy review.

2. **A layout-bug risk in the dashboard.** The assignment specifically warned about chart
   bars that can shrink down to nothing because a text label crowds them out. This was
   avoided from the start by building every bar chart so the label always has its own
   locked-in space and can't squeeze the bar to zero.

3. **A color-contrast bug caught before shipping.** The star-rating bars initially used
   colors that were almost invisible against the light background. Swapped in colors that
   actually show up.

4. **An inefficient re-run process.** Early on, if one review failed while adding emotion
   data to all 100 reviews, fixing it meant re-running all 100. The script was made
   resumable — it now skips reviews that already succeeded — so a single failure costs one
   retry, not a full re-run.

## What's not in this repo, and why

- `Gift_Cards.jsonl.gz` — large (12MB) and re-downloadable from the source above.
- `.env` — holds the API endpoint/key; `.env.example` documents the shape.
- The assignment PDF — instructor's course material, not appropriate for a public repo.

## Reproducibility

Every number above comes from a fixed choice of data and settings: `classify_batch.py`
reads the first N rows in a fixed file; `classify_balanced.py` samples with
`random.Random(42)` over the full dataset; every model call uses `temperature=0`. Re-running
the scripts should reproduce the same reviews and (with the caveat in the bugs section
above) the same predictions.
