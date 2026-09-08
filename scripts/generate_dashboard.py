"""Step 3: dashboard generator.

Reads saved run JSON from runs/ and injects a small data payload into
dashboard/template.html to produce the final, self-contained
dashboard/index.html — no server, no network, no LLM calls. Re-run this
any time template.html changes, or a new run is saved to runs/.
"""
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
DASHBOARD = ROOT / "dashboard"

REVIEW_FIELDS = ("index", "rating", "title", "text", "true_label", "predicted_label", "correct")


def build_run_payload(run_file: str, run_label: str, labels: list[str]) -> dict:
    data = json.loads((RUNS / run_file).read_text())
    return {
        "run_label": run_label,
        "labels": labels,
        "summary": data["summary"],
        "records": [{k: r[k] for k in REVIEW_FIELDS} for r in data["records"]],
    }


def main():
    payload = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "binary": build_run_payload(
            "batch100.json",
            "First 100 reviews, in file order — binary (POSITIVE / NEGATIVE)",
            ["POSITIVE", "NEGATIVE"],
        ),
    }

    template = (DASHBOARD / "template.html").read_text()
    injected = template.replace(
        "/*__DASHBOARD_DATA__*/ null",
        json.dumps(payload),
    )
    if injected == template:
        raise RuntimeError("Injection marker not found in template.html")

    out_path = DASHBOARD / "index.html"
    out_path.write_text(injected)
    print(f"Wrote {out_path} ({len(injected):,} chars)")


if __name__ == "__main__":
    main()
