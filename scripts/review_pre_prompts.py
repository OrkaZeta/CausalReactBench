import json
import os
import time
from dataclasses import asdict

from causal_react_bench.models.qwen import QwenReviewer
from causal_react_bench.prompts.configs import SYS_REVIEW
from causal_react_bench.prompts.schema import ReviewResult


INPUT = "data/causal_react_split.jsonl"
DONE = "data/causal_react_split.done"
OUTPUT = "data/causal_react_reviewed.jsonl"
REVIEWS = "data/causal_react_reviews.jsonl"


def watch_jsonl(path: str):
    while not os.path.exists(path):
        time.sleep(1)

    with open(path, "r", encoding="utf-8") as f:
        while True:
            position = f.tell()
            line = f.readline()

            if line and not line.endswith("\n"):
                if os.path.exists(DONE):
                    raise ValueError("Candidate JSONL ends with an incomplete record")
                f.seek(position)
                time.sleep(0.5)
                continue

            if line:
                yield json.loads(line)
                continue

            if os.path.exists(DONE):
                # The writer may have appended between our EOF read and marker check.
                f.seek(position)
                line = f.readline()
                if line:
                    if not line.endswith("\n"):
                        raise ValueError("Candidate JSONL ends with an incomplete record")
                    yield json.loads(line)
                    continue
                break

            time.sleep(0.5)


def make_review_prompt(row: dict) -> str:
    return f"""SOURCE_SECONDS: {row["source_duration"]:.3f}
TARGET_SECONDS: {row["duration"]:.3f}

SOURCE_PROMPT:
{row["source_prompt"]}

DERIVED_PROMPT:
{row["prompt"]}
"""


def main():
    reviewer = QwenReviewer()

    with (
        open(OUTPUT, "w", encoding="utf-8", buffering=1) as fout,
        open(REVIEWS, "w", encoding="utf-8", buffering=1) as freviews,
    ):
        for row in watch_jsonl(INPUT):
            try:
                review = ReviewResult.from_dict(reviewer.run(
                    SYS_REVIEW,
                    make_review_prompt(row),
                ))
            except Exception as e:
                print(f"[ERROR] {row['sample_id']}: {e}")
                continue

            # Save accepted and rejected reviews before filtering final samples.
            audit = {**row, "review": asdict(review), "accepted": review.accepted}
            freviews.write(json.dumps(audit, ensure_ascii=False) + "\n")
            freviews.flush()

            if not review.accepted:
                continue

            prompt = review.prompt.strip()
            result = {
                "sample_id": row["sample_id"],
                "prompt": prompt,
                "duration": row["duration"],
                "src_set": row["src_set"],
                "src_vid": row["src_vid"],
            }

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()


if __name__ == "__main__":
    main()
