import csv
import gzip
import hashlib
import json
from pathlib import Path

from causal_react_bench.prompts.preprocess import PromptBuilder
from tqdm.auto import tqdm

INPUT = "data/causal_react_raw.csv.gz"
OUTPUT = "data/causal_react_split.jsonl"
DONE = "data/causal_react_split.done"


def gen_sample_id(src_set: str, src_vid: str, prompt: str, row_index: int, candidate_index: int) -> str:
    identity = json.dumps([src_set, src_vid, prompt, row_index, candidate_index], ensure_ascii=False)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def main():
    Path(DONE).unlink(missing_ok=True)
    builder = PromptBuilder()

    with (
        gzip.open(INPUT, "rt", encoding="utf-8") as fin,
        open(OUTPUT, "w", encoding="utf-8", buffering=1) as fout,
    ):
        reader = csv.DictReader(fin)

        for row_index, row in enumerate(tqdm(reader)):
            try:
                results = builder.process(row)
            except Exception as e:
                print(f"[ERROR] {row['src_set']} {row['src_vid']}: {e}")
                continue

            for candidate_index, result in enumerate(results):
                result["sample_id"] = gen_sample_id(
                    result["src_set"], result["src_vid"], result["prompt"], row_index, candidate_index
                )

                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()

    Path(DONE).touch()


if __name__ == "__main__":
    main()
