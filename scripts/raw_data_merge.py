import csv
import gzip
import json
import os
import shutil
from dataclasses import dataclass
from typing import Generator, List

from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm


@dataclass
class Record:
    src_set: str
    src_vid: str
    prompt: str
    seconds: float


def stream_openvid(path: str) -> Generator[Record, None, None]:
    with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)

        for row in tqdm(reader, desc="Processing OpenVid-1M"):
            src_vid = (row.get("video") or "").strip()
            prompt = (row.get("caption") or "").strip()

            try:
                seconds = float(row["seconds"])
            except (KeyError, TypeError, ValueError):
                continue

            if not src_vid or not prompt or seconds <= 0:
                continue

            yield Record(
                src_set="OpenVid",
                src_vid=src_vid,
                prompt=prompt,
                seconds=seconds,
            )


def stream_activitynet(paths: List[str]) -> Generator[Record, None, None]:
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in tqdm(data, total=len(data), desc=f"Processing {os.path.basename(path)}"):
            src_vid = str(item.get("video") or "").strip()
            prompt = str(item.get("caption") or "").strip()

            try:
                seconds = float(item["duration"])
            except (KeyError, TypeError, ValueError):
                continue

            if not src_vid or not prompt or seconds <= 0:
                continue

            yield Record(
                src_set="ActivityNet",
                src_vid=src_vid,
                prompt=prompt,
                seconds=seconds,
            )


def download_to_path(
        repo_id: str,
        filename: str,
        output_path: str,
) -> None:
    if os.path.exists(output_path):
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cached_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
    )

    shutil.copy2(cached_path, output_path)


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

openvid_path = "data/OpenVid-1M.csv"

activity_net_paths = [
    "data/activitynet_captions_train.json",
    "data/activitynet_captions_val1.json",
]

csv_gz_output_path = "data/causal_react_raw.csv.gz"

fieldnames = [
    "src_set",
    "src_vid",
    "prompt",
    "seconds",
]

# ----------------------------------------------------------------------
# Download missing files
# ----------------------------------------------------------------------

os.makedirs("data", exist_ok=True)

download_to_path(
    repo_id="nkp37/OpenVid-1M",
    filename="data/train/OpenVid-1M.csv",
    output_path=openvid_path,
)

for path in activity_net_paths:
    download_to_path(
        repo_id="friedrichor/ActivityNet_Captions",
        filename=os.path.basename(path),
        output_path=path,
    )

# ----------------------------------------------------------------------
# Merge
# ----------------------------------------------------------------------

count = 0
counts_by_source = {
    "OpenVid": 0,
    "ActivityNet": 0,
}

with gzip.open(
        csv_gz_output_path,
        "wt",
        newline="",
        encoding="utf-8",
        compresslevel=6,
) as f_gz:
    writer = csv.DictWriter(f_gz, fieldnames=fieldnames)
    writer.writeheader()

    for record in stream_openvid(openvid_path):
        writer.writerow({
            "src_set": record.src_set,
            "src_vid": record.src_vid,
            "prompt": record.prompt,
            "seconds": record.seconds,
        })

        count += 1
        counts_by_source[record.src_set] += 1

    for record in stream_activitynet(activity_net_paths):
        writer.writerow({
            "src_set": record.src_set,
            "src_vid": record.src_vid,
            "prompt": record.prompt,
            "seconds": record.seconds,
        })

        count += 1
        counts_by_source[record.src_set] += 1

print(f"OpenVid:    {counts_by_source['OpenVid']:,}")
print(f"ActivityNet:{counts_by_source['ActivityNet']:,}")
print(f"Total:      {count:,}")
print(f"Saved: {csv_gz_output_path}")
