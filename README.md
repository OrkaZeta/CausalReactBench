# Causal React Bench

## Installation & Preparation

- Env

```bash
git clone https://github.com/OrkaZeta/CausalReactBench.git
cd ./CausalReactBench
uv sync
```

- Models

```bash
# Wan2.2
mkdir ./ckpt
hf download Wan-AI/Wan2.2-T2V-A14B --local-dir ./ckpt/Wan2.2-T2V-A14B
```

## Data Generate Pipeline

### 1. Events-Driven Prompt LLM Filtering

- Models:
  - DeepSeek-V4-Pro API: prompt reasoning, filtering, rewriting, and splitting
  - Qwen3.8-27B-FP8: independent review and error checking
- Src Datasets merge with `data/raw_data_merge.py` -> `data/causal_react_raw.csv.gz`:
  - HF-friedrichor/ActivityNet_Captions [train, val1]
  - HF-nkp37/OpenVid-1M [train]
- Events-Driven Filtering
  > Keep coherent scenes containing 2–4 distinct, temporally ordered visible entity events. Camera motion does not count as an event; do not invent actions.
  >
  > Use OpenVid-1M[`seconds`] and ActivityNet_Captions[`duration`] only as reference duration. Duration routing is handled by code, not by the LLM.
  >
  > 1. `sec <= 5s`: preserve the original duration and minimally clean/filter the prompt.
  > 2. `5s < sec <= 7s`: rewrite the central scene into a physically plausible 5-second prompt.
  > 3. `sec > 7s`: semantically decompose the source into multiple independent 5-second-compatible prompts; do not treat them as exact temporal cuts of the source video.
  >
  > DeepSeek-V4-Pro produces the candidate prompts. Qwen3.8-27B-FP8 independently reviews source faithfulness, single-scene consistency, physical plausibility, and duration feasibility.

Sample structure:

```json
[
  {
    "sample_id": "{non-repeat short id with [a-z0-9]}",
    "prompt": "{prompt text}",
    "duration": 5.0,
    "src_set": "{ActivityNet,OpenVid}",
    "src_vid": "{src video id: OpenVid-1M[video], ActivityNet_Captions[video]}"
  }
]
```

```bash
DS_TOKEN_CP="sk-xxxx" bash scripts/pre_prompts.bash
```

Or submit with Slurm:

```bash
mkdir -p logs
DS_TOKEN_CP="sk-xxxx" sbatch scripts/pre_prompts.bash
```

- Candidates: `data/causal_react_split.jsonl`
- ReviewResult: `data/causal_react_reviews.jsonl`
- Accepted prompts: `data/causal_react_reviewed.jsonl`

### 2. Wan2.2 video Gen

```bash
uv run python scripts/gen_full_vid.py
```

Wan2.2 T2V-A14B, 832×480, 16 fps, one H100/RTX6000PRO.
Input: `data/causal_react_reviewed.jsonl`; output: `data/causal_react_full_vid/{sample_id}.mp4`.
