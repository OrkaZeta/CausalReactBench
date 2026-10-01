#!/usr/bin/env bash
#SBATCH --partition=H100,RTX6000PRO
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=pre_prompts
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=OrkaZeta@outlook.com

set -Eeuo pipefail

RESET='\033[0m'
BLUE='\033[34m'
GREEN='\033[32m'
YELLOW='\033[33m'
RED='\033[31m'

MODE="bash"
[[ -n "${SLURM_JOB_ID:-}" ]] && MODE="slurm"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p "logs/${SLURM_JOB_NAME:-pre_prompts}"

log() {
    local level="$1"
    local color="$2"
    shift 2
    printf '[%s] %b[%s]%b [%s] %s\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" \
        "$color" "$level" "$RESET" "$MODE" "$*"
}

info()    { log INFO "$BLUE" "$@"; }
success() { log OK "$GREEN" "$@"; }
warn()    { log WARN "$YELLOW" "$@"; }
error()   { log ERROR "$RED" "$@" >&2; }


run_split() {
    info "Starting DeepSeek preprocessing"
    uv run python -m scripts.build_pre_prompts
    success "DeepSeek preprocessing finished"
}

run_review() {
    info "Starting Qwen reviewer"
    uv run python -m scripts.review_pre_prompts
    success "Qwen review finished"
}

rm -f \
    "data/causal_react_split.jsonl" \
    "data/causal_react_split.done" \
    "data/causal_react_reviewed.jsonl" \
    "data/causal_react_reviews.jsonl"

run_review &
REVIEW_PID=$!

trap 'kill "$REVIEW_PID" 2>/dev/null || true' EXIT

run_split
wait "$REVIEW_PID"

trap - EXIT
success "Pipeline finished"
