#!/usr/bin/env bash
# Prepare the WorkBench benchmark for use with this repository.
#
# This script:
#   1. Clones the upstream WorkBench repository (https://github.com/olly-styles/WorkBench).
#   2. Converts the upstream task definitions into the normalized JSON schema
#      expected by `agent_scaling.datasets.workbench.WorkbenchDataset`.
#
# Usage: bash scripts/setup_workbench.sh [--upstream-dir <path>] [--out <path>]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UPSTREAM_DIR="${UPSTREAM_DIR:-${REPO_ROOT}/third_party/WorkBench}"
OUT_JSON="${OUT_JSON:-${REPO_ROOT}/datasets/workbench.json}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"
SEED="${SEED:-42}"
WORKBENCH_COMMIT="${WORKBENCH_COMMIT:-42f1b6dc18986f47701ca83de2d63dd4f95c59a0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --upstream-dir) UPSTREAM_DIR="$2"; shift 2;;
    --out)          OUT_JSON="$2"; shift 2;;
    --sample-size)  SAMPLE_SIZE="$2"; shift 2;;
    --seed)         SEED="$2"; shift 2;;
    *)              echo "Unknown argument: $1" >&2; exit 2;;
  esac
done

mkdir -p "$(dirname "${UPSTREAM_DIR}")"
mkdir -p "$(dirname "${OUT_JSON}")"

if [[ ! -d "${UPSTREAM_DIR}/.git" ]]; then
  echo "[setup_workbench] cloning upstream WorkBench into ${UPSTREAM_DIR}"
  git clone https://github.com/olly-styles/WorkBench.git "${UPSTREAM_DIR}"
else
  echo "[setup_workbench] upstream already cloned at ${UPSTREAM_DIR}"
fi

if ! git -C "${UPSTREAM_DIR}" cat-file -e "${WORKBENCH_COMMIT}^{commit}" 2>/dev/null; then
  echo "[setup_workbench] fetching pinned upstream commit ${WORKBENCH_COMMIT}"
  git -C "${UPSTREAM_DIR}" fetch --depth=1 origin "${WORKBENCH_COMMIT}"
fi
echo "[setup_workbench] checking out pinned upstream commit ${WORKBENCH_COMMIT}"
git -C "${UPSTREAM_DIR}" checkout --detach "${WORKBENCH_COMMIT}"

echo "[setup_workbench] converting upstream tasks -> ${OUT_JSON} (stratified ${SAMPLE_SIZE}-instance subset, seed ${SEED})"
uv run python "${REPO_ROOT}/scripts/_convert_workbench.py" \
  --upstream-dir "${UPSTREAM_DIR}" \
  --out "${OUT_JSON}" \
  --sample-size "${SAMPLE_SIZE}" \
  --seed "${SEED}"

echo "[setup_workbench] wrote ${OUT_JSON}"
