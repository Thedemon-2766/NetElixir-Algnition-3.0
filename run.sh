#!/usr/bin/env bash
###############################################################################
# run.sh — AIgnition 3.0 Hackathon · NetElixir · Single entry point
#
# CONTRACT (Hackathon Submission Guide §3):
#   ./run.sh <DATA_DIR> <MODEL_PATH> <OUTPUT_PATH>
#
#   DATA_DIR    — folder containing channel CSVs  (default: ./data)
#   MODEL_PATH  — path to trained pickle model    (default: ./pickle/model.pkl)
#   OUTPUT_PATH — full path for output CSV        (default: ./output/predictions.csv)
#
# What this script does:
#   1. Sets up a Python venv and installs pinned deps (one-time; reused on reruns)
#   2. Trains the model bundle on whatever CSVs are in DATA_DIR
#   3. Runs probabilistic forecasting and writes predictions to OUTPUT_PATH
#
# Guarantees:
#   • Fails loudly on any error (set -euo pipefail)
#   • No interactive prompts
#   • No network calls at run time
#   • Fixed random seeds (SEED=42)
#   • OUTPUT_PATH is written fresh on every run (never appended)
###############################################################################
set -euo pipefail

# ── Three positional arguments with sensible defaults ─────────────────────────
DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"

# ── Forecasting horizon (days) ─────────────────────────────────────────────────
HORIZON="${HORIZON:-30}"

# ── Ensure required directories exist ─────────────────────────────────────────
mkdir -p "$(dirname "$MODEL_PATH")"
mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "======================================================================"
echo " AIgnition 3.0 — Probabilistic Revenue & ROAS Forecasting"
echo "======================================================================"
echo "  data    : $DATA_DIR"
echo "  model   : $MODEL_PATH"
echo "  output  : $OUTPUT_PATH"
echo "  horizon : $HORIZON days"
echo "======================================================================"

# ── Step 1: Python virtual environment ────────────────────────────────────────
VENV_DIR=".venv"
PYTHON_BIN="${PYTHON_BIN:-python3.14}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "INFO: $PYTHON_BIN not found — falling back to python3." >&2
  PYTHON_BIN="python3"
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "[1/3] Creating virtual environment …"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "[1/3] Reusing virtual environment at $VENV_DIR."
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[1/3] Installing pinned dependencies …"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet --only-binary=:all: -r requirements.txt

# ── Step 2: Train model on DATA_DIR ───────────────────────────────────────────
echo "[2/3] Training forecasting model on data from $DATA_DIR …"
python src/train.py \
  --data-dir  "$DATA_DIR" \
  --model-dir "$(dirname "$MODEL_PATH")"

# ── Step 3: Generate probabilistic forecasts ──────────────────────────────────
echo "[3/3] Generating probabilistic forecasts …"
python src/predict.py \
  --pickle  "$MODEL_PATH" \
  --data    "$DATA_DIR" \
  --output  "$OUTPUT_PATH" \
  --horizon "$HORIZON"

echo "======================================================================"
echo " DONE. Predictions written to: $OUTPUT_PATH"
echo "======================================================================"
