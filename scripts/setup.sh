#!/usr/bin/env bash
# Install requirements and download the dataset.
#
#   bash setup.sh                  # everything
#   bash setup.sh --no-airflow     # skip Airflow
#   bash setup.sh --no-data        # install only
#   bash setup.sh --data-only      # download only
#   SLUG=owner/name bash setup.sh  # other Kaggle dataset
set -euo pipefail

# Load secrets from .env if present (KAGGLE_USERNAME, KAGGLE_KEY, optional SLUG)
if [ -f .env ]; then set -a; . ./.env; set +a; fi

SLUG="${SLUG:-ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes}"
RAW_DIR="data/raw"
AIRFLOW_VERSION="2.10.5"

AIRFLOW=1; INSTALL=1; DATA=1
for arg in "$@"; do
  case "$arg" in
    --no-airflow) AIRFLOW=0 ;;
    --no-data)    DATA=0 ;;
    --data-only)  INSTALL=0 ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

install_requirements() {
  echo "== System packages =="
  apt-get update -y && apt-get install -y libgl1 libglib2.0-0 git unzip tmux

  echo "== Python packages =="
  python -m pip install --upgrade pip
  cd ..
  pip install -r requirements.txt

  if [ "$AIRFLOW" -eq 1 ]; then
    echo "== Airflow =="
    py=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    pip install -r requirements-airflow.txt \
      -c "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${py}.txt" \
      || echo "WARNING: Airflow install failed"
  fi

  python -c "import torch; print('CUDA:', torch.cuda.is_available())"
  pip freeze > requirements.lock.txt
}

download_data() {
  if [ ! -f "$HOME/.kaggle/kaggle.json" ] && [ ! -f "$HOME/.kaggle/access_token" ] \
     && [ -z "${KAGGLE_KEY:-}" ] && [ -z "${KAGGLE_API_TOKEN:-}" ]; then
    echo "ERROR: no Kaggle credentials. Set KAGGLE_API_TOKEN (new token) or"
    echo "       KAGGLE_USERNAME + KAGGLE_KEY (kaggle.json), in .env"
    exit 1
  fi
  [ -f "$HOME/.kaggle/kaggle.json" ] && chmod 600 "$HOME/.kaggle/kaggle.json"

  if [ -n "$(ls -A "$RAW_DIR" 2>/dev/null)" ]; then
    echo "== $RAW_DIR already filled, skipping download =="
  else
    echo "== Downloading $SLUG =="
    mkdir -p "$RAW_DIR"
    kaggle datasets download -d "$SLUG" -p "$RAW_DIR" --unzip
  fi

  echo "== Summary =="
  find "$RAW_DIR" -maxdepth 3 -type d | head -20
  find "$RAW_DIR" -type f | sed -n 's/.*\.\([^./]*\)$/\1/p' | tr 'A-Z' 'a-z' | sort | uniq -c
  sample=$(find "$RAW_DIR" -name '*.txt' | head -1)
  [ -n "$sample" ] && { echo "$sample"; head -3 "$sample"; }
}

[ "$INSTALL" -eq 1 ] && install_requirements
[ "$DATA" -eq 1 ] && download_data
echo "== Done =="