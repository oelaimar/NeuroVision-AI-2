#!/usr/bin/env bash
set -Eeuo pipefail

# Install project requirements and optionally Apache Airflow.
# Run from anywhere: bash scripts/setup.sh [--no-airflow]
# Dataset discovery and downloading are intentionally handled in the notebook.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Load project-root .env, if present.
if [[ -f "$PROJECT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_DIR/.env"
  set +a
fi

AIRFLOW_VERSION="${AIRFLOW_VERSION:-2.10.5}"
PYTHON="${PYTHON:-python3}"
INSTALL_AIRFLOW=1

usage() {
  cat <<'USAGE'
Usage: bash scripts/setup.sh [options]

Options:
  --no-airflow  Skip installing Airflow in its own virtual environment
  -h, --help    Show this help message

Dataset downloading and Train/Val folder discovery are handled in the notebook.
USAGE
}

for arg in "$@"; do
  case "$arg" in
    --no-airflow) INSTALL_AIRFLOW=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

log() { printf '\n== %s ==\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

install_requirements() {
  log "Install project requirements"
  [[ -f "$PROJECT_DIR/requirements.txt" ]] || die "requirements.txt was not found at $PROJECT_DIR/requirements.txt"
  "$PYTHON" -m pip install --upgrade pip
  "$PYTHON" -m pip install -r "$PROJECT_DIR/requirements.txt"
  "$PYTHON" -m pip check || echo "WARNING: pip check reported dependency conflicts. Review the output above." >&2

  log "Check TensorFlow GPU availability (if TensorFlow is installed)"
  "$PYTHON" - <<'PY'
try:
    import tensorflow as tf
    print("TensorFlow:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))
except ImportError:
    print("TensorFlow is not installed in this Python environment; skipping GPU check.")
PY
}

install_airflow() {
  log "Install Apache Airflow $AIRFLOW_VERSION in an isolated environment"
  local airflow_venv="$PROJECT_DIR/.venv-airflow"
  "$PYTHON" -m venv "$airflow_venv"
  local airflow_python="$airflow_venv/bin/python"
  "$airflow_python" -m pip install --upgrade pip
  "$airflow_python" -m pip install "apache-airflow==${AIRFLOW_VERSION}" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-$("$airflow_python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")').txt"
  echo "Airflow installed separately in: $airflow_venv"
}

install_requirements
if [[ "$INSTALL_AIRFLOW" == "1" ]]; then
  install_airflow
fi

log "Setup complete"