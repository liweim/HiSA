#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OSWORLD_DIR="${OSWORLD_ROOT:-${ROOT_DIR}/benchmarks/OSWorld}"
RUNNER="${OSWORLD_DIR}/scripts/python/run_multienv_hisa.py"
PATH_TO_VM="${PATH_TO_VM:-${OSWORLD_DIR}/Ubuntu.qcow2}"
RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/results/hisa_gpt5_gta1_50}"
NUM_ENVS="${NUM_ENVS:-1}"

if [[ ! -d "${OSWORLD_DIR}" ]]; then
  echo "OSWorld checkout not found: ${OSWORLD_DIR}" >&2
  exit 1
fi
if [[ ! -f "${RUNNER}" ]]; then
  echo "HiSA multi-environment runner not found: ${RUNNER}" >&2
  exit 1
fi
if [[ ! -f "${PATH_TO_VM}" ]]; then
  echo "OSWorld Ubuntu qcow2 image not found: ${PATH_TO_VM}" >&2
  echo "Download it to ${OSWORLD_DIR}/Ubuntu.qcow2 or set PATH_TO_VM." >&2
  exit 1
fi

cd "${OSWORLD_DIR}"
exec python3 "${RUNNER}" \
  --provider_name docker \
  --path_to_vm "${PATH_TO_VM}" \
  --global_planner_model gpt-5 \
  --state_manager_model gpt-5-mini \
  --visual_grounder_model gta1-7b \
  --use_qdrant_server \
  --max_steps 50 \
  --num_envs "${NUM_ENVS}" \
  --result_dir "${RESULT_DIR}" \
  --test_all_meta_path evaluation_examples/test_all.json \
  --test_config_base_dir evaluation_examples \
  --headless \
  "$@"
