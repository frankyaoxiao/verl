export DATA_ROOT=${DATA_ROOT:-~/data/swebench_mini}
DATA_ROOT=$(python - <<'PY'
import os
print(os.path.abspath(os.path.expanduser(os.environ["DATA_ROOT"])))
PY
)
export DATA_ROOT

export RUN_TRAIN=${RUN_TRAIN:-1}

echo "SWEbench smoke: model=${SWEBENCH_MODEL:-meta-llama/Llama-3.1-8B-Instruct} gpus=${SWEBENCH_NUM_GPUS:-1}" >&2

./tests/special_e2e/ppo_trainer/run_swebench_smoke.sh "$@"

if [[ "${SWEBENCH_E2B_CLEANUP:-1}" == "1" ]]; then
  echo "[post] Cleaning up E2B sandboxes (best effort)"
  python3 scripts/cleanup_e2b_sandboxes.py || true
fi
