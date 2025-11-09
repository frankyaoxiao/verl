export NCCL_DEBUG=WARN
export DATA_ROOT=${DATA_ROOT:-~/data/swebench_mini}
DATA_ROOT=$(python - <<'PY'
import os
print(os.path.abspath(os.path.expanduser(os.environ["DATA_ROOT"])))
PY
)
export DATA_ROOT

export RUN_TRAIN=${RUN_TRAIN:-1}

# Enable pre-built templates (new simplified architecture)
export SWEBENCH_TEMPLATES_ENABLE=1
export SWEBENCH_ALIAS_PREFIX=swebench
export SWEBENCH_PREWARMED=1


./tests/special_e2e/ppo_trainer/run_swebench_smoke.sh "$@"

if [[ "${SWEBENCH_E2B_CLEANUP:-1}" == "1" ]]; then
  echo "[post] Cleaning up E2B sandboxes (best effort)"
  python3 scripts/cleanup_e2b_sandboxes.py || true
fi
