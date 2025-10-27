#!/usr/bin/env bash
set -xeuo pipefail

# Ensure the verl environment is active; bail with a hint if not.
if [[ "${CONDA_DEFAULT_ENV:-}" != "verl" ]]; then
  echo "Please activate the 'verl' conda environment before running this script." >&2
  exit 1
fi

# Expand DATA_ROOT to an absolute path (defaults to ~/data/swebench_mini).
export DATA_ROOT=${DATA_ROOT:-~/data/swebench_mini}
DATA_ROOT=$(python - <<'PY'
import os
print(os.path.abspath(os.path.expanduser(os.environ["DATA_ROOT"])))
PY
)
export DATA_ROOT

# Point the tool config to the E2B-enabled SWEbench setup unless overridden.
export SWEBENCH_TOOL_CONFIG=${SWEBENCH_TOOL_CONFIG:-examples/sglang_multiturn/config/tool_config/swebench_tool_config.yaml}

# Number of GPUs to allocate to the trainer (defaults to 1).
export SWEBENCH_NUM_GPUS=${SWEBENCH_NUM_GPUS:-1}

# Always run the trainer step in the smoke test.
export RUN_TRAIN=1

# Hydra chokes on tildes; precompute absolute dataset paths.
export SWEBENCH_TRAIN="${DATA_ROOT}/rl/train.parquet"
export SWEBENCH_VAL="${DATA_ROOT}/rl/val.parquet"

./tests/special_e2e/ppo_trainer/run_swebench_smoke.sh "$@"
