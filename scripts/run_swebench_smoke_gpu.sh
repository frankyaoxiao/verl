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

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-8}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-8}
PPO_MINI_BATCH=${PPO_MINI_BATCH:-4}
export SWEBENCH_TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}
export SWEBENCH_TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES}
export SWEBENCH_VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES}
export SWEBENCH_PPO_MINI_BATCH=${PPO_MINI_BATCH}

./tests/special_e2e/ppo_trainer/run_swebench_smoke.sh "$@" \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  data.train_max_samples=${TRAIN_MAX_SAMPLES} \
  data.val_max_samples=${VAL_MAX_SAMPLES} \
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH}
