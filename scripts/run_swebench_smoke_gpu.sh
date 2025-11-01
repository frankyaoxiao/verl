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

# Default to Llama 3.1 8B unless the caller overrides.
export SWEBENCH_MODEL=${SWEBENCH_MODEL:-meta-llama/Llama-3.1-8B-Instruct}

# Force Ray to start a local instance unless the caller already supplied an address.
export RAY_ADDRESS=${RAY_ADDRESS:-local}

# Always run the trainer step in the smoke test.
export RUN_TRAIN=1

# Enable prewarmed SWEbench envs (skip conda create inside sandbox if template already contains it)
export SWEBENCH_PREWARMED=${SWEBENCH_PREWARMED:-1}

# Optional: point to a specific prebuilt E2B template alias for the SWEbench sandbox tool.
# If unset, the tool config falls back to the default alias.
# Example: export SWEBENCH_TEMPLATE_ALIAS="swebench-mini-OWNER_REPO-<hash>"
export SWEBENCH_TEMPLATE_ALIAS=${SWEBENCH_TEMPLATE_ALIAS:-}

# Enable full conversation logging for debugging (saved to tmp/conversations/)
export VERL_CONVERSATION_DUMP_DIR=${VERL_CONVERSATION_DUMP_DIR:-tmp/conversations}

# Enable detailed timing logs for debugging performance
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}

# Hydra chokes on tildes; precompute absolute dataset paths.
export SWEBENCH_TRAIN="${DATA_ROOT}/rl/train.parquet"
export SWEBENCH_VAL="${DATA_ROOT}/rl/val.parquet"

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-8}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-8}
PPO_MINI_BATCH=${PPO_MINI_BATCH:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-25000}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
export SWEBENCH_ROLLOUT_N=${SWEBENCH_ROLLOUT_N:-8}
export SWEBENCH_TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}
export SWEBENCH_TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES}
export SWEBENCH_VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES}
export SWEBENCH_PPO_MINI_BATCH=${PPO_MINI_BATCH}
export SWEBENCH_MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH}
export SWEBENCH_MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH}

./tests/special_e2e/ppo_trainer/run_swebench_smoke.sh "$@" \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  data.train_max_samples=${TRAIN_MAX_SAMPLES} \
  data.val_max_samples=${VAL_MAX_SAMPLES} \
  data.max_prompt_length=${MAX_PROMPT_LENGTH} \
  data.max_response_length=${MAX_RESPONSE_LENGTH} \
  actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH} \
  actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH} \
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH}
