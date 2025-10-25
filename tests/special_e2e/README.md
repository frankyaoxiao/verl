This folder is reserved for end-to-end tests that typically require multiple GPUs.

- `ppo_trainer/run_char_count_smoke.sh`: single-node (default 4×GPU) smoke script that creates the toy char-count dataset, performs a 1-epoch SFT warmup, merges the resulting FSDP checkpoint to HuggingFace format, and runs a short GRPO loop to verify the training stack end-to-end. Override `NUM_GPUS`, `DATA_ROOT`, `SFT_DIR`, or `RL_DIR` to customize paths.
