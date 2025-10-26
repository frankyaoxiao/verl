# SWEbench GRPO Execution Plan

Last updated: 2025-10-25

## Goals

- Fine‑tune a tool-using coding agent with GRPO on the SWEbench benchmark.
- Support multi-turn rollouts where the model iteratively proposes patches, runs tests, and receives binary rewards.
- Keep the initial proof-of-concept small (mini dataset + lightweight model) while leaving a path to scale.

## High-Level Stack

| Layer | Responsibilities | Status / Notes |
| ----- | ---------------- | -------------- |
| Dataset prep | Convert SWEbench instances into verl RLHF parquet format with per-sample tool metadata | To do |
| Sandbox/tooling | Execute repo checkout, patch apply, and SWEbench judge inside an isolated workspace | To do |
| Multi-turn rollout | Use SGLang multi-turn engine, tool schemas, and interactions to drive conversations | Supported by current infra; needs SWEbench-specific configs |
| Training loop | GRPO via `verl.trainer.main_ppo` with custom config | Supported; needs SWEbench config variant |
| Validation | Track success rate, add deterministic integration tests | To do |

## 1. Dataset Preparation

### Source
- `MariusHobbhahn/swe-bench-verified-mini` (HF dataset) for a small, reproducible starting point.

### Conversion Script
- Clone `examples/data_preprocess/gsm8k_multiturn_w_tool.py`.
- For each SWEbench item:
  - Build `prompt` as a list of chat turns:
    - `system`: explain available tools, repo slug, base commit, execution rules.
    - `user`: include issue description, failing tests, target metrics.
  - Store repo/test metadata in `extra_info`, e.g.:
    ```json
    {
      "repo": "...",
      "base_commit": "...",
      "instance_id": "...",
      "tests": ["..."],
      "need_tools_kwargs": true,
      "tools_kwargs": {
        "swebench_sandbox": {
          "create_kwargs": {
            "repo": "...",
            "base_commit": "...",
            "tests": [...]
          }
        }
      }
    }
    ```
  - Optionally include the verified patch as `ground_truth_patch` for evaluation or reward shaping.
- Output `train.parquet` / `val.parquet` under `~/data/swebench_mini/` (configurable path).
- Keep dataset IDs stable for resume.

### Decisions
- Reward signal: **binary** (1 when SWEbench judge passes, else 0).
- Sandbox implementation: **Docker-based** local containers managed by the new tool.
- Prompt template: target SWEbench’s published styles (see reference below); finalize exact wording when building dataset.

### Open Items
- **Prompt template:** align with SWEbench official instructions or customize? (Clarify desired style.)

## 2. Sandbox & Tooling

### Execution Environment
- Docker-based sandbox per concurrent rollout using SWEbench base images (or lightweight derivative).
- Flow:
  1. On tool `create`: clone/copy repo snapshot (cache locally), checkout base commit, install deps if needed.
  2. On tool `execute`: apply model-supplied diff/patch, run SWEbench judge command, capture stdout/stderr, parse verdict.
  3. Return `ToolResponse` text summarizing result, plus numeric reward (1 for success, 0 otherwise; optional penalty for invalid patches).
  4. On `release`: clean workspace.
- Reuse patterns in `verl.tools.sandbox_fusion_tools` for rate limiting and concurrency.
- Suggested container strategy:
  - Maintain a local Docker image preloaded with SWEbench dependencies + evaluation harness.
  - For each trajectory, spin up a disposable container with volume-mounted working directory containing the repo checkout.
  - Cache Git repositories under a host-side directory (`~/.cache/swebench/repos`) and clone shallow copies into container workspaces to reduce latency.
  - Enforce resource limits via Docker (CPU quota, memory) to prevent runaway processes.
  - Expose a thin Python wrapper that orchestrates `docker run`, handles stdout/stderr streaming, and enforces timeouts.
  - Consider optional “dry-run” mode that executes directly on host (without Docker) for debugging, gated behind config flag.

### Tool Schema
- Define `SWEbenchSandboxTool` (native tool) with schema similar to:
  ```json
  {
    "type": "function",
    "function": {
      "name": "run_swebench_tests",
      "description": "...",
      "parameters": {
        "type": "object",
        "properties": {
          "patch": {"type": "string", "description": "Unified diff to apply"},
          "notes": {"type": "string", "description": "Optional rationale"}
        },
        "required": ["patch"]
      }
    }
  }
  ```
- Support streaming of multiple tool calls per conversation (apply patch, run tests, revise patch).
  - Implementation delegates to `swebench.harness` utilities (`build_env_images`, `run_instance`) to reuse the official Docker build/evaluation pipeline.
  - Configuration supports a `dry_run` flag (enabled in tests/examples) that skips Docker execution and returns placeholder logs, making the tool testable on machines without Docker.
- Support streaming of multiple tool calls per conversation (apply patch, run tests, revise patch).

### Interaction Hook (Optional)
- Implement `SWEbenchInteraction` derived from `BaseInteraction`:
  - After each assistant response, run the tool automatically if the model hasn’t called it, or summarize the latest test outcome.
  - Provide structured feedback (“Tests failed: X”, “Repository state reset”).

### Open Items
- **Resource cleanup:** define policy for caching repos vs. re-cloning each run.

## 3. Training Configuration

### Config Template
- Config is provided under `verl/trainer/config/swebench_multiturn_grpo.yaml`.
- Key fields (initial run targets **allenai/OLMo-2-1124-7B-Instruct**):
  ```yaml
  data:
    train_files: /path/to/swebench/train.parquet
    val_files: /path/to/swebench/val.parquet
    max_prompt_length: 2048
    max_response_length: 2048
    return_raw_chat: True

  actor_rollout_ref:
    hybrid_engine: True
    rollout:
      name: sglang
      multi_turn:
        enable: True
        max_assistant_turns: 6
        max_user_turns: 6
        tool_config_path: examples/sglang_multiturn/config/tool_config/swebench_tool_config.yaml
        interaction_config_path: examples/sglang_multiturn/config/interaction_config/swebench_interaction.yaml # optional

  algorithm:
    adv_estimator: grpo

  trainer:
    n_gpus_per_node: 1
    total_training_steps: 500 # adjust later
    save_freq: 50
  ```
- Start with a manageable checkpoint (e.g., `allenai/OLMo-2-1124-7B-Instruct`) and conservative batch sizes (`train_batch_size=32`, `rollout.n=2`).

### Run Script
- Mirror `tests/special_e2e/ppo_trainer/run_char_count_smoke.sh`; include dataset creation, optional sandbox warmup, and GRPO launch.

### Open Items
- **Turn limits:** decide default number of assistant attempts before forced termination.

## 4. Validation & Monitoring

- Add deterministic regression test (e.g., `tests/special_e2e/ppo_trainer/test_swebench_tool.sh`) to:
  1. Initialize sandbox tool for a known SWEbench instance.
  2. Apply the verified patch.
  3. Assert reward == 1, repo cleanup succeeded.
- During training:
  - Log tool call outputs, reward distributions, and repo paths.
  - Track success rate on validation split every `trainer.test_freq` steps.
  - Optionally run a held-out evaluation script after checkpoints to compute official SWEbench metrics.

## 5. Milestones & Execution Order

1. **Dataset script** → produce sample parquet files (10 train / 2 val) and sanity-check schema.
2. **Sandbox prototype** → run SWEbench judge manually inside the target environment; document required dependencies.
3. **Tool integration** → stub tool that returns fake rewards; confirm multi-turn loop wiring.
4. **Full sandbox tool** → replace stub with real SWEbench execution, handle failure cases.
5. **Training dry run** → single rollout to confirm conversation flow, tool calls, reward logging.
6. **Mini GRPO run** → launch config on mini dataset; analyze reward curve and validation metrics.
7. **Scale decisions** → expand dataset/model, tune hyperparameters, add partial reward shaping if desired.

## Immediate Task Breakdown

1. **Dataset Ingestion**
   - Implement `recipe/swebench/create_dataset.py` mirroring GSM8K script.
   - Load `MariusHobbhahn/swe-bench-verified-mini` (`split="test"`), emit train/val parquet with style-3 prompt structure.
   - Include unit test to validate schema and presence of `tools_kwargs`.

2. **Docker Environment Setup**
   - Build local SWEbench base image (Dockerfile derived from official harness).
   - Create helper script to launch container, mount repo cache, run judge command, and teardown.
   - Document prerequisites (Docker version, disk space, env vars) in README.

3. **Tool Implementation**
   - Add `verl/tools/swebench_tool.py` with create/execute/release hooks calling Docker helper.
   - Define OpenAI tool schema + config entry under `examples/sglang_multiturn/config/tool_config/swebench_tool_config.yaml`.
   - Extend `SWEbenchInteraction` if we want automated coaching.

4. **Training Config & Smoke Test**
   - Copy config template to `examples/sglang_multiturn/config/swebench_multiturn_grpo.yaml` (OLMo 7B).
   - Write `tests/special_e2e/ppo_trainer/run_swebench_smoke.sh` mirroring char-count script.
   - Ensure script performs dataset prep, sandbox warmup (`docker pull/build`), and a short GRPO run (e.g., 5 steps).

5. **Validation Harness**
   - Add integration test that runs tool against known SWEbench instance and checks for reward=1.
   - Set up logging hooks / telemetry (Ray timeline, tool output dumps).

6. **Documentation & Ops**
   - Update README with usage instructions, environment setup, troubleshooting tips.
   - Track open questions (turn limits, caching policy) and resolve before scaling.

## Open Questions for Confirmation

### Prompt Template Reference

SWE-bench distributes canonical prompt styles in `swebench/inference/make_datasets/create_instance.py`. Notably:

- **style-2 / style-2-edits-only**
  - Opens with “You will be provided with a partial code base and an issue statement...”.
  - Surrounds the problem statement with `<issue>...</issue>`.
  - Embeds repository context under `<code>...</code>`.
  - Shows an example unified diff inside `<patch>...</patch>` and instructs the model to return a single patch file.

- **style-3**
  - Same base structure as style-2 but includes a narrative explanation of patch format and ends with “Respond below:”.

These prompt styles rely on dataset fields (`problem_statement`, `readmes`, `file_contents`). For our converter we should mirror one of these (likely `style-3`) so outputs stay comparable to official SWE-bench setups.
