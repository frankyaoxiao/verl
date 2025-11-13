# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities to convert SWE-bench datasets into verl RLHF format."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import os
from datasets import Dataset, load_dataset
from swebench.harness.test_spec.test_spec import make_test_spec, TestSpec

DEFAULT_DATASET = "MariusHobbhahn/swe-bench-verified-mini"
_REPO_ROOT = os.getenv("SWEBENCH_REPO_PATH", "/workspace/testbed")

def _build_system_prompt(max_turns: int) -> str:
    """Build system prompt with actual max_turns."""
    return (
        "You are an autonomous software engineer tasked with resolving issues in real-world repositories. "
        "Your workspace is a persistent SWEbench sandbox with two tools available:\n"
        "- `bash`: run shell commands (e.g., inspect files, run tests). Each command starts fresh in the repository root "
        f"({_REPO_ROOT}), so use `cd dir && command` if you need to work in subdirectories.\n"
        "- `submit_solution`: run the official SWEbench judge against the current repository state. Only call this when you "
        "are confident in your fix—the judge will end the sandbox session.\n\n"
        "Important: Shell state (like the current directory) doesn't persist between commands, but file changes do.\n\n"
        f"You have a maximum of {max_turns} turns to solve the issue. Plan your approach efficiently and call `submit_solution` before running out of turns.\n\n"
        "Think carefully, plan your next action, and iteratively refine your solution until the tests pass."
    )
USER_PROMPT_TEMPLATE = """Repository: {repo}
Base commit: {base_commit}
Environment setup commit: {environment_commit}

Issue:
{problem_statement}

Hints:
{hints}

Fail-to-pass tests:
{fail_to_pass}

Pass-to-pass tests:
{pass_to_pass}

Testing:
{test_command_info}

Instructions:
- Analyse the repository state and reason step by step.
- Use the `bash` tool to run shell commands and explore the sandbox (e.g., `ls`, `cat`).
- Call `submit_solution` only when you believe the fix is ready; this runs the SWEbench judge on the current repository and ends the session.
- Provide unified diff patches when you believe the issue is resolved.
"""


@dataclass(frozen=True)
class DatasetSplit:
    train: Dataset
    val: Dataset


def _extract_test_command(eval_script: str) -> str:
    """Extract the actual test command from the eval_script.
    
    Looks for patterns like:
    - ./tests/runtests.py (Django)
    - python -m pytest
    - pytest
    - tox
    etc.
    """
    import re
    
    # Common test command patterns
    patterns = [
        r'(\.\/tests\/runtests\.py[^\n]*)',  # Django style
        r'(python -m pytest[^\n]*)',
        r'(pytest[^\n]*)',
        r'(python -m unittest[^\n]*)',
        r'(tox[^\n]*)',
        r'(python setup\.py test[^\n]*)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, eval_script)
        if match:
            cmd = match.group(1).strip()
            # Clean up any shell artifacts
            cmd = cmd.rstrip(';').rstrip('&').strip()
            return cmd
    
    # Fallback: if no pattern matched, look for any line that looks like a test command
    for line in eval_script.split('\n'):
        line = line.strip()
        if any(keyword in line.lower() for keyword in ['test', 'pytest', 'runtests']):
            if line and not line.startswith('#') and not line.startswith('export'):
                # Try to extract the actual command
                if '&&' in line:
                    parts = line.split('&&')
                    for part in parts:
                        part = part.strip()
                        if any(kw in part.lower() for kw in ['test', 'pytest', 'runtests']):
                            return part.rstrip(';').rstrip('&').strip()
                return line.rstrip(';').rstrip('&').strip()
    
    return "See the evaluation script for test commands"


def _format_section(title: str, items: Iterable[str]) -> str:
    items = list(items)
    if not items:
        return f"- (none)"
    return "\n".join(f"- {item}" for item in items)


def _build_user_prompt(example: dict[str, Any]) -> str:
    hints = example.get("hints_text") or "(none provided)"
    fail_to_pass = _format_section("Fail-to-pass tests", example.get("FAIL_TO_PASS", []))
    pass_to_pass = _format_section("Pass-to-pass tests", example.get("PASS_TO_PASS", []))
    
    # Extract test command from eval_script
    test_command_info = "Test command information not available"
    try:
        test_spec: TestSpec = make_test_spec(example)
        eval_script = test_spec.eval_script if hasattr(test_spec, 'eval_script') else None
        if eval_script:
            test_cmd = _extract_test_command(eval_script)
            test_command_info = f"When `submit_solution` is called, tests will be evaluated using:\n  {test_cmd}\nYou can run this command manually via `bash` to verify your fix before submitting."
    except Exception as e:
        # If we can't extract the test command, use a generic message
        test_command_info = "Tests will be run automatically when you call `submit_solution`"

    # Replace placeholders in template
    prompt = USER_PROMPT_TEMPLATE.format(
        repo=example["repo"],
        base_commit=example["base_commit"],
        environment_commit=example.get("environment_setup_commit", "unknown"),
        problem_statement=example["problem_statement"],
        hints=hints,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        test_command_info=test_command_info,
    )
    return prompt.strip()


def build_chat_messages(example: dict[str, Any], max_turns: int = 6) -> list[dict[str, str]]:
    """Create chat template compatible messages for verl."""

    return [
        {"role": "system", "content": _build_system_prompt(max_turns)},
        {"role": "user", "content": _build_user_prompt(example)},
    ]


def build_tools_kwargs(example: dict[str, Any]) -> dict[str, Any]:
    """Construct per-instance tool kwargs for the SWEbench sandbox tool."""
    # Optional per-instance template alias injection for prewarmed templates
    template_alias: str | None = None
    try:
        enable_templates = bool(int(os.getenv("SWEBENCH_TEMPLATES_ENABLE", "0")))
    except Exception:
        enable_templates = False
    if enable_templates:
        try:
            spec: TestSpec = make_test_spec(example)
            env_key = spec.env_image_key  # sweb.env.*.<HASH>:tag
            short_hash = env_key.split(":", 1)[0].split(".")[-1][:10]
            repo = example["repo"].replace("/", "_")
            alias_prefix = os.getenv("SWEBENCH_ALIAS_PREFIX", "swebench")
            template_alias = f"{alias_prefix}-{repo}-{short_hash}"
        except Exception:
            template_alias = None

    create_kwargs = {
        "repo": example["repo"],
        "instance_id": example["instance_id"],
        "base_commit": example["base_commit"],
        "environment_setup_commit": example.get("environment_setup_commit"),
        "version": example.get("version"),
        "fail_to_pass": example.get("FAIL_TO_PASS", []),
        "pass_to_pass": example.get("PASS_TO_PASS", []),
        "patch": example.get("patch"),
        "test_patch": example.get("test_patch"),
        "problem_statement": example.get("problem_statement"),
        "dataset_instance": example,
    }
    if template_alias:
        create_kwargs["template"] = template_alias

    return {
        "bash": {"create_kwargs": create_kwargs},
        "submit_solution": {"create_kwargs": create_kwargs},
    }


def build_extra_info(example: dict[str, Any], split: str, idx: int) -> dict[str, Any]:
    """Populate extra_info column consumed by RLHFDataset."""

    extra_info = {
        "split": split,
        "index": idx,
        "instance_id": example["instance_id"],
        "need_tools_kwargs": True,
        "tools_kwargs": build_tools_kwargs(example),
    }

    # Persist artifacts that may be useful for evaluation or debugging
    extra_info["ground_truth_patch"] = example.get("patch")
    extra_info["test_patch"] = example.get("test_patch")
    extra_info["metadata"] = {
        "created_at": example.get("created_at"),
        "version": example.get("version"),
    }
    return extra_info


def build_record(example: dict[str, Any], split: str, idx: int, data_source: str, max_turns: int = 6) -> dict[str, Any]:
    """Transform a HuggingFace SWE-bench example into verl RLHF row."""

    record = {
        "data_source": data_source,
        "prompt": build_chat_messages(example, max_turns=max_turns),
        "ability": "software_engineering",
        "reward_model": {"style": "tool", "name": "swebench_sandbox"},
        "extra_info": build_extra_info(example, split, idx),
    }
    return record


def split_dataset(dataset: Dataset, val_ratio: float, seed: int) -> DatasetSplit:
    """Split dataset into train / val subsets with deterministic shuffling."""

    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")

    total = len(dataset)
    indices = np.random.default_rng(seed).permutation(total)

    val_size = int(total * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = dataset.select(train_indices.tolist())
    val_dataset = dataset.select(val_indices.tolist())

    return DatasetSplit(train=train_dataset, val=val_dataset)


def convert_split(split_name: str, dataset: Dataset, data_source: str, max_turns: int = 6) -> list[dict[str, Any]]:
    """Convert a HuggingFace dataset split into RLHF-compatible records."""

    rows = []
    for idx, example in enumerate(dataset):
        rows.append(build_record(example, split_name, idx, data_source, max_turns=max_turns))
    return rows


def save_parquet(records: list[dict[str, Any]], output_path: Path) -> None:
    """Persist records to parquet using HuggingFace Dataset."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).to_parquet(str(output_path))


def run_conversion(
    dataset_name: str,
    output_dir: Path,
    val_ratio: float,
    seed: int,
    sample_limit: int | None = None,
    max_turns: int = 6,
) -> None:
    """Main driver to convert SWEbench dataset into verl parquet files."""

    raw_dataset = load_dataset(dataset_name, split="test")
    if sample_limit is not None:
        raw_dataset = raw_dataset.select(range(min(sample_limit, len(raw_dataset))))

    split = split_dataset(raw_dataset, val_ratio, seed)

    train_records = convert_split("train", split.train, dataset_name, max_turns=max_turns)
    val_records = convert_split("val", split.val, dataset_name, max_turns=max_turns)

    save_parquet(train_records, output_dir / "train.parquet")
    save_parquet(val_records, output_dir / "val.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SWE-bench dataset into verl RLHF format.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset name or path.")
    parser.add_argument(
        "--output_dir",
        default="~/data/swebench_mini/rl",
        help="Directory to store generated parquet files.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Portion of samples reserved for validation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=None,
        help="Optional cap on number of samples to process (useful for smoke tests).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=6,
        help="Maximum assistant turns allowed in the prompt (default: 6).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    max_turns = getattr(args, 'max_turns', 6)  # Handle both max-turns and max_turns
    run_conversion(
        dataset_name=args.dataset,
        output_dir=output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        sample_limit=args.sample_limit,
        max_turns=max_turns,
    )


if __name__ == "__main__":
    main()
