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
from datasets import Dataset, load_dataset

DEFAULT_DATASET = "MariusHobbhahn/swe-bench-verified-mini"
SYSTEM_PROMPT = (
    "You are an autonomous software engineer tasked with resolving issues in real-world repositories. "
    "Your workspace is a persistent SWEbench sandbox that you can access via the `run_swebench_tests` tool. "
    "Use the tool with different actions to explore and validate your work:\n"
    "- `action=\"run_shell\"`: run shell commands (e.g., inspect files, run tests). "
    "Each command starts fresh in the repository root (/workspace/testbed), so use `cd dir && command` if you need to work in subdirectories.\n"
    "- `action=\"read_file\"` / `action=\"write_file\"`: read or modify files inside the sandbox.\n"
    "- `action=\"submit_patch\"`: apply your final diff, run the official SWEbench judge, and end the sandbox session.\n\n"
    "Important: Shell state (like current directory) doesn't persist between commands. "
    "Use `cd directory && your_command` to run commands in specific directories. "
    "However, file changes are persistent across all commands.\n\n"
    "You have a maximum of 6 turns to solve the issue. Plan your approach efficiently and make sure to call "
    "`action=\"submit_patch\"` before running out of turns.\n\n"
    "Think carefully, plan your next action, and iteratively refine your solution using these capabilities until the tests pass."
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

Instructions:
- Analyse the repository state and reason step by step.
- Use `run_swebench_tests` with the appropriate action to explore the sandbox.
  - Run shell commands with `action="run_shell"` to inspect or test your changes.
  - Read or write files with `action="read_file"` / `action="write_file"`.
  - Call `action="submit_patch"` (supplying your unified diff via the `patch` field) only when you believe the fix is ready; this runs the judge and ends the session.
- Provide unified diff patches when you believe the issue is resolved.
"""


@dataclass(frozen=True)
class DatasetSplit:
    train: Dataset
    val: Dataset


def _format_section(title: str, items: Iterable[str]) -> str:
    items = list(items)
    if not items:
        return f"- (none)"
    return "\n".join(f"- {item}" for item in items)


def _build_user_prompt(example: dict[str, Any]) -> str:
    hints = example.get("hints_text") or "(none provided)"
    fail_to_pass = _format_section("Fail-to-pass tests", example.get("FAIL_TO_PASS", []))
    pass_to_pass = _format_section("Pass-to-pass tests", example.get("PASS_TO_PASS", []))

    # Replace placeholders in template
    prompt = USER_PROMPT_TEMPLATE.format(
        repo=example["repo"],
        base_commit=example["base_commit"],
        environment_commit=example.get("environment_setup_commit", "unknown"),
        problem_statement=example["problem_statement"],
        hints=hints,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
    )
    return prompt.strip()


def build_chat_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    """Create chat template compatible messages for verl."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(example)},
    ]


def build_tools_kwargs(example: dict[str, Any]) -> dict[str, Any]:
    """Construct per-instance tool kwargs for the SWEbench sandbox tool."""

    return {
        "run_swebench_tests": {
            "create_kwargs": {
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
            },
        }
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


def build_record(example: dict[str, Any], split: str, idx: int, data_source: str) -> dict[str, Any]:
    """Transform a HuggingFace SWE-bench example into verl RLHF row."""

    record = {
        "data_source": data_source,
        "prompt": build_chat_messages(example),
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


def convert_split(split_name: str, dataset: Dataset, data_source: str) -> list[dict[str, Any]]:
    """Convert a HuggingFace dataset split into RLHF-compatible records."""

    rows = []
    for idx, example in enumerate(dataset):
        rows.append(build_record(example, split_name, idx, data_source))
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
) -> None:
    """Main driver to convert SWEbench dataset into verl parquet files."""

    raw_dataset = load_dataset(dataset_name, split="test")
    if sample_limit is not None:
        raw_dataset = raw_dataset.select(range(min(sample_limit, len(raw_dataset))))

    split = split_dataset(raw_dataset, val_ratio, seed)

    train_records = convert_split("train", split.train, dataset_name)
    val_records = convert_split("val", split.val, dataset_name)

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    run_conversion(
        dataset_name=args.dataset,
        output_dir=output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        sample_limit=args.sample_limit,
    )


if __name__ == "__main__":
    main()
