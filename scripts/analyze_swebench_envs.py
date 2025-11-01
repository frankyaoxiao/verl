#!/usr/bin/env python3
"""
Analyze SWE-bench dataset to estimate the number of prebuild templates needed.

This script groups instances by:
  1) env-only key: TestSpec.env_image_key (hash of env setup script + docker specs)
  2) repo + env key: (repo, env_image_key)
  3) repo + env + branch key (optional): (repo, env_image_key, clone_branch)

It prints a concise summary and can optionally write a JSON report.

Requirements:
  - pip install datasets swebench

Usage examples:
  - python scripts/analyze_swebench_envs.py \
      --dataset princeton-nlp/SWE-bench-verified --split test

  - python scripts/analyze_swebench_envs.py \
      --dataset MariusHobbhahn/swe-bench-verified-mini --split test --limit 64 \
      --output outputs/swebench_env_analysis.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

from datasets import load_dataset

from swebench.harness.test_spec.test_spec import TestSpec, make_test_spec


@dataclass
class GroupCounts:
    total_instances: int
    unique_repos: int
    env_only_templates: int
    repo_env_templates: int
    repo_env_branch_templates: int


def _extract_clone_branch(repo_script_list: list[str]) -> Optional[str]:
    """Best-effort extraction of the branch used in the clone command.

    Looks for lines like:
      git clone -o origin <branch> --single-branch https://github.com/<repo> <dir>
    """
    clone_re = re.compile(r"git\s+clone\s+-o\s+origin\s+([^\s]+)\s+--single-branch\s+https://github.com/")
    for line in repo_script_list:
        m = clone_re.search(line)
        if m:
            return m.group(1)
    return None


def iter_instances(dataset_name: str, split: str, limit: Optional[int], config: Optional[str]) -> Iterable[dict[str, Any]]:
    if config:
        ds = load_dataset(dataset_name, config, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)
    n = len(ds)
    if limit is not None:
        n = min(n, limit)
    for i in range(n):
        yield ds[i]


def analyze(dataset: str, split: str, limit: Optional[int], config: Optional[str]) -> tuple[GroupCounts, dict]:
    env_only_keys: Counter[str] = Counter()
    repo_env_keys: Counter[Tuple[str, str]] = Counter()
    repo_env_branch_keys: Counter[Tuple[str, str, Optional[str]]] = Counter()

    per_group_instances_env_only: Dict[str, list[str]] = defaultdict(list)
    per_group_instances_repo_env: Dict[Tuple[str, str], list[str]] = defaultdict(list)
    per_group_instances_repo_env_branch: Dict[Tuple[str, str, Optional[str]], list[str]] = defaultdict(list)

    repos: set[str] = set()
    total = 0
    errors = 0

    for ex in iter_instances(dataset, split, limit, config):
        try:
            spec: TestSpec = make_test_spec(ex)
        except Exception as exc:  # pragma: no cover - data irregularity
            errors += 1
            continue

        total += 1
        repos.add(spec.repo)

        env_key = spec.env_image_key
        repo_key = spec.repo
        branch = _extract_clone_branch(spec.repo_script_list)

        env_only_keys[env_key] += 1
        repo_env_keys[(repo_key, env_key)] += 1
        repo_env_branch_keys[(repo_key, env_key, branch)] += 1

        instance_id = ex.get("instance_id") or ex.get("INSTANCE_ID") or str(total)
        per_group_instances_env_only[env_key].append(instance_id)
        per_group_instances_repo_env[(repo_key, env_key)].append(instance_id)
        per_group_instances_repo_env_branch[(repo_key, env_key, branch)].append(instance_id)

    summary = GroupCounts(
        total_instances=total,
        unique_repos=len(repos),
        env_only_templates=len(env_only_keys),
        repo_env_templates=len(repo_env_keys),
        repo_env_branch_templates=len(repo_env_branch_keys),
    )

    details = {
        "counts": asdict(summary),
        "errors": errors,
        "env_only_groups": {k: v for k, v in sorted(env_only_keys.items(), key=lambda x: (-x[1], x[0]))},
        "repo_env_groups": {f"{k[0]}::{k[1]}": v for k, v in sorted(repo_env_keys.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))},
        "repo_env_branch_groups": {
            f"{k[0]}::{k[1]}::{k[2]}": v for k, v in sorted(repo_env_branch_keys.items(), key=lambda x: (-x[1], x[0][0], x[0][1], str(x[0][2])))
        },
    }

    return summary, details


def main():
    ap = argparse.ArgumentParser(description="Analyze SWE-bench env/template groupings")
    ap.add_argument("--dataset", required=True, help="HF dataset name or local path (e.g., princeton-nlp/SWE-bench-verified)")
    ap.add_argument("--split", default="test", help="Dataset split (default: test)")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on instances to analyze")
    ap.add_argument("--config", default=None, help="Optional dataset config name (e.g., verified)")
    ap.add_argument("--output", default=None, help="Optional JSON report path")
    args = ap.parse_args()

    summary, details = analyze(args.dataset, args.split, args.limit, args.config)

    print("SWE-bench template analysis")
    print(f"  Dataset: {args.dataset} [{args.split}]")
    print(f"  Total instances: {summary.total_instances}")
    print(f"  Unique repos:    {summary.unique_repos}")
    print(f"  env-only keys:   {summary.env_only_templates}")
    print(f"  repo+env keys:   {summary.repo_env_templates}")
    print(f"  repo+env+branch: {summary.repo_env_branch_templates}")
    if details.get("errors"):
        print(f"  Errors (skipped): {details['errors']}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(details, f, indent=2)
        print(f"\nWrote JSON report to {args.output}")


if __name__ == "__main__":
    main()
