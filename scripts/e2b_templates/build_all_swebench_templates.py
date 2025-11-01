#!/usr/bin/env python3
"""
Build E2B 2.0 sandbox templates for all (repo, env) groups in SWE-bench.

Strategy
- Group instances by TestSpec.env_image_key per repo — this identifies a unique
  environment setup (conda env) for that repo.
- For each group, construct a representative TestSpec and bake its conda environment
  into a template snapshot ("testbed"), so runtime can skip env creation.

Notes
- This script only prepares the build steps using the E2B Template API. Do not run
  in this environment; the user will execute these builds with valid credentials
  and network access.
- You may further optimize by precloning repos or adding mirrors in the template,
  but this script focuses on baking the environment to eliminate the 6–8 minute conda cost.

Docs referenced
- E2B Build System 2.0 (Python): Template(), Template.build(), set_user, set_workdir,
  apt_install, run_cmd, pip_install; base images helpers.
- SWE-bench harness TestSpec + python env creation helpers.
"""

from __future__ import annotations

import os
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

from dotenv import load_dotenv
import argparse
from datasets import load_dataset

from e2b import Template, default_build_logger

from swebench.harness.test_spec.test_spec import TestSpec, make_test_spec
from swebench.harness.test_spec.python import get_environment_yml, get_requirements
from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS


def _short_env_hash(env_image_key: str) -> str:
    try:
        core = env_image_key.split(":", 1)[0]
        parts = core.split(".")
        return parts[-1][:10]
    except Exception:
        return "unknownhash"


def _safe_repo_alias(repo: str) -> str:
    return repo.replace("/", "_")


@dataclass
class BuildConfig:
    dataset: str = "princeton-nlp/SWE-bench"
    split: str = "test"
    alias_prefix: str = "swebench"
    cpu_count: int = 2
    memory_mb: int = 4096
    workdir: str = "/workspace"
    conda_prefix: str = "/opt/miniconda3"
    env_name: str = "testbed"
    file_context_path: str = "."
    file_ignore_patterns: tuple[str, ...] = (
        ".git",
        "**/__pycache__",
        "**/*.bin",
        "**/*.safetensors",
        "**/*.pt",
        "**/*.ckpt",
        "outputs",
        "tmp",
        "tmp_logs",
        "tmp_repo_cache",
    )
    artifacts_dir: str = "build_artifacts"


def _render_dockerfile_base(conda_prefix: str, workdir: str) -> str:
    return f"""
FROM e2bdev/base

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl bzip2 ca-certificates git build-essential wget && \
    rm -rf /var/lib/apt/lists/*

RUN wget -O /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p {conda_prefix} && \
    rm /tmp/miniconda.sh

RUN bash -lc "echo 'source {conda_prefix}/etc/profile.d/conda.sh' >> /etc/profile.d/conda.sh"
RUN bash -lc "source {conda_prefix}/etc/profile.d/conda.sh && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r"
ENV PATH={conda_prefix}/bin:$PATH
RUN mkdir -p {workdir} && chmod -R 777 {workdir}
USER user
WORKDIR /home/user
"""


def _write_text(path: pathlib.Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _iter_groups(dataset: str, split: str) -> Iterable[Tuple[str, str, dict]]:
    ds = load_dataset(dataset, split=split)
    groups: Dict[Tuple[str, str], dict] = {}
    for ex in ds:
        spec: TestSpec = make_test_spec(ex)
        key = (spec.repo, spec.env_image_key)
        if key not in groups:
            groups[key] = ex
    for (repo, env_key), exemplar in groups.items():
        yield repo, env_key, exemplar


def build_all_templates(cfg: BuildConfig) -> None:
    load_dotenv()

    for repo, env_key, ex in _iter_groups(cfg.dataset, cfg.split):
        spec: TestSpec = make_test_spec(ex)

        env_hash = _short_env_hash(env_key)
        alias = f"{cfg.alias_prefix}-{_safe_repo_alias(repo)}-{env_hash}"

        specs = MAP_REPO_VERSION_TO_SPECS[repo][spec.version]
        packages = specs.get("packages", "")
        python_ver = specs.get("python")
        pip_packages = specs.get("pip_packages")

        dockerfile = _render_dockerfile_base(cfg.conda_prefix, cfg.workdir)

        if packages == "environment.yml":
            env_yml = get_environment_yml(ex, cfg.env_name)
            dockerfile += "\n" + f"""
# --- SWEbench environment (environment.yml) ---
USER root
RUN cat > /root/environment.yml <<'EOF_ENV'
{env_yml}
EOF_ENV
RUN bash -lc "source {cfg.conda_prefix}/bin/activate && conda env create -f /root/environment.yml"
"""
            if python_ver:
                dockerfile += f"RUN bash -lc \"source {cfg.conda_prefix}/bin/activate && conda activate {cfg.env_name} && conda install -y python={python_ver}\"\n"
            dockerfile += "USER user\n"
        elif packages == "requirements.txt":
            reqs_text = get_requirements(ex)
            dockerfile += "\n" + f"""
# --- SWEbench environment (requirements.txt) ---
USER root
RUN bash -lc "source {cfg.conda_prefix}/bin/activate && conda create -n {cfg.env_name} python={python_ver} -y"
RUN cat > /root/requirements.txt <<'EOF_REQ'
{reqs_text}
EOF_REQ
RUN bash -lc "source {cfg.conda_prefix}/bin/activate && conda activate {cfg.env_name} && python -m pip install -r /root/requirements.txt"
USER user
"""
        else:
            pkg_str = str(packages).strip()
            dockerfile += "\nUSER root\n"
            if pkg_str and pkg_str != "None":
                dockerfile += f"RUN bash -lc \"source {cfg.conda_prefix}/bin/activate && conda create -n {cfg.env_name} python={python_ver} {pkg_str} -y\"\n"
            else:
                dockerfile += f"RUN bash -lc \"source {cfg.conda_prefix}/bin/activate && conda create -n {cfg.env_name} python={python_ver} -y\"\n"
            if pip_packages:
                dockerfile += f"RUN bash -lc \"source {cfg.conda_prefix}/bin/activate && conda activate {cfg.env_name} && python -m pip install {' '.join(pip_packages)}\"\n"
            dockerfile += "USER user\n"

        template = Template().from_dockerfile(dockerfile)
        Template.build(template, alias=alias, cpu_count=cfg.cpu_count, memory_mb=cfg.memory_mb, on_build_logs=default_build_logger())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build E2B templates for all (repo, env) SWE-bench groups")
    ap.add_argument("--dataset", default="princeton-nlp/SWE-bench", help="HF dataset name")
    ap.add_argument("--split", default="test", help="Dataset split (default: test)")
    ap.add_argument("--alias-prefix", default="swebench", help="Template alias prefix")
    ap.add_argument("--cpu", type=int, default=2, help="CPU count for build VMs")
    ap.add_argument("--mem", type=int, default=4096, help="Memory (MB) for build VMs")
    ap.add_argument("--workdir", default="/workspace", help="Workdir inside templates")
    ap.add_argument("--conda-prefix", default="/opt/miniconda3", help="Miniconda install prefix")
    ap.add_argument("--env-name", default="testbed", help="Conda environment name to create")
    args = ap.parse_args()

    cfg = BuildConfig(
        dataset=args.dataset,
        split=args.split,
        alias_prefix=args.alias_prefix,
        cpu_count=args.cpu,
        memory_mb=args.mem,
        workdir=args.workdir,
        conda_prefix=args.conda_prefix,
        env_name=args.env_name,
    )

    print(
        f"Building ALL templates: dataset={cfg.dataset} split={cfg.split} "
        f"alias_prefix={cfg.alias_prefix} cpu={cfg.cpu_count} mem={cfg.memory_mb}"
    )
    build_all_templates(cfg)
