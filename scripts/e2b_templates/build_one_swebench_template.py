#!/usr/bin/env python3
"""
Build a single E2B 2.0 sandbox template for one SWE-bench instance (dev sanity).

References
- E2B Build System 2.0 docs (code-first templates):
  * Quickstart: templates, build_dev/build_prod scripts
  * Base image helpers, set_user, set_workdir, apt_install, pip_install, run_cmd
- E2B Template API (Python): Template(), Template.build(...), Sandbox(...)
- SWE-bench harness TestSpec + python env/repo scripts

Flow (one instance)
1) Load one SWE-bench instance from HF and construct TestSpec via make_test_spec.
2) Create an E2B Template that:
   - Starts from an Ubuntu/Python base (or e2b base image).
   - Installs minimal system packages (git, curl, build-essential).
   - Installs Miniconda under /opt/miniconda3 and configures shell activation.
   - Pre-creates /workspace and sets it as the workdir.
   - Builds the SWE-bench "testbed" conda environment for this instance by reproducing
     the harness’s env setup (requirements.txt or environment.yml).
3) Builds the template with an alias based on (repo, env hash) so it can be reused.

Note
- This script only defines the build logic. Do NOT run here. The user will execute it
  with proper credentials (`E2B_API_KEY` in .env) and decide when to upload templates.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
import argparse
from datasets import load_dataset

from e2b import Template, default_build_logger

from swebench.harness.test_spec.test_spec import TestSpec, make_test_spec
from swebench.harness.test_spec.python import get_environment_yml, get_requirements
from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS


# -------- Helpers ---------

def _short_env_hash(env_image_key: str) -> str:
    """Extract the short hash from TestSpec.env_image_key.

    Key format (example): sweb.env.py.x86_64.<HASH>:latest
    """
    try:
        core = env_image_key.split(":", 1)[0]
        parts = core.split(".")
        return parts[-1][:10]
    except Exception:
        return "unknownhash"


def _safe_repo_alias(repo: str) -> str:
    # Convert owner/repo to owner_repo
    return repo.replace("/", "_")


@dataclass
class BuildConfig:
    dataset: str = "princeton-nlp/SWE-bench"
    split: str = "test"
    index: int = 0  # which instance in the split to use
    alias_prefix: str = "swebench"
    cpu_count: int = 2
    memory_mb: int = 4096
    workdir: str = "/workspace"
    conda_prefix: str = "/opt/miniconda3"
    env_name: str = "testbed"
    file_context_path: str = "."  # keep small; rely on ignore patterns
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
    # where to place transient files copied into image (e.g., environment.yml)
    local_artifacts_dir: str = "build_artifacts"


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


def build_single_template(cfg: BuildConfig) -> None:
    load_dotenv()

    # Select an instance
    ds = load_dataset(cfg.dataset, split=cfg.split)
    ex = ds[int(cfg.index)]
    spec: TestSpec = make_test_spec(ex)

    repo = spec.repo
    env_key = spec.env_image_key
    env_hash = _short_env_hash(env_key)
    alias = f"{cfg.alias_prefix}-{_safe_repo_alias(repo)}-{env_hash}"

    # Resolve env creation method
    specs = MAP_REPO_VERSION_TO_SPECS[repo][spec.version]
    packages = specs.get("packages", "")
    python_ver = specs.get("python")
    pip_packages = specs.get("pip_packages")

    dockerfile = _render_dockerfile_base(cfg.conda_prefix, cfg.workdir)

    # Build the testbed env according to SWE-bench spec
    # Use bash -lc for conda activation commands
    if packages == "environment.yml":
        env_yml = get_environment_yml(ex, cfg.env_name)
        dockerfile += "\n" + f"""
# --- SWEbench environment (environment.yml) ---
USER root
RUN bash -lc "cat > /root/environment.yml <<'EOF_ENV'\n{env_yml}\nEOF_ENV"
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
RUN bash -lc "cat > /root/requirements.txt <<'EOF_REQ'\n{reqs_text}\nEOF_REQ"
RUN bash -lc "source {cfg.conda_prefix}/bin/activate && conda activate {cfg.env_name} && python -m pip install -r /root/requirements.txt"
USER user
"""
    else:
        # Explicit conda packages list (string). Create env accordingly.
        pkg_str = str(packages).strip()
        dockerfile += "\nUSER root\n"
        if pkg_str and pkg_str != "None":
            dockerfile += f"RUN bash -lc \"source {cfg.conda_prefix}/bin/activate && conda create -n {cfg.env_name} python={python_ver} {pkg_str} -y\"\n"
        else:
            dockerfile += f"RUN bash -lc \"source {cfg.conda_prefix}/bin/activate && conda create -n {cfg.env_name} python={python_ver} -y\"\n"
        if pip_packages:
            dockerfile += f"RUN bash -lc \"source {cfg.conda_prefix}/bin/activate && conda activate {cfg.env_name} && python -m pip install {' '.join(pip_packages)}\"\n"
        dockerfile += "USER user\n"

    # Optionally set a no-op start command; sandbox will be ready immediately
    # from e2b import wait_for_timeout  # could be used for readiness
    # template.set_start_cmd("echo SWEbench template ready")

    template = Template().from_dockerfile(dockerfile)
    Template.build(template, alias=alias, cpu_count=cfg.cpu_count, memory_mb=cfg.memory_mb, on_build_logs=default_build_logger())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build a single SWE-bench E2B template (dev sanity)")
    ap.add_argument("--dataset", default="princeton-nlp/SWE-bench", help="HF dataset name")
    ap.add_argument("--split", default="test", help="Dataset split (default: test)")
    ap.add_argument("--index", type=int, default=0, help="Index of instance in split (default: 0)")
    ap.add_argument("--alias-prefix", default="swebench", help="Template alias prefix")
    ap.add_argument("--cpu", type=int, default=2, help="CPU count for build VM")
    ap.add_argument("--mem", type=int, default=4096, help="Memory (MB) for build VM")
    ap.add_argument("--workdir", default="/workspace", help="Workdir inside the template")
    ap.add_argument("--conda-prefix", default="/opt/miniconda3", help="Miniconda install prefix")
    ap.add_argument("--env-name", default="testbed", help="Conda environment name to create")
    args = ap.parse_args()

    cfg = BuildConfig(
        dataset=args.dataset,
        split=args.split,
        index=args.index,
        alias_prefix=args.alias_prefix,
        cpu_count=args.cpu,
        memory_mb=args.mem,
        workdir=args.workdir,
        conda_prefix=args.conda_prefix,
        env_name=args.env_name,
    )

    print(
        f"Building one template: dataset={cfg.dataset} split={cfg.split} index={cfg.index} "
        f"alias_prefix={cfg.alias_prefix} cpu={cfg.cpu_count} mem={cfg.memory_mb}"
    )
    build_single_template(cfg)
