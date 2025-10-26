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
"""Docker-based execution helpers for SWEbench sandbox runs."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

import docker
from docker.errors import DockerException

from swebench.harness.constants import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION
from swebench.harness.docker_build import build_env_images
from swebench.harness.run_evaluation import run_instance
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.utils import EvaluationError

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(slots=True)
class SandboxConfig:
    """Configuration for SWEbench Docker execution."""

    image: str
    workdir: Path
    repo_cache: Path
    timeout_seconds: int = 900
    log_dir: Path = Path("~/verl_swebench_logs").expanduser()
    docker_binary: str = "docker"
    dry_run: bool = False

    def __post_init__(self) -> None:
        self.repo_cache = self.repo_cache.expanduser()
        self.log_dir = self.log_dir.expanduser()
        self.repo_cache.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


def evaluate_patch(cfg: SandboxConfig, instance: dict[str, Any], patch: str) -> dict[str, Any]:
    """Execute SWEbench harness inside Docker and return evaluation summary."""

    if cfg.dry_run:
        LOGGER.info("SWEbench sandbox dry-run: instance=%s", instance.get("instance_id"))
        return {"status": "dry_run", "logs": "Dry run: no execution performed.", "reward": 0.0}

    dataset_instance = instance.get("dataset_instance") or instance

    try:
        test_spec = make_test_spec(dataset_instance)
    except Exception as exc:  # pragma: no cover - schema issues logged
        LOGGER.exception("Failed to build TestSpec for %s", instance.get("instance_id"))
        return {"status": "spec_error", "logs": str(exc), "reward": 0.0}

    try:
        client = docker.from_env()
    except DockerException as exc:  # pragma: no cover - environment issue
        LOGGER.exception("Failed to initialise Docker client")
        return {"status": "docker_not_available", "logs": str(exc), "reward": 0.0}

    try:
        # Ensure environment images exist for the instance.
        build_env_images(client, [test_spec], force_rebuild=False)

        prediction = {
            KEY_INSTANCE_ID: test_spec.instance_id,
            KEY_MODEL: "verl-agent",
            KEY_PREDICTION: patch,
        }

        run_id = f"verl-{test_spec.instance_id}"
        result = run_instance(
            test_spec,
            prediction,
            rm_image=False,
            force_rebuild=False,
            client=client,
            run_id=run_id,
            timeout=cfg.timeout_seconds,
        )

        resolved = bool(result.get("resolved"))
        status = "passed" if resolved else "failed"
        logs = f"SWEbench resolved={resolved}"
        return {"status": status, "logs": logs, "reward": 1.0 if resolved else 0.0}

    except EvaluationError as exc:  # pragma: no cover - harness failure path
        LOGGER.exception("SWEbench evaluation failed for %s", test_spec.instance_id)
        return {"status": "evaluation_error", "logs": str(exc), "reward": 0.0}

    except Exception as exc:  # pragma: no cover - unexpected errors
        LOGGER.exception("Unexpected error during SWEbench evaluation")
        return {"status": "unexpected_error", "logs": str(exc), "reward": 0.0}

    finally:
        client.close()


__all__ = ["SandboxConfig", "evaluate_patch"]

