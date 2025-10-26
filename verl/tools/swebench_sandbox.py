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
"""SWEbench sandbox tool relying on Docker-backed execution."""

from __future__ import annotations

import logging
import os
import textwrap
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

from .swebench_docker import SandboxConfig, evaluate_patch

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class SWEbenchSandboxTool(BaseTool):
    """Tool that evaluates candidate patches using the SWEbench harness."""

    DEFAULT_SCHEMA = OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": "run_swebench_tests",
                "description": textwrap.dedent(
                    """
                    Apply a candidate patch to the repository and execute the
                    SWEbench evaluation harness. Returns pass/fail status and
                    captured logs.
                    """
                ).strip(),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patch": {
                            "type": "string",
                            "description": "Unified diff to apply to the repository.",
                        },
                        "notes": {
                            "type": "string",
                            "description": "Optional metadata or rationale (not interpreted).",
                        },
                    },
                    "required": ["patch"],
                },
            },
        }
    )

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema]):
        super().__init__(config, tool_schema or self.DEFAULT_SCHEMA)
        self._instances: dict[str, dict[str, Any]] = {}
        self.sandbox_cfg = SandboxConfig(
            image=config.get("docker_image", "swebench/base:latest"),
            workdir=Path(config.get("workdir", "/workspace")),
            repo_cache=Path(config.get("repo_cache", "~/.cache/swebench/repos")),
            timeout_seconds=config.get("timeout_seconds", 900),
            log_dir=Path(config.get("log_dir", "~/verl_swebench_logs")),
            docker_binary=config.get("docker_binary", "docker"),
            dry_run=config.get("dry_run", False),
        )

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:  # noqa: D401
        return self.tool_schema

    async def create(
        self,
        instance_id: Optional[str] = None,
        **kwargs,
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())

        create_kwargs = kwargs.get("create_kwargs", {})
        self._instances[instance_id] = {
            "repo": create_kwargs.get("repo"),
            "base_commit": create_kwargs.get("base_commit"),
            "instance_id": create_kwargs.get("instance_id", instance_id),
            "environment_setup_commit": create_kwargs.get("environment_setup_commit"),
            "version": create_kwargs.get("version"),
            "fail_to_pass": create_kwargs.get("fail_to_pass", []),
            "pass_to_pass": create_kwargs.get("pass_to_pass", []),
            "dataset_instance": create_kwargs.get("dataset_instance"),
            "last_reward": 0.0,
        }
        return instance_id, ToolResponse(text="SWEbench sandbox initialised.")

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs,
    ) -> tuple[ToolResponse, float, dict]:
        if instance_id not in self._instances:
            raise ValueError(f"Unknown SWEbench sandbox instance: {instance_id}")

        patch = parameters.get("patch")
        if not isinstance(patch, str) or not patch.strip():
            return ToolResponse(text="No patch provided."), 0.0, {"status": "invalid_patch"}

        metadata = self._instances[instance_id]
        result = evaluate_patch(
            cfg=self.sandbox_cfg,
            instance=metadata,
            patch=patch,
        )

        reward = float(result.get("reward", 0.0))
        metadata["last_reward"] = reward
        logs = result.get("logs", "")
        status = result.get("status", "unknown")
        return ToolResponse(text=logs), reward, {"status": status}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:  # noqa: D401
        if instance_id not in self._instances:
            raise ValueError(f"Unknown SWEbench sandbox instance: {instance_id}")
        return float(self._instances[instance_id].get("last_reward", 0.0))

    async def release(self, instance_id: str, **kwargs) -> None:  # noqa: D401
        self._instances.pop(instance_id, None)


__all__ = ["SWEbenchSandboxTool"]
