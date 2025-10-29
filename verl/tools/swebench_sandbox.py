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
"""SWEbench sandbox tool that executes patches using E2B cloud sandboxes."""

from __future__ import annotations

import json
import logging
import os
import textwrap
import time
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from dotenv import dotenv_values

from e2b_code_interpreter import Sandbox as E2BSandbox
from e2b.sandbox_sync.commands.command_handle import CommandExitException, CommandResult

from swebench.harness.constants import KEY_INSTANCE_ID
from swebench.harness.test_spec.test_spec import TestSpec, make_test_spec

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def load_e2b_api_key() -> str:
    env_key = os.getenv("E2B_API_KEY")
    if env_key:
        return env_key

    env_path = Path.cwd() / ".env"
    if env_path.exists():
        values = dotenv_values(env_path)
        env_key = values.get("E2B_API_KEY")
        if env_key:
            os.environ["E2B_API_KEY"] = env_key
            return env_key

    raise RuntimeError(
        "E2B_API_KEY is required for SWEbench sandbox execution. Set it in the environment or .env."
    )


class SWEbenchSandboxTool(BaseTool):
    """Tool that evaluates candidate patches using an E2B sandbox."""

    DEFAULT_SCHEMA = OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": "run_swebench_tests",
                "description": textwrap.dedent(
                    """
                    Interact with a persistent SWEbench sandbox hosted on E2B.
                    Available actions:
                      - run_shell: run a shell command inside the sandbox and capture stdout/stderr.
                      - read_file: read a file from the sandbox.
                      - write_file: write content to a file inside the sandbox.
                      - submit_patch: apply a unified diff and run the SWEbench evaluation harness.
                    If no action is provided, the tool defaults to submit_patch (legacy behaviour).
                    """
                ).strip(),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "One of: run_shell, read_file, write_file, submit_patch.",
                        },
                        "command": {
                            "type": "string",
                            "description": "Shell command to execute when action == run_shell.",
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Working directory for shell commands (defaults to repo root).",
                        },
                        "path": {
                            "type": "string",
                            "description": "File path for read_file or write_file actions.",
                        },
                        "content": {
                            "type": "string",
                            "description": "File content for write_file action.",
                        },
                        "patch": {
                            "type": "string",
                            "description": "Unified diff to apply when submitting a patch. Ignored for other actions.",
                        },
                        "notes": {
                            "type": "string",
                            "description": "Optional metadata or rationale (not interpreted).",
                        },
                    },
                    "required": [],
                },
            },
        }
    )

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema]):
        super().__init__(config, tool_schema or self.DEFAULT_SCHEMA)
        self._instances: dict[str, dict[str, Any]] = {}
        self.timeout_seconds = config.get("timeout_seconds", 900)
        self.install_timeout_seconds = config.get("install_timeout_seconds", 300)
        self.enable_e2b = config.get("enable_e2b", True)
        if self.enable_e2b:
            load_e2b_api_key()
        self.template = config.get("template")
        self.workspace = config.get("workdir", "/workspace")
        self.repo_path = config.get("repo_path", "/testbed")
        self.env_setup_timeout_seconds = config.get(
            "env_setup_timeout_seconds", self.install_timeout_seconds
        )
        self.repo_setup_timeout_seconds = config.get(
            "repo_setup_timeout_seconds", self.install_timeout_seconds
        )
        self.eval_timeout_seconds = config.get("eval_timeout_seconds", self.timeout_seconds)
        log_dir = config.get("log_dir")
        self.log_dir: Optional[Path] = Path(os.path.expanduser(log_dir)).expanduser() if log_dir else None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = config.get("model_name", "verl_agent")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:  # noqa: D401
        return self.tool_schema

    async def create(
        self,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> tuple[str, ToolResponse]:
        # SGLang passes request_id positionally while our dataset metadata also
        # encodes an instance_id inside create_kwargs. Prefer the explicit value
        # if provided and avoid duplicate binding errors.
        embedded_instance_id = kwargs.pop("instance_id", None)
        instance_id = request_id or embedded_instance_id
        if request_id is not None and embedded_instance_id is not None and request_id != embedded_instance_id:
            # Keep the first occurrence (SGLang request id) and drop the
            # embedded copy so kwargs don't carry both.
            logging.getLogger(__name__).debug(
                "Ignoring embedded instance_id %s; using request id %s",
                embedded_instance_id,
                request_id,
            )
        if instance_id is None:
            instance_id = str(uuid4())

        create_kwargs = kwargs.get("create_kwargs")
        if create_kwargs is None:
            create_kwargs = kwargs
        dataset_instance = create_kwargs.get("dataset_instance")
        if dataset_instance is None:
            raise ValueError("dataset_instance is required in tools_kwargs for SWEbench execution")

        # This executor installs dependencies per sandbox run; disable for now if too slow.
        test_spec: TestSpec = make_test_spec(dataset_instance)
        if self.repo_path != "/testbed":
            for attr in ("repo_script_list", "env_script_list", "eval_script_list"):
                setattr(
                    test_spec,
                    attr,
                    [cmd.replace("/testbed", self.repo_path) for cmd in getattr(test_spec, attr)],
                )

        record: dict[str, Any] = {
            "dataset_instance": dataset_instance,
            "test_spec": test_spec,
            "last_reward": 0.0,
            "logs": {},
        }

        if not self.enable_e2b:
            self._instances[instance_id] = record
            return instance_id, ToolResponse(text="SWEbench sandbox initialised (offline mode).")

        sandbox_kwargs: dict[str, Any] = {"timeout": self.timeout_seconds}
        if self.template:
            sandbox_kwargs["template"] = self.template

        sandbox: Optional[E2BSandbox] = None
        stage_logs: list[str] = []

        try:
            sandbox = E2BSandbox.create(**sandbox_kwargs)
            record["sandbox"] = sandbox

            # Prepare workspace directory and permissions.
            self._run_command(
                sandbox,
                f"mkdir -p {self.workspace}",
                timeout=60,
                desc="prepare workspace",
            )
            if self.repo_path.startswith("/workspace"):
                self._run_command(
                    sandbox,
                    f"mkdir -p {self.repo_path} && chmod -R 777 {self.repo_path}",
                    timeout=60,
                    desc="prepare repo directory",
                )

            # Accept conda terms (best-effort).
            tos_cmd = (
                "bash -lc '"
                "source /opt/miniconda3/etc/profile.d/conda.sh && "
                "conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && "
                "conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r'"
            )
            tos_result = self._run_command(
                sandbox,
                tos_cmd,
                timeout=120,
                desc="Accept conda ToS",
                allow_error=True,
            )
            stage_logs.append(self._format_stage("Conda terms acceptance", tos_result))

            env_result = self._run_script(
                sandbox,
                script_content=test_spec.setup_env_script,
                remote_name="setup_env.sh",
                timeout=self.env_setup_timeout_seconds,
                stage_name="Environment setup",
            )
            stage_logs.append(self._format_stage("Environment setup", env_result))

            repo_result = self._run_script(
                sandbox,
                script_content=test_spec.install_repo_script,
                remote_name="install_repo.sh",
                timeout=self.repo_setup_timeout_seconds,
                stage_name="Repository setup",
            )
            stage_logs.append(self._format_stage("Repository setup", repo_result))

            record["logs"]["setup"] = "\n\n".join(stage_logs)
            self._persist_logs(instance_id, "setup", record["logs"]["setup"])
            self._instances[instance_id] = record

            return instance_id, ToolResponse(text="SWEbench sandbox ready.")
        except Exception as exc:
            if sandbox is not None:
                try:
                    sandbox.close()
                except Exception:  # pragma: no cover - best effort cleanup
                    pass
            raise exc

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs,
    ) -> tuple[ToolResponse, float, dict]:
        if instance_id not in self._instances:
            raise ValueError(f"Unknown SWEbench sandbox instance: {instance_id}")

        record = self._instances[instance_id]
        action = parameters.get("action")
        if not action:
            action = "submit_patch"

        action = action.lower()

        if action == "run_shell":
            return self._execute_run_shell(instance_id, record, parameters)
        if action == "read_file":
            return self._execute_read_file(instance_id, record, parameters)
        if action == "write_file":
            return self._execute_write_file(instance_id, record, parameters)
        if action == "submit_patch":
            patch = parameters.get("patch")
            notes = parameters.get("notes", "")
            if not isinstance(patch, str) or not patch.strip():
                return ToolResponse(text="No patch provided."), 0.0, {"status": "invalid_patch"}
            return self._execute_submit_patch(instance_id, record, patch, notes)

        return ToolResponse(text=f"Unknown action '{action}'. Nothing executed."), 0.0, {"status": "unknown_action"}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:  # noqa: D401
        if instance_id not in self._instances:
            raise ValueError(f"Unknown SWEbench sandbox instance: {instance_id}")
        return float(self._instances[instance_id].get("last_reward", 0.0))

    async def release(self, instance_id: str, **kwargs) -> None:  # noqa: D401
        record = self._instances.pop(instance_id, None)
        if not record:
            return
        sandbox: Optional[E2BSandbox] = record.get("sandbox")
        if sandbox is not None:
            try:
                sandbox.close()
            except Exception:  # pragma: no cover - best effort cleanup
                LOGGER.exception("Failed to close sandbox for %s", instance_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _remote_path(self, filename: str) -> str:
        workspace = self.workspace.rstrip("/")
        return f"{workspace}/{filename}"

    def _run_command(
        self,
        sandbox: E2BSandbox,
        cmd: str,
        *,
        timeout: int,
        desc: str,
        cwd: Optional[str] = None,
        allow_error: bool = False,
    ) -> CommandResult:
        try:
            return sandbox.commands.run(
                cmd,
                cwd=cwd,
                timeout=timeout,
                request_timeout=timeout,
            )
        except CommandExitException as exc:
            if allow_error:
                return exc
            message = textwrap.dedent(
                f"""
                {desc} failed with exit_code={exc.exit_code}
                stdout:
                {exc.stdout}

                stderr:
                {exc.stderr}
                """
            ).strip()
            raise RuntimeError(message) from exc

    def _write_remote_file(self, sandbox: E2BSandbox, remote_path: str, content: str) -> None:
        sandbox.files.write(remote_path, content)

    def _format_stage(self, stage: str, result: CommandResult) -> str:
        parts = [f"## {stage}", f"exit_code: {result.exit_code}"]
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if stdout:
            parts.append("stdout:")
            parts.append(stdout)
        if stderr:
            parts.append("stderr:")
            parts.append(stderr)
        return "\n".join(parts)

    def _collect_output(self, result: CommandResult) -> str:
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        if stdout and stderr:
            return f"{stdout.rstrip()}\n{stderr}"
        return stdout or stderr

    def _persist_logs(self, instance_id: str, status: str, log_text: str) -> None:
        if self.log_dir is None:
            return
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_path = self.log_dir / f"{instance_id}_{status}_{timestamp}.log"
        log_path.write_text(log_text)

    def _run_script(
        self,
        sandbox: E2BSandbox,
        *,
        script_content: str,
        remote_name: str,
        timeout: int,
        stage_name: str,
        allow_error: bool = False,
    ) -> CommandResult:
        remote_path = self._remote_path(remote_name)
        self._write_remote_file(sandbox, remote_path, script_content)
        return self._run_command(
            sandbox,
            f"bash {remote_path}",
            timeout=timeout,
            desc=stage_name,
            allow_error=allow_error,
        )

    def _evaluate_patch(self, dataset_instance: dict[str, Any], patch: str) -> tuple[str, str, float]:
        """Run SWEbench evaluation inside an E2B sandbox."""

        gold_patch = dataset_instance.get("patch", "")

        if not self.enable_e2b:
            success = patch.strip() == gold_patch.strip()
            status = "passed" if success else "failed"
            reward = 1.0 if success else 0.0
            return status, "E2B execution disabled (offline mode).", reward

        test_spec: TestSpec = make_test_spec(dataset_instance)
        if self.repo_path != "/testbed":
            for attr in ("repo_script_list", "env_script_list", "eval_script_list"):
                setattr(
                    test_spec,
                    attr,
                    [cmd.replace("/testbed", self.repo_path) for cmd in getattr(test_spec, attr)],
                )
        sandbox_kwargs: dict[str, Any] = {"timeout": self.timeout_seconds}
        if self.template:
            sandbox_kwargs["template"] = self.template

        stage_logs: list[str] = []
        instance_id = dataset_instance.get("instance_id", "unknown-instance")
        status = "execution_error"
        reward = 0.0

        with E2BSandbox.create(**sandbox_kwargs) as sandbox:
            # Prepare workspace directory
            self._run_command(
                sandbox,
                f"mkdir -p {self.workspace}",
                timeout=60,
                desc="prepare workspace",
            )
            if self.repo_path.startswith("/workspace"):
                self._run_command(
                    sandbox,
                    f"mkdir -p {self.repo_path} && chmod -R 777 {self.repo_path}",
                    timeout=60,
                    desc="prepare repo directory",
                )

            tos_cmd = (
                "bash -lc '"
                "source /opt/miniconda3/etc/profile.d/conda.sh && "
                "conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && "
                "conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r'"
            )
            tos_result = self._run_command(
                sandbox,
                tos_cmd,
                timeout=120,
                desc="Accept conda ToS",
                allow_error=True,
            )
            stage_logs.append(self._format_stage("Conda terms acceptance", tos_result))

            env_result = self._run_script(
                sandbox,
                script_content=test_spec.setup_env_script,
                remote_name="setup_env.sh",
                timeout=self.env_setup_timeout_seconds,
                stage_name="Environment setup",
            )
            stage_logs.append(self._format_stage("Environment setup", env_result))

            repo_result = self._run_script(
                sandbox,
                script_content=test_spec.install_repo_script,
                remote_name="install_repo.sh",
                timeout=self.repo_setup_timeout_seconds,
                stage_name="Repository setup",
            )
            stage_logs.append(self._format_stage("Repository setup", repo_result))

            patch_remote_path = self._remote_path("candidate.patch")
            self._write_remote_file(sandbox, patch_remote_path, patch)

            patch_check = self._run_command(
                sandbox,
                f"git apply --check {patch_remote_path}",
                timeout=self.install_timeout_seconds,
                desc="Patch validation",
                cwd=self.repo_path,
                allow_error=True,
            )
            stage_logs.append(self._format_stage("Patch validation", patch_check))
            if patch_check.exit_code != 0:
                log_text = "\n\n".join(stage_logs)
                self._persist_logs(instance_id, "patch_invalid", log_text)
                return "patch_invalid", log_text, 0.0

            patch_apply = self._run_command(
                sandbox,
                f"git apply {patch_remote_path}",
                timeout=self.install_timeout_seconds,
                desc="Patch apply",
                cwd=self.repo_path,
            )
            stage_logs.append(self._format_stage("Patch apply", patch_apply))

            eval_result = self._run_script(
                sandbox,
                script_content=test_spec.eval_script,
                remote_name="run_eval.sh",
                timeout=self.eval_timeout_seconds,
                stage_name="Evaluation",
                allow_error=True,
            )
            stage_logs.append(self._format_stage("Evaluation", eval_result))
            eval_log = self._collect_output(eval_result)

        resolved = eval_result.exit_code == 0
        status = "passed" if resolved else "failed"
        reward = 1.0 if resolved else 0.0
        stage_logs.append("## Evaluation summary")
        stage_logs.append(
            textwrap.dedent(
                f"""
                exit_code: {eval_result.exit_code}
                reward: {reward}
                """
            ).strip()
        )

        log_text = "\n\n".join(stage_logs)
        self._persist_logs(instance_id, status, log_text)
        return status, log_text, reward

    def _execute_run_shell(
        self, instance_id: str, record: dict[str, Any], parameters: dict[str, Any]
    ) -> tuple[ToolResponse, float, dict]:
        if not self.enable_e2b:
            return ToolResponse(text="E2B execution is disabled."), 0.0, {"status": "disabled"}

        sandbox: E2BSandbox = record.get("sandbox")
        if sandbox is None:
            return ToolResponse(text="Sandbox is not initialised."), 0.0, {"status": "no_sandbox"}

        command = parameters.get("command")
        if not command:
            return ToolResponse(text="Missing 'command' parameter for run_shell."), 0.0, {"status": "invalid_command"}

        cwd = parameters.get("cwd") or self.repo_path
        formatted_command = f"bash -lc {json.dumps(command)}"
        result = self._run_command(
            sandbox,
            formatted_command,
            timeout=parameters.get("timeout", self.timeout_seconds),
            desc=f"Shell command ({command})",
            cwd=cwd,
            allow_error=True,
        )
        stage_text = self._format_stage("Shell command", result)
        self._persist_logs(instance_id, "shell", stage_text)
        status = "completed" if result.exit_code == 0 else "failed"
        metrics = {"status": status, "exit_code": result.exit_code}
        return ToolResponse(text=stage_text), 0.0, metrics

    def _execute_read_file(
        self, instance_id: str, record: dict[str, Any], parameters: dict[str, Any]
    ) -> tuple[ToolResponse, float, dict]:
        if not self.enable_e2b:
            return ToolResponse(text="E2B execution is disabled."), 0.0, {"status": "disabled"}

        sandbox: E2BSandbox = record.get("sandbox")
        if sandbox is None:
            return ToolResponse(text="Sandbox is not initialised."), 0.0, {"status": "no_sandbox"}

        path = parameters.get("path")
        if not path:
            return ToolResponse(text="Missing 'path' parameter for read_file."), 0.0, {"status": "invalid_path"}

        try:
            content = sandbox.files.read(path)
        except Exception as exc:  # pragma: no cover - depends on remote FS
            message = f"Failed to read file {path}: {exc}"
            self._persist_logs(instance_id, "read_file_error", message)
            return ToolResponse(text=message), 0.0, {"status": "failed"}

        max_len = parameters.get("max_bytes", 40000)
        truncated = ""
        if isinstance(content, str):
            text = content
        else:
            text = content.decode("utf-8", errors="replace")
        if max_len and len(text) > max_len:
            truncated = f"\n\n[Output truncated at {max_len} bytes]"
            text = text[:max_len]

        response_text = f"Path: {path}\n\n{text}{truncated}"
        self._persist_logs(instance_id, "read_file", response_text)
        return ToolResponse(text=response_text), 0.0, {"status": "completed"}

    def _execute_write_file(
        self, instance_id: str, record: dict[str, Any], parameters: dict[str, Any]
    ) -> tuple[ToolResponse, float, dict]:
        if not self.enable_e2b:
            return ToolResponse(text="E2B execution is disabled."), 0.0, {"status": "disabled"}

        sandbox: E2BSandbox = record.get("sandbox")
        if sandbox is None:
            return ToolResponse(text="Sandbox is not initialised."), 0.0, {"status": "no_sandbox"}

        path = parameters.get("path")
        content = parameters.get("content")
        if not path or content is None:
            return (
                ToolResponse(text="Both 'path' and 'content' parameters are required for write_file."),
                0.0,
                {"status": "invalid_parameters"},
            )

        try:
            sandbox.files.write(path, content)
        except Exception as exc:  # pragma: no cover - depends on remote FS
            message = f"Failed to write file {path}: {exc}"
            self._persist_logs(instance_id, "write_file_error", message)
            return ToolResponse(text=message), 0.0, {"status": "failed"}

        message = f"Wrote {len(content)} bytes to {path}."
        self._persist_logs(instance_id, "write_file", message)
        return ToolResponse(text=message), 0.0, {"status": "completed"}

    def _execute_submit_patch(
        self,
        instance_id: str,
        record: dict[str, Any],
        patch: str,
        notes: str,
    ) -> tuple[ToolResponse, float, dict]:
        if not self.enable_e2b:
            dataset_instance = record["dataset_instance"]
            gold_patch = dataset_instance.get("patch", "")
            success = patch.strip() == gold_patch.strip()
            reward = 1.0 if success else 0.0
            status = "passed" if success else "failed"
            message = "E2B execution disabled. Offline comparison used."
            record["last_reward"] = reward
            return ToolResponse(text=message), reward, {"status": status}

        sandbox: E2BSandbox = record.get("sandbox")
        if sandbox is None:
            return ToolResponse(text="Sandbox is not initialised."), 0.0, {"status": "no_sandbox"}

        test_spec: TestSpec = record["test_spec"]
        dataset_instance = record["dataset_instance"]
        stage_logs: list[str] = []

        repo_path = self.repo_path
        base_commit = dataset_instance["base_commit"]

        # Clean working tree before applying patch.
        reset_cmds = [
            f"git reset --hard {base_commit}",
            "git clean -fd",
        ]
        for cmd in reset_cmds:
            result = self._run_command(
                sandbox,
                cmd,
                timeout=self.install_timeout_seconds,
                desc=f"Repository prepare ({cmd})",
                cwd=repo_path,
                allow_error=False,
            )
            stage_logs.append(self._format_stage("Repository prepare", result))

        patch_remote_path = self._remote_path("candidate.patch")
        self._write_remote_file(sandbox, patch_remote_path, patch)

        patch_check = self._run_command(
            sandbox,
            f"git apply --check {patch_remote_path}",
            timeout=self.install_timeout_seconds,
            desc="Patch validation",
            cwd=repo_path,
            allow_error=True,
        )
        stage_logs.append(self._format_stage("Patch validation", patch_check))
        if patch_check.exit_code != 0:
            log_text = "\n\n".join(stage_logs)
            self._persist_logs(instance_id, "patch_invalid", log_text)
            record["last_reward"] = 0.0
            return (
                ToolResponse(text=f"{log_text}\n\nPatch did not apply cleanly."),
                0.0,
                {"status": "patch_invalid"},
            )

        patch_apply = self._run_command(
            sandbox,
            f"git apply {patch_remote_path}",
            timeout=self.install_timeout_seconds,
            desc="Patch apply",
            cwd=repo_path,
        )
        stage_logs.append(self._format_stage("Patch apply", patch_apply))

        eval_result = self._run_script(
            sandbox,
            script_content=test_spec.eval_script,
            remote_name="run_eval.sh",
            timeout=self.eval_timeout_seconds,
            stage_name="Evaluation",
            allow_error=True,
        )
        stage_logs.append(self._format_stage("Evaluation", eval_result))
        eval_log = self._collect_output(eval_result)

        resolved = eval_result.exit_code == 0
        status = "passed" if resolved else "failed"
        reward = 1.0 if resolved else 0.0
        record["last_reward"] = reward

        stage_logs.append("## Evaluation summary")
        summary = textwrap.dedent(
            f"""
            exit_code: {eval_result.exit_code}
            reward: {reward}
            notes: {notes}
            """
        ).strip()
        stage_logs.append(summary)

        # Reset repository again so the sandbox stays clean for further exploration.
        cleanup_cmds = [
            f"git reset --hard {base_commit}",
            "git clean -fd",
        ]
        for cmd in cleanup_cmds:
            cleanup_result = self._run_command(
                sandbox,
                cmd,
                timeout=self.install_timeout_seconds,
                desc=f"Repository cleanup ({cmd})",
                cwd=repo_path,
                allow_error=True,
            )
            stage_logs.append(self._format_stage("Repository cleanup", cleanup_result))

        log_text = "\n\n".join(stage_logs)
        self._persist_logs(instance_id, status, log_text)

        response_text = f"{log_text}\n\nEvaluation log:\n{eval_log}"
        metrics = {"status": status, "exit_code": eval_result.exit_code}
        return ToolResponse(text=response_text), reward, metrics


__all__ = ["SWEbenchSandboxTool"]
