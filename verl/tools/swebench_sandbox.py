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

import asyncio
import json
import logging
import os
import textwrap
import time
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4
import threading
import posixpath

from dotenv import dotenv_values
from types import SimpleNamespace

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
                        Note: each command starts fresh in /workspace/testbed. Use 'cd dir && command' to work in subdirectories.
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
                            "description": "Shell command to execute when action == run_shell. Each command starts fresh in /workspace/testbed. Use 'cd directory && your_command' to run commands in other directories.",
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
        # Optional: pre-warmed environment skip. If enabled and an environment
        # named `prewarm_env_name` already exists, skip running the harness
        # env setup script to save time (templates may bake the env already).
        env_flag = os.getenv("SWEBENCH_PREWARMED", "1")
        try:
            env_flag_bool = bool(int(env_flag))
        except Exception:
            env_flag_bool = False
        self.prewarm: bool = bool(config.get("prewarm", env_flag_bool))
        self.prewarm_env_name: str = config.get("prewarm_env_name", "testbed")
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

        # Worktree mode settings for per-question persistent sandboxes.
        self.worktree_mode: bool = config.get("worktree_mode", False)
        self.worktree_root: str = config.get("worktree_root", posixpath.join(self.workspace, "worktrees"))
        self._canonicals_lock = threading.Lock()
        self._canonicals: dict[str, dict[str, Any]] = {}
        self._req_to_canon: dict[str, str] = {}
        self._req_worktrees: dict[str, str] = {}

    def _env_exists(self, sandbox: E2BSandbox) -> bool:
        """Check whether the pre-warmed conda env exists in the sandbox.

        Requires conda to be installed at /opt/miniconda3 (as in our templates).
        Best-effort: return False on any error.
        """
        try:
            cmd = (
                "bash -lc '"
                "source /opt/miniconda3/etc/profile.d/conda.sh && "
                "conda env list'"
            )
            res = self._run_command(
                sandbox, cmd, timeout=60, desc="Check conda envs", allow_error=True
            )
            if res.exit_code != 0:
                return False
            out = (res.stdout or "")
            return self.prewarm_env_name in out
        except Exception:
            return False

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
            return instance_id, ToolResponse()

        sandbox_kwargs: dict[str, Any] = {"timeout": self.timeout_seconds}
        # Template selection priority:
        # 1) Per-request override from create_kwargs (e.g., dataset-provided)
        # 2) Tool-level default from config
        req_template = None
        try:
            req_template = (kwargs.get("template") or {}).get("alias")  # tolerate structured input
        except Exception:
            req_template = kwargs.get("template")
        create_kwargs_template = None
        if isinstance(kwargs.get("create_kwargs"), dict):
            create_kwargs_template = kwargs["create_kwargs"].get("template")
        chosen_template = create_kwargs_template or req_template or self.template
        if chosen_template:
            sandbox_kwargs["template"] = chosen_template
        # Log and stage configuration summary for debugging
        config_summary = textwrap.dedent(
            f"""
            Configuration
            template: {chosen_template or '(default)'}
            prewarm: {self.prewarm} (env={self.prewarm_env_name})
            workspace: {self.workspace}
            repo_path: {self.repo_path}
            worktree_mode: {self.worktree_mode}
            worktree_root: {self.worktree_root}
            timeout: {self.timeout_seconds}s
            """
        ).strip()
        LOGGER.info(config_summary.replace("\n", " | "))

        sandbox: Optional[E2BSandbox] = None
        stage_logs: list[str] = []
        stage_logs.append(config_summary)

        def _ensure_canonical_and_worktree():
            nonlocal sandbox, stage_logs
            canonical_id = dataset_instance.get("instance_id", instance_id)
            # Create per-canonical entry and lock if absent
            with self._canonicals_lock:
                if canonical_id not in self._canonicals:
                    self._canonicals[canonical_id] = {"lock": threading.Lock(), "initialized": False}
            canonical = self._canonicals[canonical_id]
            with canonical["lock"]:
                if not canonical.get("initialized", False):
                    t_start = time.time()
                    LOGGER.info(f"[TIMING] {canonical_id[:8]} - Starting E2B sandbox creation")
                    sbox = E2BSandbox.create(**sandbox_kwargs)
                    canonical.update(
                        {
                            "sandbox": sbox,
                            "refcount": 0,
                            "root_repo_path": self.repo_path,
                            "worktree_root": self.worktree_root,
                        }
                    )
                    LOGGER.info(
                        f"[TIMING] {canonical_id[:8]} - E2B sandbox created in {time.time() - t_start:.2f}s"
                    )
                    # Prepare workspace and ToS
                    self._run_command(sbox, f"mkdir -p {self.workspace}", timeout=60, desc="prepare workspace")
                    self._run_command(sbox, f"mkdir -p {self.worktree_root}", timeout=60, desc="prepare worktree root")
                    tos_cmd = (
                        "bash -lc '"
                        "source /opt/miniconda3/etc/profile.d/conda.sh && "
                        "conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && "
                        "conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r'"
                    )
                    tos_result = self._run_command(
                        sbox, tos_cmd, timeout=120, desc="Accept conda ToS", allow_error=True
                    )
                    stage_logs.append(self._format_stage("Conda terms acceptance", tos_result))
                    # Env setup
                    t_env_start = time.time()
                    LOGGER.info(f"[TIMING] {canonical_id[:8]} - Starting environment setup")
                    if self.prewarm and self._env_exists(sbox):
                        # Skip running env script if template is pre-warmed.
                        env_result = SimpleNamespace(
                            exit_code=0,
                            stdout=f"Prewarmed env '{self.prewarm_env_name}' detected; skipping env setup.",
                            stderr="",
                        )
                    else:
                        env_result = self._run_script(
                            sbox,
                            script_content=test_spec.setup_env_script,
                            remote_name="setup_env.sh",
                            timeout=self.env_setup_timeout_seconds,
                            stage_name="Environment setup",
                        )
                    stage_logs.append(self._format_stage("Environment setup", env_result))
                    LOGGER.info(
                        f"[TIMING] {canonical_id[:8]} - Environment setup completed in {time.time() - t_env_start:.2f}s"
                    )
                    # Repo install once
                    self._run_command(
                        sbox, f"rm -rf {self.repo_path}", timeout=60, desc="clean testbed directory", allow_error=True
                    )
                    t_repo_start = time.time()
                    LOGGER.info(f"[TIMING] {canonical_id[:8]} - Starting repository setup")
                    repo_result = self._run_script(
                        sbox,
                        script_content=test_spec.install_repo_script,
                        remote_name="install_repo.sh",
                        timeout=self.repo_setup_timeout_seconds,
                        stage_name="Repository setup",
                    )
                    stage_logs.append(self._format_stage("Repository setup", repo_result))
                    LOGGER.info(
                        f"[TIMING] {canonical_id[:8]} - Repository setup completed in {time.time() - t_repo_start:.2f}s"
                    )
                    canonical["initialized"] = True
                    self._persist_logs(canonical_id, "setup", "\n\n".join(stage_logs))

                # Add a per-request worktree for this repeat
                sbox = canonical["sandbox"]
                worktree_path = posixpath.join(self.worktree_root, instance_id)
                base_commit = dataset_instance["base_commit"]
                self._run_command(sbox, f"mkdir -p {self.worktree_root}", timeout=60, desc="mkdir worktrees root")
                add_cmd = f"git -C {self.repo_path} worktree add -f {worktree_path} {base_commit}"
                wt_add = self._run_command(
                    sbox, add_cmd, timeout=self.repo_setup_timeout_seconds, desc="Add worktree", allow_error=True
                )
                if wt_add.exit_code != 0:
                    self._run_command(
                        sbox, f"git -C {self.repo_path} worktree prune", timeout=120, desc="Prune worktrees", allow_error=True
                    )
                    self._run_command(
                        sbox, add_cmd, timeout=self.repo_setup_timeout_seconds, desc="Add worktree (retry)", allow_error=False
                    )
                self._run_command(
                    sbox,
                    f"git config --global --add safe.directory {worktree_path}",
                    timeout=60,
                    desc="Mark worktree safe",
                    allow_error=True,
                )
                canonical["refcount"] = canonical.get("refcount", 0) + 1
                self._req_to_canon[instance_id] = canonical_id
                self._req_worktrees[instance_id] = worktree_path
                record["sandbox"] = sbox
                record["worktree_path"] = worktree_path
                record["canonical_id"] = canonical_id
                self._instances[instance_id] = record

        # Legacy single-sandbox setup per request
        def _do_setup():
            nonlocal sandbox, stage_logs
            # Create sandbox (synchronous)
            t_start = time.time()
            LOGGER.info(f"[TIMING] {instance_id[:8]} - Starting E2B sandbox creation")
            sandbox = E2BSandbox.create(**sandbox_kwargs)
            record["sandbox"] = sandbox
            LOGGER.info(f"[TIMING] {instance_id[:8]} - E2B sandbox created in {time.time() - t_start:.2f}s")

            # Prepare workspace directory and permissions.
            self._run_command(
                sandbox,
                f"mkdir -p {self.workspace}",
                timeout=60,
                desc="prepare workspace",
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

            t_env_start = time.time()
            LOGGER.info(f"[TIMING] {instance_id[:8]} - Starting environment setup")
            if self.prewarm and self._env_exists(sandbox):
                env_result = SimpleNamespace(
                    exit_code=0,
                    stdout=f"Prewarmed env '{self.prewarm_env_name}' detected; skipping env setup.",
                    stderr="",
                )
            else:
                env_result = self._run_script(
                    sandbox,
                    script_content=test_spec.setup_env_script,
                    remote_name="setup_env.sh",
                    timeout=self.env_setup_timeout_seconds,
                    stage_name="Environment setup",
                )
            stage_logs.append(self._format_stage("Environment setup", env_result))
            LOGGER.info(f"[TIMING] {instance_id[:8]} - Environment setup completed in {time.time() - t_env_start:.2f}s")

            # Safety: Remove repo_path if it exists before running install_repo_script.
            self._run_command(
                sandbox,
                f"rm -rf {self.repo_path}",
                timeout=60,
                desc="clean testbed directory",
                allow_error=True,
            )

            t_repo_start = time.time()
            LOGGER.info(f"[TIMING] {instance_id[:8]} - Starting repository setup")
            repo_result = self._run_script(
                sandbox,
                script_content=test_spec.install_repo_script,
                remote_name="install_repo.sh",
                timeout=self.repo_setup_timeout_seconds,
                stage_name="Repository setup",
            )
            stage_logs.append(self._format_stage("Repository setup", repo_result))
            LOGGER.info(f"[TIMING] {instance_id[:8]} - Repository setup completed in {time.time() - t_repo_start:.2f}s")

            record["logs"]["setup"] = "\n\n".join(stage_logs)
            self._persist_logs(instance_id, "setup", record["logs"]["setup"])
            self._instances[instance_id] = record

        try:
            t_total_start = time.time()
            if self.worktree_mode:
                LOGGER.info(
                    f"[TIMING] {instance_id[:8]} - Starting canonical/worktree setup (worktree_mode=True)"
                )
                await asyncio.to_thread(_ensure_canonical_and_worktree)
            else:
                LOGGER.info(
                    f"[TIMING] {instance_id[:8]} - Starting complete sandbox setup (create + env + repo)"
                )
                await asyncio.to_thread(_do_setup)
            total_elapsed = time.time() - t_total_start
            LOGGER.info(f"[TIMING] {instance_id[:8]} - Total sandbox setup completed in {total_elapsed:.2f}s")

            return instance_id, ToolResponse()
        except Exception as exc:
            if sandbox is not None:
                try:
                    sandbox.kill()
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

        # Run blocking E2B API calls in thread pool to avoid blocking event loop
        # This allows multiple tool calls to execute in parallel via asyncio.gather()
        start_time = time.time()
        LOGGER.info(f"[TIMING] {instance_id[:8]} - Starting tool execution: {action}")
        
        if action == "run_shell":
            result = await asyncio.to_thread(self._execute_run_shell, instance_id, record, parameters)
        elif action == "read_file":
            result = await asyncio.to_thread(self._execute_read_file, instance_id, record, parameters)
        elif action == "write_file":
            result = await asyncio.to_thread(self._execute_write_file, instance_id, record, parameters)
        elif action == "submit_patch":
            patch = parameters.get("patch")
            notes = parameters.get("notes", "")
            if not isinstance(patch, str) or not patch.strip():
                return ToolResponse(text="No patch provided."), 0.0, {"status": "invalid_patch"}
            result = await asyncio.to_thread(self._execute_submit_patch, instance_id, record, patch, notes)
        else:
            return ToolResponse(text=f"Unknown action '{action}'. Nothing executed."), 0.0, {"status": "unknown_action"}
        
        elapsed = time.time() - start_time
        LOGGER.info(f"[TIMING] {instance_id[:8]} - Completed {action} in {elapsed:.2f}s")
        return result

    async def calc_reward(self, instance_id: str, **kwargs) -> float:  # noqa: D401
        if instance_id not in self._instances:
            raise ValueError(f"Unknown SWEbench sandbox instance: {instance_id}")
        return float(self._instances[instance_id].get("last_reward", 0.0))

    async def release(self, instance_id: str, **kwargs) -> None:  # noqa: D401
        record = self._instances.pop(instance_id, None)
        if not record:
            return
        if self.worktree_mode and self.enable_e2b:
            canonical_id = record.get("canonical_id") or self._req_to_canon.pop(instance_id, None)
            worktree_path = record.get("worktree_path") or self._req_worktrees.pop(instance_id, None)
            if canonical_id is None:
                return
            canonical = self._canonicals.get(canonical_id)
            if canonical is None:
                return
            sbox: Optional[E2BSandbox] = canonical.get("sandbox")
            with canonical["lock"]:
                if sbox is not None and worktree_path:
                    try:
                        self._run_command(
                            sbox,
                            f"git -C {self.repo_path} worktree remove -f {worktree_path}",
                            timeout=120,
                            desc="Remove worktree",
                            allow_error=True,
                        )
                        self._run_command(
                            sbox, f"git -C {self.repo_path} worktree prune", timeout=60, desc="Prune worktrees", allow_error=True
                        )
                    except Exception:  # pragma: no cover
                        LOGGER.exception("Failed to remove worktree %s", worktree_path)
                canonical["refcount"] = max(0, canonical.get("refcount", 1) - 1)
                if canonical["refcount"] == 0 and sbox is not None:
                    try:
                        sbox.kill()
                    except Exception:  # pragma: no cover
                        LOGGER.exception("Failed to close canonical sandbox for %s", canonical_id)
                    self._canonicals.pop(canonical_id, None)
            return

        # Legacy one-sandbox-per-request behavior
        sandbox: Optional[E2BSandbox] = record.get("sandbox")
        if sandbox is not None:
            try:
                sandbox.kill()
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
            # NOTE: Do NOT create repo_path directory here! The install_repo_script
            # will create it via git clone. Pre-creating it causes "directory exists" errors.

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

            if self.prewarm and self._env_exists(sandbox):
                env_result = SimpleNamespace(
                    exit_code=0,
                    stdout=f"Prewarmed env '{self.prewarm_env_name}' detected; skipping env setup.",
                    stderr="",
                )
            else:
                env_result = self._run_script(
                    sandbox,
                    script_content=test_spec.setup_env_script,
                    remote_name="setup_env.sh",
                    timeout=self.env_setup_timeout_seconds,
                    stage_name="Environment setup",
                )
            stage_logs.append(self._format_stage("Environment setup", env_result))

            # Safety: Remove repo_path if it exists before running install_repo_script.
            # The script expects to git clone into an empty/non-existent directory.
            # This handles edge cases where the directory might be created by env setup or other scripts.
            self._run_command(
                sandbox,
                f"rm -rf {self.repo_path}",
                timeout=60,
                desc="clean testbed directory",
                allow_error=True,
            )

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

        # Use per-request worktree if available
        worktree_path = record.get("worktree_path")
        cwd = worktree_path or self.repo_path
        # Rewrite absolute repo paths in the command when using worktrees
        if self.worktree_mode and worktree_path and self.repo_path and self.repo_path in command:
            try:
                command = command.replace(self.repo_path, worktree_path)
            except Exception:
                pass
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

        # Resolve relative paths against worktree/repo root; rewrite absolute /workspace/testbed to worktree.
        worktree_path = record.get("worktree_path")
        if not path.startswith("/"):
            base_dir = worktree_path or self.repo_path
            path = posixpath.join(base_dir, path)
        elif self.worktree_mode and worktree_path and path.startswith(self.repo_path.rstrip("/")):
            path = worktree_path + path[len(self.repo_path):]

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

        worktree_path = record.get("worktree_path")
        if not path.startswith("/"):
            base_dir = worktree_path or self.repo_path
            path = posixpath.join(base_dir, path)
        elif self.worktree_mode and worktree_path and path.startswith(self.repo_path.rstrip("/")):
            path = worktree_path + path[len(self.repo_path):]

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

        # Use per-request worktree directory if present
        repo_path = record.get("worktree_path", self.repo_path)
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

        patch_remote_path = posixpath.join(repo_path, "candidate.patch")
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

        # Re-target eval script to use the worktree path as repo directory
        eval_script = test_spec.eval_script.replace(self.repo_path, repo_path)
        eval_result = self._run_script(
            sandbox,
            script_content=eval_script,
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
