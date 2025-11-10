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

    _STATE_REGISTRY: dict[str, dict[str, Any]] = {}

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
        self.timeout_seconds = config.get("timeout_seconds", 900)
        self.install_timeout_seconds = config.get("install_timeout_seconds", 300)
        self.enable_e2b = config.get("enable_e2b", True)
        if self.enable_e2b:
            load_e2b_api_key()
        self.template = config.get("template")
        self.workspace = config.get("workdir", "/workspace")
        self.repo_path = config.get("repo_path", "/testbed")
        # Auto template selection
        self.auto_template: bool = bool(config.get("auto_template", False))
        self.alias_prefix: str = config.get("alias_prefix", "swebench")
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
        
        # Timing directory for tracking operation durations
        timing_dir = config.get("timing_dir", os.getenv("VERL_TIMING_DIR", "tmp/timing"))
        self.timing_dir: Optional[Path] = Path(os.path.expanduser(timing_dir)).expanduser() if timing_dir else None
        if self.timing_dir is not None:
            self.timing_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name = config.get("model_name", "verl_agent")
        # Output truncation for tool responses (extensible caps)
        self.max_output_tokens: int = int(config.get("max_output_tokens", 10_000))
        self.chars_per_token: int = int(config.get("chars_per_token", 4))
        self.truncate_side: str = str(config.get("truncate_side", "middle"))
        self.mode: str = str(config.get("mode", "legacy")).lower()

        # Simplified state management - just track active sandbox instances
        state_key = self._config_state_key(config)
        state = self._STATE_REGISTRY.setdefault(
            state_key,
            {
                "instances": {},
                "create_lock": threading.Lock(),
                "pending_events": {},
            },
        )
        self._state_key = state_key
        self._instances = state["instances"]
        self._create_lock = state["create_lock"]
        self._pending_events: dict[str, asyncio.Event] = state["pending_events"]

    @staticmethod
    def _short_env_hash(env_image_key: str) -> str:
        try:
            core = env_image_key.split(":", 1)[0]
            parts = core.split(".")
            return parts[-1][:10]
        except Exception:
            return "unknownhash"

    @staticmethod
    def _safe_repo_alias(repo: str) -> str:
        return repo.replace("/", "_")

    @staticmethod
    def _config_state_key(config: dict) -> str:
        filtered = {k: v for k, v in config.items() if k != "mode"}

        def _normalize(val: Any) -> Any:
            if isinstance(val, Path):
                return str(val)
            return val

        normalized = {k: _normalize(v) for k, v in filtered.items()}
        try:
            return json.dumps(normalized, sort_keys=True, default=str)
        except TypeError:
            # Fallback: coerce any non-serialisable values to string
            coerced = {k: (_normalize(v) if isinstance(v, (str, int, float, bool, type(None))) else str(v)) for k, v in normalized.items()}
            return json.dumps(coerced, sort_keys=True, default=str)

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
        """Create a new sandbox instance from a pre-built template.
        
        The new simplified architecture:
        1. Determine which template to use based on (repo, env_hash)
        2. Spawn a fresh E2B sandbox from that template
        3. Git checkout to the specific base_commit
        4. Done! No worktrees, no sharing, just clean simple sandboxes
        """
        # Extract instance_id from various sources
        embedded_instance_id = kwargs.pop("instance_id", None)
        instance_id = request_id or embedded_instance_id
        if instance_id is None:
            instance_id = str(uuid4())

        create_kwargs = kwargs.get("create_kwargs")
        if create_kwargs is None:
            create_kwargs = kwargs
        dataset_instance = create_kwargs.get("dataset_instance")
        if dataset_instance is None:
            raise ValueError("dataset_instance is required in tools_kwargs for SWEbench execution")

        # Check if already exists (shouldn't happen, but handle it)
        while True:
            with self._create_lock:
                existing = self._instances.get(instance_id)
                if existing is not None:
                    LOGGER.warning(f"Instance {instance_id[:8]} already exists, reusing it")
                    return instance_id, ToolResponse()
                wait_event = self._pending_events.get(instance_id)
                if wait_event is None:
                    wait_event = asyncio.Event()
                    self._pending_events[instance_id] = wait_event
                    break
            await wait_event.wait()

        try:
            # Create test spec
            test_spec: TestSpec = make_test_spec(dataset_instance)
            
            record: dict[str, Any] = {
                "dataset_instance": dataset_instance,
                "test_spec": test_spec,
                "last_reward": 0.0,
                "logs": {},
            }
    
            if not self.enable_e2b:
                self._instances[instance_id] = record
                return instance_id, ToolResponse()
    
            # Determine which template to use
            sandbox_kwargs: dict[str, Any] = {"timeout": self.timeout_seconds}
            
            # Template selection: auto-generate from (repo, env_hash)
            template_alias = None
            if self.auto_template:
                try:
                    env_key = test_spec.env_image_key
                    repo = dataset_instance.get("repo")
                    if env_key and repo:
                        env_hash = self._short_env_hash(env_key)
                        template_alias = f"{self.alias_prefix}-{self._safe_repo_alias(repo)}-{env_hash}"
                        sandbox_kwargs["template"] = template_alias
                        # When using pre-built templates, the repository is at /workspace/testbed
                        self.workspace = "/workspace"
                        self.repo_path = "/workspace/testbed"
                        LOGGER.info(f"[template-selection] Using auto-generated template: {template_alias}")
                except Exception as exc:
                    LOGGER.warning(f"[template-selection] Failed to auto-generate template name: {exc}")
                    template_alias = None
            
            # Fall back to configured template if auto-selection failed
            if not template_alias and self.template:
                template_alias = self.template
                sandbox_kwargs["template"] = template_alias
                LOGGER.info(f"[template-selection] Using configured template: {template_alias}")
    
            LOGGER.info(
                f"[TIMING] {instance_id[:8]} - Starting sandbox creation (template={template_alias or 'default'})"
            )
            t_start = time.time()
            
            # Run sandbox creation in thread pool to avoid blocking event loop
            def _create_and_checkout():
                # Create E2B sandbox from template
                sandbox = E2BSandbox.create(**sandbox_kwargs)
                
                LOGGER.info(
                    f"[SANDBOX] Created sandbox from template={template_alias or 'default'}, "
                    f"sandbox_id={sandbox.sandbox_id}"
                )
                
                # Git checkout to the specific base commit
                # The template already has the repo cloned at /workspace/testbed
                base_commit = dataset_instance["base_commit"]
                checkout_cmd = f"cd {self.repo_path} && git checkout {base_commit}"
                
                checkout_result = self._run_command(
                    sandbox,
                    checkout_cmd,
                    timeout=60,
                    desc=f"Checkout {base_commit[:8]}",
                    allow_error=False,
                )
                
                LOGGER.info(
                    f"[GIT] Checked out {base_commit[:8]} (exit_code={checkout_result.exit_code})"
                )
                
                return sandbox
            
            # Execute in thread pool
            sandbox = await asyncio.to_thread(_create_and_checkout)
            
            total_elapsed = time.time() - t_start
            LOGGER.info(f"[TIMING] {instance_id[:8]} - Sandbox ready in {total_elapsed:.2f}s")
            
            # Store in registry
            record["sandbox"] = sandbox
            self._instances[instance_id] = record
            
            # Log timing data
            self._log_timing(
                request_id=instance_id,
                operation="sandbox_setup",
                duration=total_elapsed,
                instance_id=dataset_instance.get("instance_id", "unknown"),
                worktree_mode=False  # No more worktrees!
            )
    
            return instance_id, ToolResponse()
            
        except Exception as exc:
            LOGGER.error(f"[ERROR] Failed to create sandbox: {exc}")
            raise exc
    
        finally:
            with self._create_lock:
                event = self._pending_events.pop(instance_id, None)
            if event is not None:
                event.set()

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
        parameters = parameters or {}
        mode = self.mode

        # Run blocking E2B API calls in thread pool to avoid blocking event loop
        # This allows multiple tool calls to execute in parallel via asyncio.gather()
        start_time = time.time()
        if mode == "bash":
            LOGGER.info(f"[TIMING] {instance_id[:8]} - Starting tool execution: bash")
            result = await asyncio.to_thread(self._execute_run_shell, instance_id, record, parameters)
        elif mode == "submit_solution":
            LOGGER.info(f"[TIMING] {instance_id[:8]} - Starting tool execution: submit_solution")
            # Ignore any stray parameters but surface a warning for transparency.
            extra_params = {k: v for k, v in parameters.items() if v not in (None, "", [], {})}
            if extra_params:
                LOGGER.debug(
                    "[submit_solution] Ignoring unexpected parameters for %s: %s",
                    instance_id,
                    list(extra_params.keys()),
                )
            result = await asyncio.to_thread(self._execute_submit_solution, instance_id, record)
        else:
            action = parameters.get("action")
            if not action:
                action = "submit_patch"
            action = action.lower()
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
        label = "submit_solution" if mode == "submit_solution" else ("bash" if mode == "bash" else action)
        LOGGER.info(f"[TIMING] {instance_id[:8]} - Completed {label} in {elapsed:.2f}s")
        
        # Log timing data to file for analysis
        self._log_timing(
            request_id=instance_id,
            operation=label,
            duration=elapsed,
            instance_id=record.get("dataset_instance", {}).get("instance_id", "unknown")
        )
        
        return result

    async def calc_reward(self, instance_id: str, **kwargs) -> float:  # noqa: D401
        if instance_id not in self._instances:
            raise ValueError(f"Unknown SWEbench sandbox instance: {instance_id}")
        return float(self._instances[instance_id].get("last_reward", 0.0))

    async def release(self, instance_id: str, **kwargs) -> None:  # noqa: D401
        """Release a sandbox instance by killing it.
        
        Simplified: just kill the sandbox. No worktrees, no sharing, no reference counting.
        """
        record = self._instances.pop(instance_id, None)
        if not record:
            return
        
        sandbox: Optional[E2BSandbox] = record.get("sandbox")
        if sandbox is not None:
            try:
                LOGGER.info(f"[RELEASE] Killing sandbox for {instance_id[:8]}")
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

    def _truncate_text_for_model(self, text: str) -> str:
        try:
            max_chars = self.max_output_tokens * max(1, self.chars_per_token)
        except Exception:
            max_chars = 40_000
        if len(text) <= max_chars:
            return text
        side = (self.truncate_side or "middle").lower()
        notice = f"\n\n[Output truncated to ~{self.max_output_tokens} tokens]"
        if side == "head":
            return text[:max_chars] + notice
        if side == "tail":
            return text[-max_chars:] + notice
        # middle by default
        keep = max_chars // 2
        return text[:keep] + "\n...\n" + text[-keep:] + notice

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
    
    def _log_timing(self, request_id: str, operation: str, duration: float, **metadata) -> None:
        """Log timing data to a JSON file for post-analysis."""
        if self.timing_dir is None:
            return
        
        timing_entry = {
            "timestamp": time.time(),
            "request_id": request_id,
            "operation": operation,
            "duration_seconds": duration,
            "mode": self.mode,
            **metadata
        }
        
        # Append to a daily timing log file
        date_str = time.strftime("%Y%m%d")
        timing_file = self.timing_dir / f"timing_{date_str}.jsonl"
        
        with open(timing_file, 'a') as f:
            f.write(json.dumps(timing_entry) + '\n')

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
            # Try fast repo setup using local mirror if available; fall back to harness script otherwise.
            owner_repo = dataset_instance.get("repo")
            mirror_path = posixpath.join("/opt/mirror", owner_repo + ".git") if owner_repo else None
            fast_repo_ok = False
            if mirror_path:
                # Check mirror existence
                check = self._run_command(
                    sandbox,
                    f"bash -lc 'test -d {mirror_path}'",
                    timeout=30,
                    desc="check mirror",
                    allow_error=True,
                )
                if check.exit_code == 0:
                    # Clean any previous dir, then clone from mirror and reset to base_commit
                    self._run_command(
                        sandbox,
                        f"rm -rf {self.repo_path}",
                        timeout=60,
                        desc="clean testbed directory",
                        allow_error=True,
                    )
                    cmds = [
                        f"git clone --shared {mirror_path} {self.repo_path}",
                        f"git -C {self.repo_path} reset --hard {dataset_instance['base_commit']}",
                        f"git -C {self.repo_path} remote remove origin || true",
                        f"git -C {self.repo_path} config --global --add safe.directory {self.repo_path}",
                    ]
                    for c in cmds:
                        res = self._run_command(
                            sandbox,
                            c,
                            timeout=self.repo_setup_timeout_seconds,
                            desc=f"Fast repo setup ({c})",
                            cwd=self.workspace,
                            allow_error=False,
                        )
                        stage_logs.append(self._format_stage("Repository setup (fast)", res))
                    fast_repo_ok = True
            if not fast_repo_ok:
                # Safety: Remove repo_path if it exists before running install_repo_script.
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
        # Truncate for model consumption while keeping full logs persisted
        stage_text_trunc = self._truncate_text_for_model(stage_text)
        self._persist_logs(instance_id, "shell", stage_text)
        status = "completed" if result.exit_code == 0 else "failed"
        metrics = {"status": status, "exit_code": result.exit_code}
        return ToolResponse(text=stage_text_trunc), 0.0, metrics

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
        response_text_trunc = self._truncate_text_for_model(response_text)
        self._persist_logs(instance_id, "read_file", response_text)
        return ToolResponse(text=response_text_trunc), 0.0, {"status": "completed"}

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
            resp_text = f"{log_text}\n\nPatch did not apply cleanly."
            return (ToolResponse(text=self._truncate_text_for_model(resp_text)), 0.0, {"status": "patch_invalid"})

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
        response_text_trunc = self._truncate_text_for_model(response_text)
        metrics = {"status": status, "exit_code": eval_result.exit_code}
        return ToolResponse(text=response_text_trunc), reward, metrics

    def _execute_submit_solution(
        self,
        instance_id: str,
        record: dict[str, Any],
    ) -> tuple[ToolResponse, float, dict]:
        if not self.enable_e2b:
            return ToolResponse(text="E2B execution is disabled."), 0.0, {"status": "disabled"}

        sandbox: Optional[E2BSandbox] = record.get("sandbox")
        if sandbox is None:
            return ToolResponse(text="Sandbox is not initialised."), 0.0, {"status": "no_sandbox"}

        test_spec: TestSpec = record["test_spec"]
        dataset_instance = record["dataset_instance"]
        worktree_path = record.get("worktree_path") or self.repo_path

        stage_logs: list[str] = []

        git_status_cmd = (
            f"bash -lc {json.dumps(f'cd {worktree_path} && git status --short --branch')}"
            if worktree_path
            else "bash -lc 'git status --short --branch'"
        )
        git_status = self._run_command(
            sandbox,
            git_status_cmd,
            timeout=120,
            desc="git status",
            cwd=worktree_path,
            allow_error=True,
        )
        stage_logs.append(self._format_stage("Git status", git_status))

        eval_script = test_spec.eval_script
        if worktree_path and worktree_path != self.repo_path:
            try:
                eval_script = eval_script.replace(self.repo_path, worktree_path)
            except Exception:  # pragma: no cover - best effort path rewrite
                LOGGER.debug(
                    "[submit_solution] Failed to rewrite repo_path '%s' to worktree '%s'",
                    self.repo_path,
                    worktree_path,
                )

        eval_result = self._run_script(
            sandbox,
            script_content=eval_script,
            remote_name="run_eval.sh",
            timeout=self.eval_timeout_seconds,
            stage_name="Evaluation",
            allow_error=True,
        )
        stage_logs.append(self._format_stage("Evaluation", eval_result))

        resolved = eval_result.exit_code == 0
        reward = 1.0 if resolved else 0.0
        status = "passed" if resolved else "failed"
        record["last_reward"] = reward

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
        record["logs"]["submission"] = log_text

        metrics = {
            "status": status,
            "exit_code": eval_result.exit_code,
            "instance_id": dataset_instance.get("instance_id"),
        }
        response_text = self._truncate_text_for_model(log_text)
        return ToolResponse(text=response_text), reward, metrics


__all__ = ["SWEbenchSandboxTool"]
