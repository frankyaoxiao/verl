#!/usr/bin/env python3
"""
Interactive E2B sandbox shell

Spin up an E2B sandbox with a specified template and run an interactive loop
that executes bash commands inside the sandbox, returning stdout/stderr.

Notes
- Reads E2B_API_KEY from environment or .env (if python-dotenv is installed).
- Each command runs in a fresh shell; cwd/env aren't persistent between calls.
  Use "cd dir && your_command" within a single line when needed.
- The default working directory is inferred from the template:
  - 'swebench-conda' -> /home/user/testbed
  - otherwise        -> /workspace/testbed
- You can override the working directory with --cwd.

Examples
  python3 scripts/e2b_shell.py --template swebench-conda
  python3 scripts/e2b_shell.py --template my-prewarmed-alias --timeout 900
  python3 scripts/e2b_shell.py --template my-prewarmed-alias --cwd /workspace/testbed

Exit the REPL with: exit, quit, :q, or CTRL+D.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional


def _maybe_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(dotenv_path=".env")
    except Exception:
        # Optional dependency; continue if missing
        pass


def _ensure_api_key() -> None:
    if os.getenv("E2B_API_KEY"):
        return
    _maybe_load_dotenv()
    if os.getenv("E2B_API_KEY"):
        return
    print(
        "E2B_API_KEY is not set. Export it or add it to .env (E2B_API_KEY=...).",
        file=sys.stderr,
    )
    sys.exit(2)


def _infer_default_cwd(template: Optional[str]) -> str:
    t = (template or "").strip()
    return "/home/user/testbed" if (t == "swebench-conda") else "/workspace/testbed"


def _format_result(prefix: str, exit_code: int, stdout: str, stderr: str) -> str:
    parts = [f"## {prefix}", f"exit_code: {exit_code}"]
    if stdout:
        parts.append("stdout:")
        parts.append(stdout.rstrip("\n"))
    if stderr:
        parts.append("stderr:")
        parts.append(stderr.rstrip("\n"))
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive E2B sandbox shell")
    p.add_argument(
        "--template",
        default="swebench-conda",
        help="Sandbox template alias (default: swebench-conda)",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-command timeout in seconds (default: 600)",
    )
    p.add_argument(
        "--cwd",
        default=None,
        help="Working directory for commands (default inferred from template)",
    )
    p.add_argument(
        "--skip-conda-tos",
        action="store_true",
        help="Skip conda ToS acceptance step (best-effort)",
    )
    p.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable fallback to 'swebench-conda' if template creation fails",
    )
    # SWEbench init options (to mirror GRPO tool environment)
    p.add_argument(
        "--init-swebench",
        action="store_true",
        help="Run SWEbench environment + repository setup before entering REPL",
    )
    p.add_argument(
        "--dataset",
        default="MariusHobbhahn/swe-bench-verified-mini",
        help="HF dataset for SWEbench instance selection (default: verified-mini)",
    )
    p.add_argument(
        "--instance-id",
        default=None,
        help="Specific SWEbench instance_id to initialize (takes precedence over --index)",
    )
    p.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index into the dataset split when --instance-id is not provided",
    )
    p.add_argument(
        "--split",
        default="test",
        help="Dataset split to use (default: test)",
    )
    p.add_argument(
        "--worktree-mode",
        action="store_true",
        default=True,
        help="Create and use a Git worktree for the instance (default: enabled)",
    )
    p.add_argument(
        "--skip-env-setup",
        action="store_true",
        help="Skip running TestSpec setup_env_script (simulates prewarmed env)",
    )
    return p.parse_args()


def main() -> None:
    _ensure_api_key()

    try:
        from e2b_code_interpreter import Sandbox as E2BSandbox  # type: ignore
        from e2b.sandbox_sync.commands.command_handle import (
            CommandExitException,  # type: ignore
        )
    except Exception as exc:  # pragma: no cover - import/runtime environment
        print(
            "Failed to import E2B SDK. Install with: pip install e2b-code-interpreter",
            file=sys.stderr,
        )
        raise

    args = parse_args()
    template = (args.template or "").strip() or None
    timeout = int(args.timeout)
    base_cwd = args.cwd or _infer_default_cwd(template)

    # Create sandbox
    print(f"Connecting to E2B sandbox (template={template or '(default)'}; timeout={timeout}s)...")

    def _create_sandbox(tmpl: Optional[str]):
        kwargs = {"timeout": timeout}
        if tmpl:
            kwargs["template"] = tmpl
        return E2BSandbox.create(**kwargs)

    sandbox = None
    try:
        try:
            sandbox = _create_sandbox(template)
        except Exception:
            if not args.no_fallback and template and template != "swebench-conda":
                print(
                    f"Template '{template}' failed; falling back to 'swebench-conda'...",
                    file=sys.stderr,
                )
                sandbox = _create_sandbox("swebench-conda")
                base_cwd = args.cwd or _infer_default_cwd("swebench-conda")
            else:
                raise

        print("Sandbox created.")

        # Prepare minimal workspace dirs similar to our tool behavior
        try:
            workspace_root = os.path.dirname(base_cwd) or "/workspace"
            worktrees_root = os.path.join(workspace_root, "worktrees")
            # Ensure parent, repo path, and sibling worktrees root exist
            mk_cmd = (
                f"mkdir -p {json.dumps(workspace_root)} && "
                f"mkdir -p {json.dumps(base_cwd)} && "
                f"mkdir -p {json.dumps(worktrees_root)}"
            )
            sandbox.commands.run(
                mk_cmd,
                timeout=60,
                request_timeout=60,
            )
        except Exception:
            pass  # best effort

        if not args.skip_conda_tos:
            tos_cmd = (
                "bash -lc '"
                "source /opt/miniconda3/etc/profile.d/conda.sh && "
                "conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && "
                "conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r'"
            )
            try:
                sandbox.commands.run(tos_cmd, timeout=120, request_timeout=120)
            except CommandExitException as exc:  # best-effort
                formatted = _format_result("Conda ToS", exc.exit_code, exc.stdout or "", exc.stderr or "")
                print(formatted)
            except Exception:
                pass

        # Optionally perform SWEbench initialization: env + repo + worktree
        if args.init_swebench:
            try:
                from datasets import load_dataset  # type: ignore
                from swebench.harness.test_spec.test_spec import make_test_spec  # type: ignore
            except Exception as exc:
                print(
                    "Missing dependencies for SWEbench init. Install: pip install datasets swebench",
                    file=sys.stderr,
                )
                raise

            # Select instance
            ds = load_dataset(args.dataset, split=args.split)
            example = None
            if args.instance_id:
                for ex in ds:
                    if ex.get("instance_id") == args.instance_id:
                        example = ex
                        break
                if example is None:
                    raise RuntimeError(f"instance_id not found in dataset: {args.instance_id}")
            else:
                if args.index < 0 or args.index >= len(ds):
                    raise RuntimeError(f"index out of range: {args.index} (size={len(ds)})")
                example = ds[int(args.index)]

            spec = make_test_spec(example)
            # If repo_path deviates from /testbed, rewrite scripts accordingly
            repo_path = base_cwd
            def _rewrite_list(cmds: list[str]) -> list[str]:
                return [cmd.replace("/testbed", repo_path) for cmd in cmds]
            try:
                spec.repo_script_list = _rewrite_list(spec.repo_script_list)
                spec.env_script_list = _rewrite_list(spec.env_script_list)
                spec.eval_script_list = _rewrite_list(spec.eval_script_list)
            except Exception:
                # Best-effort; some versions expose only merged scripts below
                pass

            # Accept ToS already attempted above; run env setup unless skipped
            def _write_and_run(remote_name: str, content: str, stage: str, timeout_s: int):
                # place scripts in workspace root sibling
                workspace_root = os.path.dirname(repo_path) or "/workspace"
                remote_path = f"{workspace_root}/{remote_name}"
                sandbox.files.write(remote_path, content)
                return sandbox.commands.run(
                    f"bash {json.dumps(remote_path)}",
                    timeout=timeout_s,
                    request_timeout=timeout_s,
                )

            if not args.skip_env_setup:
                env_script = getattr(spec, "setup_env_script", None)
                if env_script:
                    _ = _write_and_run("setup_env.sh", env_script, "Environment setup", timeout)

            # Clean and install repo
            try:
                sandbox.commands.run(
                    f"rm -rf {json.dumps(repo_path)}",
                    timeout=60,
                    request_timeout=60,
                )
            except Exception:
                pass
            install_script = getattr(spec, "install_repo_script", None)
            if install_script:
                _ = _write_and_run("install_repo.sh", install_script, "Repository setup", timeout)

            # Worktree handling (mirrors tool default behavior)
            if args.worktree_mode:
                workspace_root = os.path.dirname(repo_path) or "/workspace"
                worktrees_root = os.path.join(workspace_root, "worktrees")
                worktree_path = os.path.join(worktrees_root, str(example.get("instance_id", "repl")))
                # Ensure worktrees root
                sandbox.commands.run(
                    f"mkdir -p {json.dumps(worktrees_root)}",
                    timeout=60,
                    request_timeout=60,
                )
                # Create worktree (force, then mark safe)
                base_commit = example.get("base_commit") or getattr(spec, "base_commit", None) or "HEAD"
                sandbox.commands.run(
                    f"git -C {json.dumps(repo_path)} worktree add -f {json.dumps(worktree_path)} {json.dumps(base_commit)}",
                    timeout=timeout,
                    request_timeout=timeout,
                )
                sandbox.commands.run(
                    f"git config --global --add safe.directory {json.dumps(worktree_path)}",
                    timeout=60,
                    request_timeout=60,
                )
                base_cwd = worktree_path

        # Initial diagnostics
        try:
            res = sandbox.commands.run("bash -lc 'pwd'", cwd=base_cwd, timeout=30, request_timeout=30)
            print(_format_result("Initial pwd", res.exit_code, res.stdout or "", res.stderr or ""))
        except Exception:
            pass

        # Instructions
        print()
        print("Interactive shell ready.")
        print("- Each command runs in a fresh shell; use 'cd dir && <cmd>' where needed.")
        print("- Base working directory:", base_cwd)
        print("- Exit with: exit | quit | :q | Ctrl+D")
        print()

        # REPL
        while True:
            try:
                line = input("e2b$ ").strip()
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print()
                continue
            if not line:
                continue
            if line.lower() in {"exit", "quit", ":q"}:
                break

            cmd = f"bash -lc {json.dumps(line)}"
            try:
                result = sandbox.commands.run(
                    cmd,
                    cwd=base_cwd,
                    timeout=timeout,
                    request_timeout=timeout,
                )
                print(
                    _format_result(
                        "Command",
                        getattr(result, "exit_code", 0),
                        getattr(result, "stdout", "") or "",
                        getattr(result, "stderr", "") or "",
                    )
                )
            except CommandExitException as exc:
                # Non-zero exit; still show outputs
                print(_format_result("Command", exc.exit_code, exc.stdout or "", exc.stderr or ""))
            except KeyboardInterrupt:
                print()
                continue
            except Exception as exc:
                print(f"Error: {exc}")

    finally:
        if sandbox is not None:
            try:
                sandbox.kill()
            except Exception:
                pass


if __name__ == "__main__":
    main()
