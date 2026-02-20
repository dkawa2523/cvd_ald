#!/usr/bin/env python3
"""Codex CLI autorun runner.

- Reads tasks from tasks/tasks.json
- Executes tasks sequentially via `codex exec`
- Records completion state in runs/autorun_state.json
- Stops on tasks with stop_after=true with exit code 42
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


EXIT_CHECKPOINT = 42


@dataclass
class CodexFeatures:
    model_flag: Optional[str] = None          # e.g. --model or -m
    reasoning_flag: Optional[str] = None      # e.g. --reasoning
    config_flag: Optional[str] = None         # e.g. --config
    prompt_file_flag: Optional[str] = None    # e.g. --prompt-file / --file
    workspace_write_flag: Optional[str] = None  # e.g. --workspace-write
    workspace_flag: Optional[str] = None        # e.g. --workspace
    read_only_flag: Optional[str] = None        # e.g. --read-only
    supports_stdin: bool = True               # optimistic default


def _run(cmd: List[str], *, cwd: Optional[str] = None, input_text: Optional[str] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd,
        input=(input_text.encode("utf-8") if input_text is not None else None),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def detect_codex_features() -> CodexFeatures:
    # We assume `codex` is on PATH.
    cp = _run(["codex", "exec", "--help"])
    help_text = (cp.stdout + cp.stderr).decode("utf-8", errors="ignore")

    f = CodexFeatures()

    # Model flag detection
    if "--model" in help_text:
        f.model_flag = "--model"
    elif re_search_usage_short_flag(help_text, "-m"):
        f.model_flag = "-m"

    # Reasoning flag detection
    if "--reasoning" in help_text:
        f.reasoning_flag = "--reasoning"

    # Config flag detection
    if "--config" in help_text:
        f.config_flag = "--config"

    # Prompt file flag detection
    if "--prompt-file" in help_text:
        f.prompt_file_flag = "--prompt-file"
    elif "--file" in help_text:
        # ambiguous; only use if it appears to refer to prompt file
        f.prompt_file_flag = "--file"
    elif re_search_usage_short_flag(help_text, "-f"):
        f.prompt_file_flag = "-f"

    # Workspace write mode detection
    if "--workspace-write" in help_text:
        f.workspace_write_flag = "--workspace-write"
    if "--workspace" in help_text:
        f.workspace_flag = "--workspace"
    if "--read-only" in help_text:
        f.read_only_flag = "--read-only"

    # Stdin support: if usage suggests a positional prompt argument, stdin may not work.
    # We keep supports_stdin=True by default, but will fall back to positional if needed.
    if "Usage:" in help_text and "codex exec <" in help_text:
        f.supports_stdin = True  # still usually okay
    return f


def re_search_usage_short_flag(help_text: str, flag: str) -> bool:
    # heuristically detect short flags in help output
    return f" {flag}," in help_text or f" {flag} " in help_text or f"\n  {flag}" in help_text


def load_tasks(tasks_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(tasks_path.read_text(encoding="utf-8"))
    return data["tasks"]


def load_state(state_path: Path) -> Dict[str, Any]:
    if not state_path.exists():
        return {"completed": {}, "created_at": time.time()}
    return json.loads(state_path.read_text(encoding="utf-8"))


def save_state(state_path: Path, state: Dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def build_prompt(task: Dict[str, Any], template_path: Path) -> str:
    # Minimal prompt assembly (string formatting).
    policy = read_text(Path("docs/adr/0001-initial-decisions.md"))
    ctx = read_text(Path("docs/CONTEXT.md"))
    arch = read_text(Path("docs/ARCHITECTURE.md"))
    evalp = read_text(Path("docs/EVAL_PROTOCOL.md"))
    req = read_text(Path("docs/REQUIREMENTS.md"))
    trace = read_text(Path("docs/TRACEABILITY.md"))

    template = template_path.read_text(encoding="utf-8")

    def fmt_list(xs: List[str]) -> str:
        return "\n".join([f"- {x}" for x in xs]) if xs else "- (none)"

    scope = task.get("scope_limits", {})
    prompt = template.format(
        task_id=task["task_id"],
        title=task["title"],
        milestone=task["milestone"],
        type=task["type"],
        description=task.get("description", "").strip(),
        acceptance=fmt_list(task.get("acceptance_criteria", [])),
        verification=fmt_list(task.get("verification_commands", [])),
        depends=fmt_list(task.get("depends_on", [])),
        max_changed_files=scope.get("max_changed_files"),
        max_diff_lines=scope.get("max_diff_lines"),
        allowed_dirs="\n".join([f"- {d}" for d in scope.get("allowed_dirs", [])]) or "- (none)",
        forbidden_actions="\n".join([f"- {a}" for a in scope.get("forbidden_actions", [])]) or "- (none)",
        policy_lock=policy,
        context=ctx,
        architecture=arch,
        eval_protocol=evalp,
        requirements=req,
        traceability=trace,
    )
    return prompt


def codex_exec(features: CodexFeatures, prompt_text: str, prompt_file: Path) -> int:
    # Build command with best-effort flag support.
    cmd: List[str] = ["codex", "exec"]

    # Prefer passing a config snippet if supported.
    cfg_path = Path("config/codex_config_snippet.toml")
    if features.config_flag and cfg_path.exists():
        cmd += [features.config_flag, str(cfg_path)]

    # Set model and reasoning if supported.
    if features.model_flag:
        cmd += [features.model_flag, "gpt-5.2-codex"]
    if features.reasoning_flag:
        cmd += [features.reasoning_flag, "high"]

    # Force write-enabled workspace if possible.
    if features.workspace_write_flag:
        cmd += [features.workspace_write_flag]
    elif features.workspace_flag:
        # We try common "write" value.
        cmd += [features.workspace_flag, "write"]
    elif features.read_only_flag:
        # Some CLIs accept --read-only=false, others treat --read-only as a boolean switch.
        # We try the value form first.
        cmd += [f"{features.read_only_flag}=false"]

    # Provide prompt
    if features.prompt_file_flag:
        cmd += [features.prompt_file_flag, str(prompt_file)]
        cp = _run(cmd, cwd=str(Path.cwd()))
    else:
        # Fallback: stdin
        cp = _run(cmd, cwd=str(Path.cwd()), input_text=prompt_text)

    sys.stdout.write(cp.stdout.decode("utf-8", errors="ignore"))
    sys.stderr.write(cp.stderr.decode("utf-8", errors="ignore"))
    return cp.returncode


def main(argv: List[str]) -> int:
    root = Path.cwd()
    tasks_path = root / "tasks" / "tasks.json"
    state_path = root / "runs" / "autorun_state.json"
    prompts_dir = root / "runs" / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(tasks_path)
    state = load_state(state_path)

    # Detect codex CLI availability
    if shutil_which("codex") is None:
        print("[autorun] ERROR: `codex` command not found in PATH.", file=sys.stderr)
        print("[autorun] Install Codex CLI before running autorun.", file=sys.stderr)
        return 2

    # Detect CLI features once and store in state for reproducibility.
    features = detect_codex_features()
    state.setdefault("codex_features", {})
    state["codex_features"] = features.__dict__
    save_state(state_path, state)

    completed: Dict[str, Any] = state.get("completed", {})

    for task in tasks:
        tid = task["task_id"]
        if tid in completed and completed[tid].get("status") == "ok":
            print(f"[autorun] SKIP {tid} (already completed)")
            continue

        print(f"[autorun] START {tid}: {task['title']}")

        template_name = {
            "implement": "implement.prompt.txt",
            "review": "review.prompt.txt",
            "decision": "decision.prompt.txt",
            "checkpoint": "checkpoint.prompt.txt",
        }.get(task.get("type", "implement"), "implement.prompt.txt")

        template_path = root / "prompts" / template_name
        prompt_text = build_prompt(task, template_path)
        prompt_file = prompts_dir / f"{tid}.prompt.txt"
        prompt_file.write_text(prompt_text, encoding="utf-8")

        rc = codex_exec(features, prompt_text, prompt_file)
        if rc != 0:
            print(f"[autorun] ERROR: task {tid} failed with codex exit code {rc}", file=sys.stderr)
            # Do not mark complete; allow resume.
            return rc

        # Mark completed
        completed[tid] = {"status": "ok", "completed_at": time.time()}
        state["completed"] = completed
        state["last_completed"] = tid
        save_state(state_path, state)

        print(f"[autorun] DONE {tid}")

        if task.get("stop_after", False):
            print(f"[autorun] STOP_AFTER triggered by {tid}. Exiting with {EXIT_CHECKPOINT}.")
            return EXIT_CHECKPOINT

    print("[autorun] All tasks completed.")
    return 0


def shutil_which(cmd: str) -> Optional[str]:
    # minimal `which` to avoid importing shutil in older minimal environments
    paths = os.environ.get("PATH", "").split(os.pathsep)
    exts = [""]
    if os.name == "nt":
        exts = os.environ.get("PATHEXT", "").split(os.pathsep) or [".exe", ".cmd", ".bat"]
    for p in paths:
        for ext in exts:
            c = Path(p) / f"{cmd}{ext}"
            if c.exists() and os.access(str(c), os.X_OK):
                return str(c)
    return None


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
