#!/usr/bin/env python3
"""Codex CLI autorun runner.

- Reads tasks from tasks/tasks.json
- Executes tasks sequentially via `codex exec`
- Records completion state in runs/autorun_state.json
- Enforces single-runner lock via runs/autorun.lock
- Stops on tasks with stop_after=true with exit code 42
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_DEPENDENCY = 3
EXIT_SCOPE = 4
EXIT_LOCK_TIMEOUT = 5
EXIT_CHECKPOINT = 42

INTERNAL_IGNORE_PREFIXES = (
    ".git/",
    ".pytest_cache/",
    "results/",
    "runs/prompts/",
)
INTERNAL_IGNORE_SEGMENTS = (
    "__pycache__",
)
INTERNAL_IGNORE_SUFFIXES = (
    ".pyc",
    ".pyo",
)
INTERNAL_IGNORE_FILES = {
    "runs/autorun.lock",
    "runs/autorun_state.json",
    "runs/autorun_commands.log",
    ".DS_Store",
}


@dataclass
class CodexFeatures:
    model_flag: Optional[str] = None          # e.g. --model or -m
    reasoning_flag: Optional[str] = None      # e.g. --reasoning
    config_flag: Optional[str] = None         # e.g. --config (key=value form)
    prompt_file_flag: Optional[str] = None    # e.g. --prompt-file / --file
    sandbox_flag: Optional[str] = None        # e.g. --sandbox
    skip_git_repo_check_flag: Optional[str] = None  # e.g. --skip-git-repo-check
    workspace_write_flag: Optional[str] = None  # e.g. --workspace-write (older CLIs)
    workspace_flag: Optional[str] = None        # e.g. --workspace (older CLIs)
    read_only_flag: Optional[str] = None        # e.g. --read-only (older CLIs)
    supports_stdin: bool = True


@dataclass
class FileSnapshot:
    digest: str
    text: Optional[str]
    size: int


def _run(
    cmd: List[str],
    *,
    cwd: Optional[str] = None,
    input_text: Optional[str] = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd,
        input=(input_text.encode("utf-8") if input_text is not None else None),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def re_search_usage_short_flag(help_text: str, flag: str) -> bool:
    # Heuristic: detect short flags in help output.
    return f" {flag}," in help_text or f" {flag} " in help_text or f"\n  {flag}" in help_text


def detect_codex_features() -> CodexFeatures:
    cp = _run(["codex", "exec", "--help"])
    help_text = (cp.stdout + cp.stderr).decode("utf-8", errors="ignore")

    f = CodexFeatures()

    if "--model" in help_text:
        f.model_flag = "--model"
    elif re_search_usage_short_flag(help_text, "-m"):
        f.model_flag = "-m"

    if "--reasoning" in help_text:
        f.reasoning_flag = "--reasoning"

    if "--config" in help_text:
        f.config_flag = "--config"

    if "--prompt-file" in help_text:
        f.prompt_file_flag = "--prompt-file"
    elif "--file" in help_text:
        f.prompt_file_flag = "--file"
    elif re_search_usage_short_flag(help_text, "-f"):
        f.prompt_file_flag = "-f"

    if "--sandbox" in help_text:
        f.sandbox_flag = "--sandbox"

    if "--skip-git-repo-check" in help_text:
        f.skip_git_repo_check_flag = "--skip-git-repo-check"

    # Backward compatibility with older CLIs.
    if "--workspace-write" in help_text:
        f.workspace_write_flag = "--workspace-write"
    if "--workspace" in help_text:
        f.workspace_flag = "--workspace"
    if "--read-only" in help_text:
        f.read_only_flag = "--read-only"

    if "Usage:" in help_text and "codex exec <" in help_text:
        f.supports_stdin = True
    return f


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
    return template.format(
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


def is_git_repo(root: Path) -> bool:
    cp = _run(["git", "rev-parse", "--is-inside-work-tree"], cwd=str(root))
    return cp.returncode == 0 and cp.stdout.decode("utf-8", errors="ignore").strip() == "true"


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Codex autorun task executor")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts/check contracts/print commands but do not execute codex tasks.",
    )
    parser.add_argument(
        "--git-check",
        choices=("auto", "strict", "skip"),
        default="auto",
        help="Git repository handling policy for codex exec.",
    )
    parser.add_argument(
        "--lock-timeout-sec",
        type=int,
        default=20,
        help="Max seconds to wait for runs/autorun.lock before failing.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.3-codex",
        help="Model name for codex exec.",
    )
    parser.add_argument(
        "--reasoning",
        default="high",
        help="Reasoning mode when supported by the local Codex CLI.",
    )
    parser.add_argument(
        "--validate-task-contracts",
        action="store_true",
        help="Validate tasks/tasks.json contract rules and exit.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help="Run at most N incomplete tasks in this invocation (0 means no limit).",
    )
    return parser.parse_args(argv)


def should_ignore_path(rel_path: str) -> bool:
    rel = rel_path.replace("\\", "/")
    if rel in INTERNAL_IGNORE_FILES:
        return True
    if any(seg in INTERNAL_IGNORE_SEGMENTS for seg in rel.split("/")):
        return True
    if any(rel.endswith(suffix) for suffix in INTERNAL_IGNORE_SUFFIXES):
        return True
    for prefix in INTERNAL_IGNORE_PREFIXES:
        if rel.startswith(prefix):
            return True
    return False


def snapshot_workspace(root: Path) -> Dict[str, FileSnapshot]:
    snapshot: Dict[str, FileSnapshot] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if should_ignore_path(rel):
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        text: Optional[str] = None
        if len(data) <= 1_000_000:
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = None
        snapshot[rel] = FileSnapshot(digest=digest, text=text, size=len(data))
    return snapshot


def path_allowed(rel_path: str, allowed_specs: List[str]) -> bool:
    rel = rel_path.replace("\\", "/").lstrip("./")
    for spec in allowed_specs:
        s = spec.replace("\\", "/").lstrip("./")
        if s.endswith("/"):
            if rel.startswith(s):
                return True
        elif rel == s or rel.startswith(f"{s}/"):
            return True
    return False


def file_diff_line_count(old: Optional[FileSnapshot], new: Optional[FileSnapshot]) -> int:
    if old is None and new is None:
        return 0
    if old is None and new is not None:
        if new.text is not None:
            return len(new.text.splitlines())
        return max(1, new.size // 80)
    if old is not None and new is None:
        if old.text is not None:
            return len(old.text.splitlines())
        return max(1, old.size // 80)
    assert old is not None and new is not None
    if old.text is None or new.text is None:
        if old.digest == new.digest:
            return 0
        return max(1, abs(new.size - old.size) // 80 + 1)
    diff = difflib.unified_diff(old.text.splitlines(), new.text.splitlines(), lineterm="")
    count = 0
    for line in diff:
        if line.startswith(("+++", "---", "@@")):
            continue
        if line.startswith("+") or line.startswith("-"):
            count += 1
    return count


def validate_scope_limits(
    task: Dict[str, Any],
    before: Dict[str, FileSnapshot],
    after: Dict[str, FileSnapshot],
) -> List[str]:
    errors: List[str] = []
    scope = task.get("scope_limits", {})
    allowed_dirs = scope.get("allowed_dirs", []) or []
    max_files = scope.get("max_changed_files")
    max_diff_lines = scope.get("max_diff_lines")

    changed_paths: List[str] = []
    total_diff_lines = 0
    all_paths = set(before.keys()) | set(after.keys())
    for rel in sorted(all_paths):
        old = before.get(rel)
        new = after.get(rel)
        old_digest = old.digest if old is not None else None
        new_digest = new.digest if new is not None else None
        if old_digest == new_digest:
            continue
        changed_paths.append(rel)
        total_diff_lines += file_diff_line_count(old, new)

    if isinstance(max_files, int) and len(changed_paths) > max_files:
        errors.append(f"scope violation: changed_files={len(changed_paths)} > max_changed_files={max_files}")
    if isinstance(max_diff_lines, int) and total_diff_lines > max_diff_lines:
        errors.append(f"scope violation: diff_lines={total_diff_lines} > max_diff_lines={max_diff_lines}")

    disallowed = [p for p in changed_paths if not path_allowed(p, allowed_dirs)]
    if disallowed:
        errors.append("scope violation: disallowed changed paths detected")
        for p in disallowed:
            errors.append(f"  - {p}")
    return errors


def validate_task_contracts(tasks: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []

    ids: List[str] = [t.get("task_id", "") for t in tasks]
    if len(ids) != len(set(ids)):
        errors.append("duplicate task_id detected in tasks/tasks.json")
    id_set = set(ids)

    stop_after_tasks = [t["task_id"] for t in tasks if t.get("stop_after")]
    if len(stop_after_tasks) != 1:
        errors.append(f"exactly one stop_after=true task is required, found: {len(stop_after_tasks)}")
    if stop_after_tasks and stop_after_tasks[0] != "P0-999":
        errors.append(f"stop_after task must be P0-999, found: {stop_after_tasks[0]}")

    for task in tasks:
        tid = task["task_id"]
        deps = task.get("depends_on", [])
        for dep in deps:
            if dep not in id_set:
                errors.append(f"{tid}: missing dependency task_id '{dep}'")

        vcs = task.get("verification_commands", [])
        if not vcs:
            errors.append(f"{tid}: verification_commands is empty")
            continue
        for cmd in vcs:
            if "scripts/commands.sh" not in cmd:
                errors.append(f"{tid}: verification command must reference scripts/commands.sh: {cmd}")
    return errors


def ensure_task_dependencies(task: Dict[str, Any], completed: Dict[str, Any]) -> List[str]:
    tid = task["task_id"]
    missing: List[str] = []
    for dep in task.get("depends_on", []):
        if dep not in completed or completed.get(dep, {}).get("status") != "ok":
            missing.append(dep)
    if missing:
        return [f"task {tid} has incomplete dependencies: {', '.join(missing)}"]
    return []


class LockHandle:
    def __init__(self, path: Path, payload: Dict[str, Any]) -> None:
        self.path = path
        self.payload = payload
        self.acquired = False

    def acquire(self, timeout_sec: int) -> bool:
        start = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)

        while True:
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(json.dumps(self.payload, indent=2, sort_keys=True))
                self.acquired = True
                return True
            except FileExistsError:
                if self._clear_stale_lock():
                    continue
                if time.time() - start >= timeout_sec:
                    return False
                time.sleep(0.25)

    def _clear_stale_lock(self) -> bool:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            # If lock file is corrupt, treat as stale.
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            return True

        pid = payload.get("pid")
        if isinstance(pid, int) and pid > 0 and _pid_exists(pid):
            return False

        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        return True

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        self.acquired = False


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def append_command_log(log_path: Path, record: Dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True))
        fh.write("\n")


def build_exec_command(
    *,
    features: CodexFeatures,
    args: argparse.Namespace,
    root: Path,
    prompt_file: Path,
    git_repo: bool,
) -> List[str]:
    cmd: List[str] = ["codex", "exec"]

    if args.git_check == "skip":
        if features.skip_git_repo_check_flag:
            cmd.append(features.skip_git_repo_check_flag)
    elif args.git_check == "strict":
        if not git_repo:
            raise RuntimeError("git-check strict: current directory is not a git repository")
    else:  # auto
        if not git_repo and features.skip_git_repo_check_flag:
            cmd.append(features.skip_git_repo_check_flag)

    # POLICY_LOCK: force write-enabled workspace.
    if features.sandbox_flag:
        cmd += [features.sandbox_flag, "workspace-write"]
    elif features.workspace_write_flag:
        cmd += [features.workspace_write_flag]
    elif features.workspace_flag:
        cmd += [features.workspace_flag, "write"]
    elif features.read_only_flag:
        cmd += [f"{features.read_only_flag}=false"]
    else:
        raise RuntimeError("cannot force write-enabled mode: no compatible codex exec flag detected")

    # CLI 0.98.0 expects --config key=value (not config file path).
    if features.config_flag and args.model:
        cmd += [features.config_flag, f'model="{args.model}"']

    if features.model_flag and args.model:
        cmd += [features.model_flag, args.model]

    if features.reasoning_flag and args.reasoning:
        cmd += [features.reasoning_flag, args.reasoning]

    if features.prompt_file_flag:
        cmd += [features.prompt_file_flag, str(prompt_file)]
    return cmd


def codex_exec(
    *,
    features: CodexFeatures,
    args: argparse.Namespace,
    root: Path,
    prompt_text: str,
    prompt_file: Path,
    git_repo: bool,
) -> int:
    cmd = build_exec_command(
        features=features,
        args=args,
        root=root,
        prompt_file=prompt_file,
        git_repo=git_repo,
    )
    append_command_log(
        root / "runs" / "autorun_commands.log",
        {"ts": time.time(), "command": cmd, "prompt_file": str(prompt_file)},
    )
    if args.dry_run:
        print(f"[autorun] DRY-RUN: {' '.join(cmd)}")
        return 0

    if features.prompt_file_flag:
        cp = _run(cmd, cwd=str(root))
    else:
        cp = _run(cmd, cwd=str(root), input_text=prompt_text)
    sys.stdout.write(cp.stdout.decode("utf-8", errors="ignore"))
    sys.stderr.write(cp.stderr.decode("utf-8", errors="ignore"))
    return cp.returncode


def shutil_which(cmd: str) -> Optional[str]:
    # Minimal `which` to avoid importing shutil in constrained environments.
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


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    root = Path.cwd()
    tasks_path = root / "tasks" / "tasks.json"
    state_path = root / "runs" / "autorun_state.json"
    lock_path = root / "runs" / "autorun.lock"
    prompts_dir = root / "runs" / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(tasks_path)
    contract_errors = validate_task_contracts(tasks)
    if contract_errors:
        for err in contract_errors:
            print(f"[autorun] CONTRACT ERROR: {err}", file=sys.stderr)
        return EXIT_FAILURE
    if args.validate_task_contracts:
        print("[autorun] task contracts: OK")
        return EXIT_OK

    if shutil_which("codex") is None:
        print("[autorun] ERROR: `codex` command not found in PATH.", file=sys.stderr)
        print("[autorun] Install Codex CLI before running autorun.", file=sys.stderr)
        return EXIT_FAILURE

    lock = LockHandle(
        lock_path,
        payload={"pid": os.getpid(), "started_at": time.time(), "cwd": str(root)},
    )
    if not lock.acquire(timeout_sec=args.lock_timeout_sec):
        print(
            f"[autorun] ERROR: lock timeout ({args.lock_timeout_sec}s): {lock_path}",
            file=sys.stderr,
        )
        return EXIT_LOCK_TIMEOUT

    try:
        state = {"completed": {}, "created_at": time.time()} if args.dry_run else load_state(state_path)
        features = detect_codex_features()
        state.setdefault("codex_features", {})
        state["codex_features"] = features.__dict__
        state["argv"] = argv
        if not args.dry_run:
            save_state(state_path, state)

        git_repo = is_git_repo(root)
        if args.git_check == "strict" and not git_repo:
            print("[autorun] ERROR: --git-check strict but current directory is not a git repo.", file=sys.stderr)
            return EXIT_FAILURE

        completed: Dict[str, Any] = state.get("completed", {})
        executed_in_this_run = 0
        for task in tasks:
            tid = task["task_id"]
            if tid in completed and completed[tid].get("status") == "ok":
                print(f"[autorun] SKIP {tid} (already completed)")
                continue

            dep_errors = ensure_task_dependencies(task, completed)
            if dep_errors:
                for err in dep_errors:
                    print(f"[autorun] ERROR: {err}", file=sys.stderr)
                return EXIT_DEPENDENCY

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

            before_snapshot = snapshot_workspace(root)
            rc = codex_exec(
                features=features,
                args=args,
                root=root,
                prompt_text=prompt_text,
                prompt_file=prompt_file,
                git_repo=git_repo,
            )
            if rc != 0:
                print(f"[autorun] ERROR: task {tid} failed with codex exit code {rc}", file=sys.stderr)
                return rc

            after_snapshot = snapshot_workspace(root)
            scope_errors = validate_scope_limits(task, before_snapshot, after_snapshot)
            if scope_errors:
                for err in scope_errors:
                    print(f"[autorun] ERROR: {err}", file=sys.stderr)
                return EXIT_SCOPE

            if args.dry_run:
                print(f"[autorun] DRY-RUN DONE {tid} (state not updated)")
                completed[tid] = {"status": "ok", "completed_at": time.time()}
                executed_in_this_run += 1
                if args.max_tasks > 0 and executed_in_this_run >= args.max_tasks:
                    print(f"[autorun] MAX_TASKS reached ({args.max_tasks}) in dry-run mode.")
                    return EXIT_OK
                continue

            completed[tid] = {"status": "ok", "completed_at": time.time()}
            state["completed"] = completed
            state["last_completed"] = tid
            save_state(state_path, state)
            print(f"[autorun] DONE {tid}")
            executed_in_this_run += 1

            if args.max_tasks > 0 and executed_in_this_run >= args.max_tasks:
                print(f"[autorun] MAX_TASKS reached ({args.max_tasks}).")
                return EXIT_OK

            if task.get("stop_after", False):
                print(f"[autorun] STOP_AFTER triggered by {tid}. Exiting with {EXIT_CHECKPOINT}.")
                return EXIT_CHECKPOINT

        print("[autorun] All tasks completed.")
        return EXIT_OK
    finally:
        lock.release()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
