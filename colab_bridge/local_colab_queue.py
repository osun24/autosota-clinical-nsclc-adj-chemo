"""Local side of the Colab GPU worker queue.

This script is intended to be run by the local autoresearch loop. It creates
DeepSurv job JSON files in a queue directory shared with Colab, watches status,
and collects returned artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_URL = "https://github.com/osun24/autosota-nsclc-adj-chemo.git"
DEFAULT_COMMAND = "python deepsurv_arena/train.py"
DEFAULT_ENV = {
    "DEEPSURV_ARENA_N_TRIALS": "20",
    "DEEPSURV_ARENA_BOOTSTRAPS": "2",
    "DEEPSURV_ARENA_EPOCHS": "200",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def ensure_queue(root: Path) -> None:
    for name in ["queued", "running", "done", "failed", "cancelled"]:
        (root / name).mkdir(parents=True, exist_ok=True)


def git_output(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def current_commit() -> str:
    return git_output(["rev-parse", "HEAD"])


def current_branch() -> str:
    return git_output(["rev-parse", "--abbrev-ref", "HEAD"])


def worktree_dirty() -> bool:
    return bool(git_output(["status", "--short"]))


def remote_contains_commit(commit: str) -> bool:
    try:
        out = git_output(["branch", "-r", "--contains", commit])
    except subprocess.CalledProcessError:
        return False
    return bool(out.strip())


def parse_env(items: list[str]) -> dict[str, str]:
    env = dict(DEFAULT_ENV)
    for item in items:
        if "=" not in item:
            raise ValueError(f"--env must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        env[key] = value
    return env


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def submit(args: argparse.Namespace) -> None:
    queue_root = args.queue_root.resolve()
    ensure_queue(queue_root)
    if args.require_clean and worktree_dirty():
        raise SystemExit("Refusing to submit: git worktree is dirty. Commit first or omit --require-clean.")
    commit = args.commit or current_commit()
    if args.require_pushed and not remote_contains_commit(commit):
        raise SystemExit(f"Refusing to submit: commit {commit} is not visible on a remote branch. Push first.")
    if not args.require_pushed and not remote_contains_commit(commit):
        print(f"warning: commit {commit} is not visible on a remote branch; Colab may not be able to fetch it.")
    job_id = args.job_id or f"deepsurv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    job_path = queue_root / "queued" / f"{job_id}.json"
    if job_path.exists():
        raise SystemExit(f"Job already exists: {job_path}")
    job = {
        "schema_version": 1,
        "job_id": job_id,
        "arena": "deepsurv",
        "repo_url": args.repo_url,
        "branch": args.branch or current_branch(),
        "commit": commit,
        "command": args.command,
        "env": parse_env(args.env),
        "created_at": utc_now(),
        "created_by": "local_colab_queue.py",
        "lease_seconds": int(args.lease_seconds),
        "heartbeat_seconds": int(args.heartbeat_seconds),
        "max_attempts": int(args.max_attempts),
        "attempt": 0,
        "notes": args.notes or "",
        "sealed_test_policy": "Colab worker copies clinical train/validation only; the test CSV is not part of this protocol.",
    }
    write_json_atomic(job_path, job)
    print(f"queued {job_id}: {job_path}")


def find_job(queue_root: Path, job_id: str) -> tuple[str, Path] | None:
    for state in ["queued", "running", "done", "failed", "cancelled"]:
        direct = queue_root / state / f"{job_id}.json"
        if direct.exists():
            return state, direct
        done_meta = queue_root / state / job_id / "job.json"
        if done_meta.exists():
            return state, done_meta
    return None


def status(args: argparse.Namespace) -> None:
    queue_root = args.queue_root.resolve()
    ensure_queue(queue_root)
    if args.job_id:
        found = find_job(queue_root, args.job_id)
        if not found:
            raise SystemExit(f"No job found for {args.job_id}")
        state, path = found
        payload = json.loads(path.read_text())
        print(json.dumps({"state": state, "path": str(path), **payload}, indent=2, sort_keys=True))
        return
    for state in ["queued", "running", "done", "failed", "cancelled"]:
        count = len(list((queue_root / state).glob("*.json"))) + len([p for p in (queue_root / state).glob("*") if p.is_dir()])
        print(f"{state}: {count}")


def watch(args: argparse.Namespace) -> None:
    queue_root = args.queue_root.resolve()
    terminal = {"done", "failed", "cancelled"}
    while True:
        found = find_job(queue_root, args.job_id)
        if not found:
            print(f"{utc_now()} missing {args.job_id}")
        else:
            state, path = found
            payload = json.loads(path.read_text())
            msg = f"{utc_now()} {args.job_id} state={state} attempt={payload.get('attempt')}"
            if state == "running":
                msg += f" worker={payload.get('worker_id')} heartbeat={payload.get('last_heartbeat')}"
            print(msg, flush=True)
            if state in terminal:
                break
        time.sleep(float(args.interval))


def collect(args: argparse.Namespace) -> None:
    queue_root = args.queue_root.resolve()
    done_dir = queue_root / "done" / args.job_id
    if not done_dir.exists():
        raise SystemExit(f"Done directory does not exist: {done_dir}")
    result_path = done_dir / "result.json"
    if result_path.exists():
        print(result_path.read_text())
    artifact_tar = done_dir / "repo_artifacts.tar.gz"
    if artifact_tar.exists():
        with tarfile.open(artifact_tar, "r:gz") as tar:
            safe_extract(tar, REPO_ROOT)
        print(f"extracted artifacts into {REPO_ROOT}")
    else:
        print("no repo_artifacts.tar.gz found")
    if args.copy_logs:
        log_dir = REPO_ROOT / "colab_jobs_collected" / args.job_id
        log_dir.mkdir(parents=True, exist_ok=True)
        for name in ["stdout.txt", "stderr.txt", "job.json", "result.json"]:
            src = done_dir / name
            if src.exists():
                shutil.copy2(src, log_dir / name)
        print(f"copied logs to {log_dir}")


def safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        if not str(target).startswith(str(dest) + os.sep):
            raise RuntimeError(f"Refusing unsafe tar member path: {member.name}")
    tar.extractall(dest)


def requeue_stale(args: argparse.Namespace) -> None:
    queue_root = args.queue_root.resolve()
    ensure_queue(queue_root)
    now = datetime.now(timezone.utc)
    moved = 0
    for path in (queue_root / "running").glob("*.json"):
        payload = json.loads(path.read_text())
        expires_at = parse_time(payload.get("lease_expires_at"))
        if expires_at is None or expires_at > now:
            continue
        if int(payload.get("attempt", 0)) >= int(payload.get("max_attempts", args.max_attempts)):
            failed_dir = queue_root / "failed" / payload["job_id"]
            failed_dir.mkdir(parents=True, exist_ok=True)
            payload["failed_at"] = utc_now()
            payload["failure_reason"] = "stale lease exceeded max_attempts"
            write_json_atomic(failed_dir / "job.json", payload)
            path.unlink()
            moved += 1
            continue
        payload["requeued_at"] = utc_now()
        payload["requeue_reason"] = "stale running lease"
        payload["attempt"] = int(payload.get("attempt", 0)) + 1
        for key in ["worker_id", "started_at", "last_heartbeat", "lease_expires_at"]:
            payload.pop(key, None)
        write_json_atomic(queue_root / "queued" / path.name, payload)
        path.unlink()
        moved += 1
    print(f"requeued_or_failed {moved} stale job(s)")


def cancel(args: argparse.Namespace) -> None:
    queue_root = args.queue_root.resolve()
    ensure_queue(queue_root)
    found = find_job(queue_root, args.job_id)
    if not found:
        raise SystemExit(f"No job found for {args.job_id}")
    state, path = found
    if state in {"done", "failed", "cancelled"}:
        raise SystemExit(f"Cannot cancel terminal job in state {state}")
    payload = json.loads(path.read_text())
    payload["cancelled_at"] = utc_now()
    payload["cancelled_by"] = "local_colab_queue.py"
    write_json_atomic(queue_root / "cancelled" / f"{args.job_id}.json", payload)
    path.unlink()
    print(f"cancelled {args.job_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local side of the Colab DeepSurv GPU queue.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_queue_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument("--queue-root", type=Path, default=Path(os.environ.get("COLAB_QUEUE_ROOT", "colab_jobs")))

    p = sub.add_parser("submit")
    add_queue_arg(p)
    p.add_argument("--job-id")
    p.add_argument("--repo-url", default=os.environ.get("COLAB_REPO_URL", DEFAULT_REPO_URL))
    p.add_argument("--branch")
    p.add_argument("--commit")
    p.add_argument("--command", default=DEFAULT_COMMAND)
    p.add_argument("--env", action="append", default=[])
    p.add_argument("--notes")
    p.add_argument("--lease-seconds", type=int, default=1800)
    p.add_argument("--heartbeat-seconds", type=int, default=30)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--require-clean", action="store_true")
    p.add_argument("--require-pushed", action="store_true", help="Refuse jobs whose commit is not visible on a remote branch.")
    p.set_defaults(func=submit)

    for name, func in [("status", status), ("watch", watch), ("collect", collect), ("requeue-stale", requeue_stale), ("cancel", cancel)]:
        p = sub.add_parser(name)
        add_queue_arg(p)
        if name != "status":
            p.add_argument("--job-id", required=name not in {"requeue-stale"})
        else:
            p.add_argument("--job-id")
        if name == "watch":
            p.add_argument("--interval", type=float, default=30.0)
        if name == "collect":
            p.add_argument("--copy-logs", action="store_true")
        if name == "requeue-stale":
            p.add_argument("--max-attempts", type=int, default=3)
        p.set_defaults(func=func)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
