"""Colab side DeepSurv GPU worker.

Run this in a manually started Colab GPU runtime. It leases DeepSurv jobs from
a shared Drive queue, runs the requested commit, and writes artifacts back to
the queue. It must never copy or load the sealed test CSV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import tarfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

REQUIRED_DATA_FILES = ["clinicalTrain.csv", "clinicalValidation.csv"]
FORBIDDEN_DATA_FILES = ["affyfRMATest.csv"]
DEFAULT_REPO_URL = "https://github.com/osun24/autosota-nsclc-adj-chemo.git"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def ensure_queue(root: Path) -> None:
    for name in ["queued", "running", "done", "failed", "cancelled"]:
        (root / name).mkdir(parents=True, exist_ok=True)


def assert_gpu() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU available in this Colab runtime.")
    print(f"CUDA GPU: {torch.cuda.get_device_name(0)}")


def assert_data_dir(data_dir: Path) -> None:
    for name in REQUIRED_DATA_FILES:
        if not (data_dir / name).exists():
            raise FileNotFoundError(f"Missing required train/validation data file: {data_dir / name}")
    for name in FORBIDDEN_DATA_FILES:
        if (data_dir / name).exists():
            raise RuntimeError(f"Forbidden sealed test file present in Colab data dir: {data_dir / name}")


def run_cmd(cmd: list[str] | str, cwd: Path | None = None, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, env=env, text=True, check=check, shell=isinstance(cmd, str), capture_output=True)


def ensure_repo(repo_dir: Path, repo_url: str) -> None:
    if (repo_dir / ".git").exists():
        run_cmd(["git", "fetch", "--all", "--prune"], cwd=repo_dir)
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(["git", "clone", repo_url, str(repo_dir)], cwd=repo_dir.parent)


def checkout_commit(repo_dir: Path, commit: str) -> None:
    run_cmd(["git", "fetch", "--all", "--prune"], cwd=repo_dir)
    run_cmd(["git", "checkout", "--force", commit], cwd=repo_dir)
    run_cmd(
        [
            "git",
            "clean",
            "-fdx",
            "-e",
            "clinicalTrain.csv",
            "-e",
            "clinicalValidation.csv",
            "-e",
            ".colab_requirements_sha256",
        ],
        cwd=repo_dir,
    )


def install_requirements_if_needed(repo_dir: Path) -> None:
    req = repo_dir / "requirements.txt"
    if not req.exists():
        return
    digest = hashlib.sha256(req.read_bytes()).hexdigest()
    stamp = repo_dir / ".colab_requirements_sha256"
    if stamp.exists() and stamp.read_text().strip() == digest:
        return
    print("installing requirements.txt")
    proc = subprocess.run(
        ["python", "-m", "pip", "install", "-q", "-r", str(req)],
        cwd=repo_dir,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"pip install -r requirements.txt failed with return code {proc.returncode}")
    stamp.write_text(digest + "\n")


def copy_train_valid_data(data_dir: Path, repo_dir: Path) -> None:
    assert_data_dir(data_dir)
    for name in REQUIRED_DATA_FILES:
        shutil.copy2(data_dir / name, repo_dir / name)
    for name in FORBIDDEN_DATA_FILES:
        forbidden = repo_dir / name
        if forbidden.exists():
            raise RuntimeError(f"Forbidden sealed test file present in worker repo: {forbidden}")


def recover_stale_running(queue_root: Path) -> None:
    now = datetime.now(timezone.utc)
    for path in (queue_root / "running").glob("*.json"):
        job = json.loads(path.read_text())
        expires_at = parse_time(job.get("lease_expires_at"))
        if expires_at is None or expires_at > now:
            continue
        if int(job.get("attempt", 0)) >= int(job.get("max_attempts", 3)):
            failed_dir = queue_root / "failed" / job["job_id"]
            failed_dir.mkdir(parents=True, exist_ok=True)
            job["failed_at"] = utc_now()
            job["failure_reason"] = "stale lease exceeded max_attempts"
            write_json_atomic(failed_dir / "job.json", job)
            path.unlink()
            print(f"failed stale job after max attempts: {job['job_id']}")
            continue
        job["attempt"] = int(job.get("attempt", 0)) + 1
        job["requeued_at"] = utc_now()
        job["requeue_reason"] = "stale running lease after worker disconnect"
        for key in ["worker_id", "started_at", "last_heartbeat", "lease_expires_at"]:
            job.pop(key, None)
        write_json_atomic(queue_root / "queued" / f"{job['job_id']}.json", job)
        path.unlink()
        print(f"requeued stale job: {job['job_id']}")


def lease_one_job(queue_root: Path, worker_id: str) -> tuple[dict, Path] | None:
    for path in sorted((queue_root / "queued").glob("*.json")):
        job = json.loads(path.read_text())
        if int(job.get("attempt", 0)) >= int(job.get("max_attempts", 3)):
            failed_dir = queue_root / "failed" / job["job_id"]
            failed_dir.mkdir(parents=True, exist_ok=True)
            job["failed_at"] = utc_now()
            job["failure_reason"] = "max_attempts reached before pickup"
            write_json_atomic(failed_dir / "job.json", job)
            path.unlink()
            continue
        job["worker_id"] = worker_id
        job["started_at"] = utc_now()
        job["last_heartbeat"] = utc_now()
        lease_seconds = int(job.get("lease_seconds", 1800))
        job["lease_expires_at"] = (datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)).isoformat()
        running_path = queue_root / "running" / path.name
        try:
            path.replace(running_path)
        except FileNotFoundError:
            continue
        write_json_atomic(running_path, job)
        return job, running_path
    return None


def heartbeat_loop(running_path: Path, stop: threading.Event, interval: int) -> None:
    while not stop.wait(max(5, int(interval))):
        if not running_path.exists():
            return
        job = json.loads(running_path.read_text())
        job["last_heartbeat"] = utc_now()
        lease_seconds = int(job.get("lease_seconds", 1800))
        job["lease_expires_at"] = (datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)).isoformat()
        write_json_atomic(running_path, job)


def newest_run_dirs(repo_dir: Path, before: set[str]) -> list[Path]:
    runs = repo_dir / "deepsurv_arena" / "runs"
    if not runs.exists():
        return []
    after = {p.name for p in runs.iterdir() if p.is_dir()}
    names = sorted(after - before)
    if names:
        return [runs / name for name in names]
    existing = sorted([p for p in runs.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    return existing[:1]


def make_artifact_tar(repo_dir: Path, run_dirs: list[Path], out_tar: Path) -> None:
    with tarfile.open(out_tar, "w:gz") as tar:
        for run_dir in run_dirs:
            tar.add(run_dir, arcname=str(run_dir.relative_to(repo_dir)))


def parse_result(run_dirs: list[Path]) -> dict:
    for run_dir in run_dirs:
        result_path = run_dir / "result.json"
        if result_path.exists():
            return json.loads(result_path.read_text())
    return {}


def run_job(job: dict, running_path: Path, args: argparse.Namespace, worker_id: str) -> None:
    repo_dir = args.work_dir / "repo"
    output_base = args.queue_root / "_worker_outputs" / job["job_id"]
    output_base.mkdir(parents=True, exist_ok=True)
    stdout_path = output_base / "stdout.txt"
    stderr_path = output_base / "stderr.txt"
    stop_hb = threading.Event()
    hb = threading.Thread(
        target=heartbeat_loop,
        args=(running_path, stop_hb, int(job.get("heartbeat_seconds", 30))),
        daemon=True,
    )
    hb.start()
    return_code = None
    try:
        ensure_repo(repo_dir, job.get("repo_url") or args.repo_url)
        checkout_commit(repo_dir, job["commit"])
        if args.install_requirements:
            install_requirements_if_needed(repo_dir)
        copy_train_valid_data(args.data_dir, repo_dir)
        runs_dir = repo_dir / "deepsurv_arena" / "runs"
        before = {p.name for p in runs_dir.iterdir() if p.is_dir()} if runs_dir.exists() else set()
        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in job.get("env", {}).items()})
        env.setdefault("PYTHONUNBUFFERED", "1")
        command = job.get("command", "python deepsurv_arena/train.py")
        with stdout_path.open("w") as stdout_f, stderr_path.open("w") as stderr_f:
            proc = subprocess.Popen(command, cwd=repo_dir, env=env, shell=True, text=True, stdout=stdout_f, stderr=stderr_f)
            return_code = proc.wait()
        run_dirs = newest_run_dirs(repo_dir, before)
        result = parse_result(run_dirs)
        result.update({
            "job_id": job["job_id"],
            "worker_id": worker_id,
            "return_code": int(return_code),
            "commit": job["commit"],
            "completed_at": utc_now(),
        })
        if return_code == 0:
            done_dir = args.queue_root / "done" / job["job_id"]
            done_dir.mkdir(parents=True, exist_ok=True)
            make_artifact_tar(repo_dir, run_dirs, done_dir / "repo_artifacts.tar.gz")
            write_json_atomic(done_dir / "result.json", result)
            job["completed_at"] = utc_now()
            job["return_code"] = int(return_code)
            write_json_atomic(done_dir / "job.json", job)
            shutil.copy2(stdout_path, done_dir / "stdout.txt")
            shutil.copy2(stderr_path, done_dir / "stderr.txt")
            running_path.unlink(missing_ok=True)
            print(f"done {job['job_id']}")
        else:
            failed_dir = args.queue_root / "failed" / job["job_id"]
            failed_dir.mkdir(parents=True, exist_ok=True)
            result["failure_reason"] = f"command returned {return_code}"
            write_json_atomic(failed_dir / "result.json", result)
            job["failed_at"] = utc_now()
            job["return_code"] = int(return_code)
            job["failure_reason"] = result["failure_reason"]
            write_json_atomic(failed_dir / "job.json", job)
            shutil.copy2(stdout_path, failed_dir / "stdout.txt")
            shutil.copy2(stderr_path, failed_dir / "stderr.txt")
            running_path.unlink(missing_ok=True)
            print(f"failed {job['job_id']} return_code={return_code}")
    except Exception as exc:
        failed_dir = args.queue_root / "failed" / job["job_id"]
        failed_dir.mkdir(parents=True, exist_ok=True)
        job["failed_at"] = utc_now()
        job["failure_reason"] = repr(exc)
        if return_code is not None:
            job["return_code"] = int(return_code)
        write_json_atomic(failed_dir / "job.json", job)
        if stdout_path.exists():
            shutil.copy2(stdout_path, failed_dir / "stdout.txt")
        if stderr_path.exists():
            shutil.copy2(stderr_path, failed_dir / "stderr.txt")
        running_path.unlink(missing_ok=True)
        print(f"failed {job['job_id']}: {exc!r}")
    finally:
        stop_hb.set()
        hb.join(timeout=5)


def worker_loop(args: argparse.Namespace) -> None:
    args.queue_root = args.queue_root.resolve()
    args.data_dir = args.data_dir.resolve()
    args.work_dir = args.work_dir.resolve()
    ensure_queue(args.queue_root)
    assert_gpu()
    assert_data_dir(args.data_dir)
    worker_id = f"{socket.gethostname()}-{uuid4().hex[:8]}"
    print(f"worker_id={worker_id}")
    while True:
        recover_stale_running(args.queue_root)
        leased = lease_one_job(args.queue_root, worker_id)
        if leased is None:
            if args.once:
                print("no queued jobs")
                return
            time.sleep(float(args.poll_seconds))
            continue
        job, running_path = leased
        if job.get("arena") != "deepsurv":
            failed_dir = args.queue_root / "failed" / job["job_id"]
            failed_dir.mkdir(parents=True, exist_ok=True)
            job["failed_at"] = utc_now()
            job["failure_reason"] = f"unsupported arena {job.get('arena')}"
            write_json_atomic(failed_dir / "job.json", job)
            running_path.unlink(missing_ok=True)
            continue
        print(f"picked up {job['job_id']} commit={job['commit']}")
        run_job(job, running_path, args, worker_id)
        if args.once:
            return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Colab DeepSurv GPU worker.")
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--work-dir", type=Path, default=Path("/content/autosota_colab_worker"))
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--install-requirements", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--once", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    worker_loop(args)


if __name__ == "__main__":
    main()
