from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path


DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
STILL_ACTIVE = 259


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def windows_process_exit_code(pid: int) -> int | None:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except OSError:
            return -1
        return None
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        return -1
    try:
        code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        if not ok:
            return -1
        return None if code.value == STILL_ACTIVE else int(code.value)
    finally:
        kernel32.CloseHandle(handle)


def read_meta(job_dir: Path) -> dict:
    meta_path = job_dir / "job.json"
    if not meta_path.exists():
        raise SystemExit(f"missing job metadata: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def write_text_tail(path: Path, chars: int) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    return data[-chars:].decode("utf-8", errors="replace").strip()


def build_llama_args(args: argparse.Namespace) -> list[str]:
    cmd = [
        str(resolve(args.llama_cli)),
        "-m",
        str(resolve(args.model)),
        "-f",
        str(resolve(args.prompt_file)),
        "-n",
        str(args.tokens),
        "-c",
        str(args.ctx),
        "-ngl",
        str(args.ngl),
        "--temp",
        str(args.temp),
        "--top-p",
        str(args.top_p),
        "--no-display-prompt",
        "--single-turn",
        "--simple-io",
        "--log-disable",
    ]
    if args.chat_template:
        cmd += ["--chat-template", args.chat_template]
    if args.reasoning:
        cmd += ["--reasoning", args.reasoning]
    if args.reasoning_budget is not None:
        cmd += ["--reasoning-budget", str(args.reasoning_budget)]
    if args.extra:
        cmd += args.extra
    return cmd


def command_start(args: argparse.Namespace) -> int:
    job_dir = resolve(args.out_dir) / args.name
    job_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = job_dir / "stdout.txt"
    stderr_path = job_dir / "stderr.txt"
    worker_stdout_path = job_dir / "worker.stdout.txt"
    worker_stderr_path = job_dir / "worker.stderr.txt"
    done_path = job_dir / "done.json"
    if done_path.exists():
        done_path.unlink()

    cmd = build_llama_args(args)
    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    meta = {
        "name": args.name,
        "started_at": utc_now(),
        "cwd": str(Path.cwd()),
        "worker_pid": None,
        "llama_pid": None,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "worker_stdout": str(worker_stdout_path),
        "worker_stderr": str(worker_stderr_path),
        "done": str(done_path),
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        "command": cmd,
    }

    creationflags = 0
    if os.name == "nt":
        creationflags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP

    meta_path = job_dir / "job.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    worker_cmd = [sys.executable, str(Path(__file__).resolve()), "_worker", str(job_dir)]
    with worker_stdout_path.open("wb") as stdout, worker_stderr_path.open("wb") as stderr:
        proc = subprocess.Popen(
            worker_cmd,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            cwd=str(Path.cwd()),
            env=env,
            creationflags=creationflags,
            close_fds=True,
        )
    meta["worker_pid"] = proc.pid
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"job_dir": str(job_dir), "worker_pid": proc.pid}, ensure_ascii=False))
    return 0


def command_worker(args: argparse.Namespace) -> int:
    job_dir = resolve(args.job_dir)
    meta_path = job_dir / "job.json"
    meta = read_meta(job_dir)
    env = os.environ.copy()
    if meta.get("cuda_visible_devices") is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(meta["cuda_visible_devices"])
    started = utc_now()
    with Path(meta["stdout"]).open("wb") as stdout, Path(meta["stderr"]).open("wb") as stderr:
        proc = subprocess.Popen(
            meta["command"],
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            cwd=meta["cwd"],
            env=env,
            close_fds=True,
        )
        meta["llama_pid"] = proc.pid
        meta["llama_started_at"] = started
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        exit_code = proc.wait()
    done = {
        "started_at": started,
        "finished_at": utc_now(),
        "exit_code": int(exit_code),
    }
    Path(meta["done"]).write_text(json.dumps(done, indent=2), encoding="utf-8")
    return int(exit_code)


def command_status(args: argparse.Namespace) -> int:
    job_dir = resolve(args.job_dir)
    meta = read_meta(job_dir)
    done_path = Path(meta["done"])
    done = json.loads(done_path.read_text(encoding="utf-8")) if done_path.exists() else None
    worker_pid = meta.get("worker_pid")
    llama_pid = meta.get("llama_pid")
    worker_exit = windows_process_exit_code(int(worker_pid)) if worker_pid else -1
    status = "exited" if done else ("running" if worker_exit is None else "unknown")
    exit_code = done.get("exit_code") if done else None
    result = {
        "job_dir": str(job_dir),
        "worker_pid": worker_pid,
        "llama_pid": llama_pid,
        "status": status,
        "exit_code": exit_code,
        "stdout_bytes": Path(meta["stdout"]).stat().st_size if Path(meta["stdout"]).exists() else 0,
        "stderr_bytes": Path(meta["stderr"]).stat().st_size if Path(meta["stderr"]).exists() else 0,
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0 if exit_code in (None, 0) else int(exit_code)


def command_wait(args: argparse.Namespace) -> int:
    deadline = time.monotonic() + args.timeout
    while True:
        meta = read_meta(resolve(args.job_dir))
        done_path = Path(meta["done"])
        if done_path.exists():
            done = json.loads(done_path.read_text(encoding="utf-8"))
            command_status(args)
            exit_code = int(done["exit_code"])
            return 0 if exit_code == 0 else int(exit_code)
        if time.monotonic() >= deadline:
            print(json.dumps({"status": "timeout", "job_dir": str(resolve(args.job_dir))}, ensure_ascii=False))
            return 124
        time.sleep(args.interval)


def command_show(args: argparse.Namespace) -> int:
    meta = read_meta(resolve(args.job_dir))
    stdout = write_text_tail(Path(meta["stdout"]), args.tail)
    stderr = write_text_tail(Path(meta["stderr"]), args.tail)
    if stdout:
        print(stdout)
    if stderr:
        print("\n[stderr]\n" + stderr)
    return 0


def extract_generation(stdout: str) -> str:
    text = stdout.replace("\r\n", "\n").replace("\r", "\n")
    marker = "<start_of_turn>model"
    if marker in text:
        text = text.rsplit(marker, 1)[-1]
    if "[ Prompt:" in text:
        text = text.split("[ Prompt:", 1)[0]
    if "Exiting..." in text:
        text = text.split("Exiting...", 1)[0]
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line and line != ">"]
    return "\n".join(lines).strip()


def command_result(args: argparse.Namespace) -> int:
    meta = read_meta(resolve(args.job_dir))
    stdout_path = Path(meta["stdout"])
    if not stdout_path.exists():
        return 1
    print(extract_generation(stdout_path.read_text(encoding="utf-8", errors="replace")))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start and inspect detached llama.cpp jobs.")
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start")
    start.add_argument("--name", required=True)
    start.add_argument("--out-dir", default="reports/llama_jobs")
    start.add_argument("--llama-cli", default=r"D:\Projects\training\llama_b9209_cuda\llama-cli.exe")
    start.add_argument("--model", required=True)
    start.add_argument("--prompt-file", required=True)
    start.add_argument("--gpu", default="1")
    start.add_argument("--tokens", type=int, default=128)
    start.add_argument("--ctx", type=int, default=2048)
    start.add_argument("--ngl", type=int, default=99)
    start.add_argument("--temp", type=float, default=0.2)
    start.add_argument("--top-p", type=float, default=0.9)
    start.add_argument("--reasoning", choices=["on", "off", "auto"], default="off")
    start.add_argument("--reasoning-budget", type=int, default=0)
    start.add_argument("--chat-template")
    start.add_argument("extra", nargs=argparse.REMAINDER)
    start.set_defaults(func=command_start)

    status = sub.add_parser("status")
    status.add_argument("job_dir")
    status.set_defaults(func=command_status)

    wait = sub.add_parser("wait")
    wait.add_argument("job_dir")
    wait.add_argument("--timeout", type=float, default=180.0)
    wait.add_argument("--interval", type=float, default=1.0)
    wait.set_defaults(func=command_wait)

    show = sub.add_parser("show")
    show.add_argument("job_dir")
    show.add_argument("--tail", type=int, default=4000)
    show.set_defaults(func=command_show)

    result = sub.add_parser("result")
    result.add_argument("job_dir")
    result.set_defaults(func=command_result)

    worker = sub.add_parser("_worker")
    worker.add_argument("job_dir")
    worker.set_defaults(func=command_worker)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
