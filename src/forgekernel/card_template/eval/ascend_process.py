"""Linux subreaper for an evaluator and compiler workers that create new sessions.

The supervisor owns only descendants it starts. A process group alone cannot end
setsid() workers; adoption keeps even double-forked workers in this ownership tree.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


class Stopped(Exception):
    pass


def processes() -> dict[int, tuple[int, str, str]]:
    out = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            text = path.read_text()
            fields = text[text.rindex(")") + 2:].split()
            out[int(path.parent.name)] = (int(fields[1]), fields[0], fields[19])
        except (OSError, ValueError, IndexError):
            pass
    return out


def descendants(root: int, table: dict) -> dict:
    owned = {root}
    while True:
        more = {pid for pid, (ppid, _, _) in table.items() if ppid in owned}
        if more <= owned:
            break
        owned.update(more)
    return {pid: table[pid] for pid in owned - {root} if table[pid][1] != "Z"}


def end_descendants(grace_s: float = 5.0) -> bool:
    root = os.getpid()
    deadline = time.monotonic() + grace_s
    signalled = set()
    while True:
        table = processes()
        owned = descendants(root, table)
        if not owned:
            break
        hard = time.monotonic() >= deadline
        for pid, (_, _, birth) in owned.items():
            identity = (pid, birth, hard)
            if identity not in signalled:
                try:
                    os.kill(pid, signal.SIGKILL if hard else signal.SIGTERM)
                except ProcessLookupError:
                    pass
                signalled.add(identity)
        # Reap adopted orphans; a zombie is not device work, but must not accumulate.
        while True:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
                if pid == 0:
                    break
            except ChildProcessError:
                break
        if time.monotonic() >= deadline + grace_s:
            print("[ascend-process] cleanup failed, surviving owned pids: "
                  + ",".join(map(str, owned)), flush=True)
            return False
        time.sleep(0.05)
    while True:
        try:
            if os.waitpid(-1, os.WNOHANG)[0] == 0:
                break
        except ChildProcessError:
            break
    return True


def run(argv: list[str], grace_s: float = 5.0) -> int:
    if sys.platform != "linux" or not Path("/proc/self/stat").is_file():
        raise RuntimeError("Ascend evaluator cleanup requires Linux /proc and PR_SET_CHILD_SUBREAPER")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "cannot become evaluator subreaper")

    def stop(signum, _frame):
        raise Stopped(f"signal {signum}")
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    rc = 2
    try:
        proc = subprocess.Popen(argv)
        try:
            rc = proc.wait()
        except Stopped as exc:
            print(f"[ascend-process] stopped: {exc}; ending owned descendants", flush=True)
            rc = 143
    finally:
        # A second TERM must not interrupt cleanup or strand a new-session compiler.
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        if not end_descendants(grace_s):
            rc = 137
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return rc if rc >= 0 else 128 - rc


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--":
        args.pop(0)
    raise SystemExit(run(args))
