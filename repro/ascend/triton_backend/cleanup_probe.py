#!/usr/bin/env python3
"""NPU cleanup acceptance probe. Run ONLY through a generated Triton gpu_entry.

`spawn` leaves an NPU-owning, TERM-ignoring new-session orphan for the entry's
subreaper to end. After that job returns, `check` verifies its identity is gone
and a fresh NPU allocation/synchronization succeeds. No operator gate/KB claim.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def identity(pid):
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
        fields = text[text.rindex(')') + 2:].split()
        return fields[19] if fields[0] != 'Z' else None
    except FileNotFoundError:
        return None


def under_supervisor():
    pid = os.getppid()
    for _ in range(32):
        if pid <= 1:
            return False
        try:
            root = Path(f'/proc/{pid}')
            if b'ascend_process.py' in root.joinpath('cmdline').read_bytes().split(b'\0'):
                return True
            # An absolute script path is the usual invocation.
            if any(p.endswith(b'/ascend_process.py') for p in root.joinpath('cmdline').read_bytes().split(b'\0')):
                return True
            text = root.joinpath('stat').read_text()
            pid = int(text[text.rindex(')') + 2:].split()[1])
        except (OSError, ValueError):
            return False
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('spawn', 'check'))
    parser.add_argument('--pid', type=int)
    parser.add_argument('--birth')
    args = parser.parse_args(argv)
    if args.mode == 'check':
        if not args.pid or not args.birth:
            parser.error('check requires --pid and --birth from the earlier spawn job')
        if identity(args.pid) == args.birth:
            raise RuntimeError('owned NPU worker survived job cleanup')
        import torch
        import torch_npu  # noqa: F401
        value = torch.ones(4, device='npu').sum().item()
        torch.npu.synchronize()
        if value != 4:
            raise RuntimeError('post-cleanup device allocation failed')
        print('NPU-CLEANUP-OK ' + json.dumps({'pid': args.pid, 'birth': args.birth,
              'owned_worker_alive': False, 'fresh_npu_sum': value}), flush=True)
        return 0
    if not under_supervisor():
        raise RuntimeError('spawn must run under the frozen Ascend process supervisor')
    with tempfile.TemporaryDirectory(prefix='fk-cleanup-probe-') as tmp:
        ready = Path(tmp) / 'ready'
        worker = f'''import json, os, signal, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
import torch, torch_npu
value = torch.ones(1024, device="npu")
assert value.sum().item() == 1024
torch.npu.synchronize()
signal.signal(signal.SIGTERM, signal.SIG_IGN)
text = Path("/proc/self/stat").read_text()
birth = text[text.rindex(")") + 2:].split()[19]
Path({str(ready)!r}).write_text(json.dumps({{"pid": os.getpid(), "birth": birth}}))
os.close(1); os.close(2)
time.sleep(600)
'''
        middle = f'''import subprocess, sys
subprocess.Popen([sys.executable, "-c", {worker!r}], start_new_session=True)
'''
        proc = subprocess.Popen([sys.executable, '-c', middle], start_new_session=True)
        proc.wait(timeout=10)
        deadline = time.monotonic() + 45
        while not ready.is_file() and time.monotonic() < deadline:
            time.sleep(0.1)
        if not ready.is_file():
            raise RuntimeError('escaped NPU worker did not become ready')
        print('NPU-CLEANUP-WORKER ' + ready.read_text(), flush=True)
    # The frozen gpu_entry's subreaper, not this probe, must discover and kill it.
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
