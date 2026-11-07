#!/usr/bin/env python3
"""Exercise the actual generated Triton card, without starting an agent campaign.

The candidate is a fixed RFC/public diagnostic kernel, not an optimized agent
result. Both correctness and timing come from the full official CANN evaluator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
import uuid

from forgekernel import taskcard
from forgekernel.evidence import archive_job_artifacts
from forgekernel.localenv import load_machine
from forgekernel.loop.parse import parse_eval
from forgekernel.paths import repo_root
from forgekernel.submit.config import load_config
from forgekernel.submit.direct import DirectRunner
from forgekernel.task import load_task


def cleanup_acceptance(card: Path, run_dir: Path, out: Path) -> dict:
    """Prove TERM/KILL cleanup with a real NPU-owning new-session orphan."""
    from dataclasses import replace
    notes = card / 'notes'
    notes.mkdir(exist_ok=True)
    shutil.copy2(repo_root() / 'repro/ascend/triton_backend/cleanup_probe.py', notes / 'cleanup_probe.py')
    taskcard._commit(card, 'test: bounded NPU descendant cleanup probe')
    cfg = replace(load_config(card.name, base_dir=run_dir), timeout=90)
    runner = DirectRunner(cfg)
    spawn_log = out / 'cleanup-spawn.log'
    with spawn_log.open('w') as handle:
        spawn_status = runner.run_mode('main', 'cmd:python3 notes/cleanup_probe.py spawn', handle)
    marker = 'NPU-CLEANUP-WORKER '
    workers = [json.loads(line[len(marker):]) for line in spawn_log.read_text().splitlines()
               if line.startswith(marker)]
    if spawn_status != 'Succeeded' or len(workers) != 1:
        raise RuntimeError(f'cleanup worker did not run; see {spawn_log}')
    worker = workers[0]
    check_log = out / 'cleanup-check.log'
    with check_log.open('w') as handle:
        check_status = runner.run_mode('main', 'cmd:python3 notes/cleanup_probe.py check '
            f'--pid {int(worker["pid"])} --birth {int(worker["birth"])}', handle)
    if check_status != 'Succeeded' or 'NPU-CLEANUP-OK ' not in check_log.read_text():
        raise RuntimeError(f'NPU cleanup failed; see {check_log}')
    result = {'spawn_status': spawn_status, 'check_status': check_status, 'worker': worker,
              'owned_worker_alive': False, 'fresh_npu_sum': 4,
              'scope': 'new-session orphan ignoring TERM; supervisor KILL/reaping and fresh allocation'}
    (out / 'cleanup.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cann-bench-dir', type=Path, required=True,
                        help='local checkout of exactly the remote CANN-Bench commit')
    parser.add_argument('--machine-root', type=Path, default=repo_root(),
                        help='directory containing the controller local.env')
    parser.add_argument('--operator', choices=('exp', 'gqa'), default='exp')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--reference-jobs', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--cleanup-check', action='store_true',
                        help='after evaluation, verify cleanup of a real NPU-owning escaped worker')
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error('--out must be a fresh directory')
    if min(args.jobs, args.reference_jobs, args.timeout) < 1:
        parser.error('jobs and timeout must be positive')
    root = repo_root()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True)
    sys.path.insert(0, str(root / 'scripts'))
    from init_pypto_cann_task import main as build_task
    task_dir = args.out / 'tasks' / f'{args.operator}-triton'
    options = [str(task_dir), '--language', 'triton', '--cann-bench-dir', str(args.cann_bench_dir)]
    if args.operator == 'exp':
        reference = root / 'repro/ascend/public_triton/cann_bench/exp.py'
        options += ['--level', 'level1', '--operator-path', 'exp', '--operator', 'Exp',
                    '--entry', 'exp', '--signature', 'x: torch.Tensor, base: float = -1.0, '
                    'scale: float = 1.0, shift: float = 0.0', '--reference-file', str(reference),
                    '--reference-name', 'fixed Triton diagnostic baseline (not a vendor A/B)']
        candidate = reference.read_text()
    else:
        candidate = (root / 'repro/ascend/triton_backend/kernels.py').read_text() + '''

def gqa(query, key, value, scaleValue=-1.0, is_causal=False):
    return launch_gqa(query, key, value, scale=scaleValue, causal=is_causal)[0]
'''
    build_task(options)
    task = load_task(task_dir)
    machine = load_machine(args.machine_root)
    # DirectRunner keys the remote tree by run-directory name. Never let two
    # independent acceptance invocations extract over one another's live job.
    run_dir = args.out / f'run-{uuid.uuid4().hex[:12]}'
    run = SimpleNamespace(dir=run_dir, operators_dir=run_dir / 'operators')
    card = taskcard.install_board_trees(run, task, machine)
    taskcard.write_board_loop_toml(run, task, card, deny=[str(task_dir / 'leaderboard')], ncu=False)
    # The student would author this file during a real campaign. Here the known,
    # unchanged diagnostic kernel isolates harness integration from agent quality.
    (card / 'kernel.py').write_text(candidate)
    taskcard._commit(card, 'test: fixed Triton backend diagnostic candidate')
    taskcard.clone_workspace(card, run.dir / 'ws')
    results = []
    for role, tree, jobs in (('candidate', card, args.jobs),
                             ('reference', run.operators_dir / taskcard.ref_tree(task.name),
                              args.reference_jobs)):
        taskcard.write_submit(tree, task, machine, timeout=args.timeout)
        taskcard._commit(tree, 'test: bounded hardware acceptance job')
        cfg = load_config(tree.name, base_dir=run.dir)
        if role == 'candidate':
            log = args.out / 'preflight.log'
            with log.open('w') as handle:
                status = DirectRunner(cfg).run_mode('main', 'preflight', handle)
            if status != 'Succeeded' or 'ASCEND-PREFLIGHT-OK triton=' not in log.read_text():
                raise RuntimeError(f'Triton NPU preflight failed; see {log}')
            print(json.dumps({'preflight': status}), flush=True)
        for i in range(jobs):
            log = args.out / f'{role}-{i + 1}.log'
            started = time.monotonic()
            with log.open('w') as handle:
                status = DirectRunner(cfg).run_mode('main', 'all', handle)
            text = log.read_text()
            artifacts = archive_job_artifacts(text, args.out / f'{role}-{i + 1}-artifacts')
            ev = parse_eval(text)
            result = {'role': role, 'job': i + 1, 'status': status,
                      'wall_seconds': time.monotonic() - started,
                      'correctness': ev.correctness, 'error': ev.error,
                      'device_median_ms': ev.device_median_ms,
                      'cann_bench': ev.cann_bench, 'artifact_count': len(artifacts)}
            if role == 'reference' and any(a['name'].startswith('triton/') for a in artifacts):
                raise RuntimeError('reference compilation evidence crossed the disclosure boundary')
            results.append(result)
            (args.out / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
            print(json.dumps({k: result[k] for k in ('role', 'job', 'status', 'wall_seconds',
                  'correctness', 'error', 'device_median_ms', 'artifact_count')}), flush=True)
            if not ev.correctness or ev.device_median_ms is None:
                return 2
    if args.cleanup_check:
        print(json.dumps({'cleanup': cleanup_acceptance(card, run.dir, args.out)}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
