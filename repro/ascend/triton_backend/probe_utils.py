"""Bounded evidence helpers. Importable without Torch, Triton, or an NPU."""
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import statistics
import time


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str)+'\n')


def fingerprint(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def environment():
    import torch
    import torch_npu
    import triton
    from triton.runtime import driver
    from triton.compiler.compiler import make_backend
    target = driver.active.get_current_target()
    if target.backend != 'npu':
        raise RuntimeError(f'this diagnostic requires an NPU backend, got {target}')
    return dict(device_name=torch.npu.get_device_name(), target=str(target),
                versions={p: importlib.metadata.version(p)
                          for p in ('triton', 'triton-ascend', 'torch', 'torch-npu')},
                modules={m.__name__: fingerprint(m.__file__) for m in (torch, torch_npu, triton)},
                compiler_options=vars(make_backend(target).parse_options({})))


def install_compile_listener(out):
    """Observe documented compiler calls, including disk hits; no monkeypatch."""
    import triton
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    records = []

    def listener(*, src, metadata, metadata_group, times, cache_hit):
        record = dict(name=metadata['name'], hash=metadata['hash'], cache_hit=cache_hit,
                      compiler_total_us=times.total, lowering_us=times.total_lowering,
                      lowering_stages_us=times.lowering_stages)
        records.append(record)
        directory = out / metadata['hash']
        directory.mkdir(exist_ok=True)
        for name, source in metadata_group.items():
            path = Path(source)
            if path.is_file():
                shutil.copy2(path, directory/path.name)
        save(directory/'observed_metadata.json', metadata)
        save(out/'compilations.json', records)

    previous = triton.knobs.compilation.listener
    triton.knobs.compilation.listener = listener
    return records, previous


def host_samples(fn, batch=16, trials=5):
    import torch
    host, event = [], []
    for _ in range(trials):
        torch.npu.synchronize()
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        begin = time.perf_counter_ns()
        for _ in range(batch):
            fn()
        host.append((time.perf_counter_ns()-begin)/1000/batch)
        end.record()
        torch.npu.synchronize()
        event.append(start.elapsed_time(end)*1000/batch)
    return dict(host_callable_us=host, event_stream_us=event,
                host_median_us=statistics.median(host), event_median_us=statistics.median(event),
                scope='preallocated output; hot cache; event may include host supply gaps')


def single_task_profile_summary(csv_path, calls):
    with Path(csv_path).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if calls < 1 or len(rows) != calls or len({r['Name'] for r in rows}) != 1:
        raise ValueError('not one recorded task per call; logical call grouping required')
    values = [float(r['Duration(us)']) for r in rows]
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('invalid task duration')
    return dict(calls=calls, recorded_tasks=len(rows), name=rows[0]['Name'],
                types=sorted({r['Type'] for r in rows}),
                block_nums=sorted({r['Block Num'] for r in rows}),
                single_task_duration_us=values, median_us=statistics.median(values),
                scope='one task per call verified, not an arbitrary multi-task duration sum')


def profile(fn, out, calls=5):
    import torch
    import torch_npu
    out = Path(out)
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out)),
    ):
        for _ in range(calls):
            fn()
        torch.npu.synchronize()
    files = list(out.rglob('kernel_details.csv'))
    if len(files) != 1:
        raise ValueError('expected one profiler export in a fresh output directory')
    return single_task_profile_summary(files[0], calls)
