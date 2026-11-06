#!/usr/bin/env python3
"""Run bounded inputs on the unmodified official Pro FlashAttention kernel.

This validates language mechanisms, not the public 20-case GQA benchmark.
All comparisons use an independent CPU FP64 attention computation.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    base = Path(__file__).resolve().parent
    source = (base / args.source).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)
    import torch
    import torch_npu
    torch.npu.set_device(args.device)
    device = f'npu:{args.device}'
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location('official_fa', source)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    records = []
    # Small/full tiles, both tails, chained KV, multiple logical blocks.
    for sq, sk, blocks in [(64, 64, 1), (34, 62, 1), (162, 190, 1),
                           (512, 512, 4), (1, 2048, 1)]:
        torch.manual_seed(20261106)
        q, k, v = [(torch.rand(n, 128) * 2 - 1).half() for n in (sq, sk, sk)]
        expected = (torch.softmax(q.double() @ k.double().T / math.sqrt(128), -1)
                    @ v.double()).half()
        q, k, v = [t.to(device) for t in (q, k, v)]
        y = torch.full((sq, 128), float('nan'), dtype=torch.float16, device=device)
        call = lambda: mod.fa_perf_tkv_preload_dn_kernel[None, blocks](q, k, v, y)
        start = time.perf_counter()
        readings = []
        previous = None
        for _ in range(3):
            y.fill_(float('nan'))
            call()
            torch.npu.synchronize()
            got = y.cpu()
            torch.testing.assert_close(got, expected, rtol=5e-3, atol=5e-3)
            if previous is not None:
                assert torch.equal(got, previous), 'non-deterministic attention output'
            previous = got.clone()
            readings.append(float((got-expected).abs().max()))
        elapsed = time.perf_counter() - start
        enqueue, stream = [], []
        for _ in range(3):
            torch.npu.synchronize()
            s, e = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            s.record()
            begin = time.perf_counter_ns()
            for _ in range(8):
                call()
            enqueue.append((time.perf_counter_ns()-begin)/8000)
            e.record()
            torch.npu.synchronize()
            stream.append(s.elapsed_time(e)*1000/8)
        record = dict(sq=sq, sk=sk, dim=128, blocks=blocks,
                      first_three_calls_including_compile_s=elapsed,
                      max_abs_samples=readings,
                      all_output_finite=bool(torch.isfinite(got).all()),
                      deterministic_output=True,
                      compiled_instances=len(mod.fa_perf_tkv_preload_dn_kernel._compiled_by_signature),
                      host_callable_us=enqueue, event_stream_us=stream,
                      host_median_us=statistics.median(enqueue),
                      event_median_us=statistics.median(stream),
                      scope='FP16 non-causal 2D attention mechanism validation, NOT GQA gate')
        records.append(record)
        print('CASE_JSON ' + json.dumps(record), flush=True)
        (out/'results.json').write_text(json.dumps(records, indent=2)+'\n')
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out/'profile')),
    ):
        for _ in range(5):
            call()
        torch.npu.synchronize()
    print('VALIDATED', len(records), 'geometries, 3 poisoned-output checks each', flush=True)


if __name__ == '__main__':
    main()
