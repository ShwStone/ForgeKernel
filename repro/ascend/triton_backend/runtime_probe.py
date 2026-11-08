#!/usr/bin/env python3
"""Observe cache, grid mapping, eager/graph and bounded autotune on an NPU."""
import argparse
import json
import math
import os
from pathlib import Path
import time

from probe_utils import environment, host_samples, install_compile_listener, profile, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--cache', help='explicit compilation-cache directory; default is under --out')
    parser.add_argument('--block', type=int, default=1024)
    parser.add_argument('--mode', default='simd_simt_template', choices=['simd', 'simd_simt_template', 'simt_only'])
    parser.add_argument('--advanced', action='store_true')
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ['TRITON_CACHE_DIR'] = str(Path(args.cache).resolve() if args.cache else out/'cache')
    os.chdir(out)
    import torch
    import torch_npu  # noqa: F401
    import triton
    from kernels import add_probe
    torch.npu.set_device(args.device)
    save(out/'environment.json', environment())
    records, previous_listener = install_compile_listener(out/'compiler-artifacts')
    result = dict(block=args.block, requested_mode=args.mode)
    try:
        torch.manual_seed(20261106)
        x_cpu = torch.rand(28*1024)
        x = x_cpu.to(f'npu:{args.device}')
        y = torch.empty_like(x)
        fn = lambda: add_probe[(triton.cdiv(x.numel(), args.block),)](
            x, y, x.numel(), args.block, compile_mode=args.mode)
        started = time.perf_counter()
        y.fill_(float('nan'))
        compiled = fn()
        torch.npu.synchronize()
        result['first_call_compile_launch_sync_s'] = time.perf_counter()-started
        torch.testing.assert_close(y.cpu(), x_cpu+1, rtol=0, atol=0)
        result['output_exact'] = True
        result['compiled_hash'] = compiled.hash
        result['metadata'] = compiled.metadata._asdict()
        result['eager'] = host_samples(fn)
        result['eager']['profile'] = profile(fn, out/'eager-profile')
        if args.advanced:
            grid_checks = []
            for blocks in (73, 70001):
                n = blocks*32-7
                a_cpu = torch.rand(n)
                a = a_cpu.to(x.device)
                z = torch.full_like(a, float('nan'))
                kernel = add_probe[(blocks,)](a, z, n, 32, compile_mode=args.mode)
                torch.npu.synchronize()
                torch.testing.assert_close(z.cpu(), a_cpu+1, rtol=0, atol=0)
                grid_profile = profile(lambda: add_probe[(blocks,)](a, z, n, 32, compile_mode=args.mode),
                                       out/f'grid-{blocks}-profile')
                grid_checks.append(dict(logical_blocks=blocks, exact=True,
                                        metadata=kernel.metadata._asdict(), profile=grid_profile))
            result['grid_checks'] = grid_checks
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                fn()
            graph_checks = []
            for value in (3., -2.):
                x.fill_(value)
                y.fill_(float('nan'))
                graph.replay()
                torch.npu.synchronize()
                torch.testing.assert_close(y.cpu(), torch.full_like(x_cpu, value+1), rtol=0, atol=0)
                graph_checks.append(dict(input_value=value, exact=True))
            result['graph'] = host_samples(graph.replay)
            result['graph']['changed_input_checks'] = graph_checks
            result['graph']['profile'] = profile(graph.replay, out/'graph-profile')
            from triton.backends.ascend.testing import do_bench_npu
            bench = {}
            for clear in (False, True):
                value = do_bench_npu(fn, warmup=1, active=5, clear_l2_cache=clear,
                                     prof_dir=str(out/f'bench-clear{int(clear)}'), keep_res=True,
                                     target_kernel_name=result['eager']['profile']['name'])
                assert math.isfinite(value) and value > 0, 'unavailable profiler measurement'
                bench[str(clear)] = dict(mean_device_ms=value, warmup=1, active=5,
                                         clear_l2_cache=clear, method='forced torch_npu profiler, exact kernel name')
            result['do_bench_npu'] = bench
            tuning_calls = []

            def bounded_bench(call, quantiles):
                value = do_bench_npu(call, warmup=1, active=5)
                assert math.isfinite(value) and value > 0
                tuning_calls.append(value)
                return value  # API accepts a scalar; deliberately device mean, not quantiles.

            tuner = triton.autotune(
                configs=[triton.Config({'BLOCK': b}, num_warps=32, num_stages=2) for b in (256, 1024)],
                key=['N'], do_bench=bounded_bench)(add_probe)
            grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK']),)
            for _ in range(2):
                y.fill_(float('nan'))
                tuner[grid](x, y, x.numel())
                torch.npu.synchronize()
                torch.testing.assert_close(y.cpu(), x.cpu()+1, rtol=0, atol=0)
            assert len(tuning_calls) == 2, 'expected two candidates, with in-process selection reuse'
            result['bounded_autotune'] = dict(candidate_count=2, benchmark_calls=len(tuning_calls),
                                              mean_device_ms=tuning_calls, selected=tuner.best_config.kwargs,
                                              repeated_call_exact=True)
        result['compilations'] = records
        result['scope'] = 'independent diagnostics; not a ForgeKernel task or official gate'
        save(out/'results.json', result)
        print('RUNTIME_JSON '+json.dumps(result, default=str), flush=True)
    except Exception as exc:
        save(out/'failure.json', dict(error=repr(exc), partial=result, compilations=records))
        raise
    finally:
        triton.knobs.compilation.listener = previous_listener


if __name__ == '__main__':
    main()
