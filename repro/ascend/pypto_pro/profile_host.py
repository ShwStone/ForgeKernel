#!/usr/bin/env python3
"""Attribute Pro eager launch overhead; never changes installed package files.

The optional diagnostic caches static exception-dump command construction,
keeps per-call tensor/ABI dump metadata enabled, and restores the original
function afterwards. It is not a production cache policy or backend patch.
"""
import argparse
import cProfile
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import pstats
import statistics
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--diagnostic-cache', action='store_true')
    args = parser.parse_args()
    source, out = Path(args.source).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device(args.device)
    device = f'npu:{args.device}'
    spec = importlib.util.spec_from_file_location('official_fa', source)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    torch.manual_seed(42)
    q, k, v = [torch.randn(64, 128, device=device, dtype=torch.float16) for _ in range(3)]
    y = torch.empty_like(q)
    launch = mod.fa_perf_tkv_preload_dn_kernel[None, 1]
    fn = lambda: launch(q, k, v, y)
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(50):
        fn()
    profiler.disable()
    torch.npu.synchronize()
    text = io.StringIO()
    pstats.Stats(profiler, stream=text).strip_dirs().sort_stats('cumtime').print_stats(25)
    (out / 'cprofile.txt').write_text(text.getvalue())

    def measure():
        values = []
        for _ in range(5):
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            for _ in range(16):
                fn()
            values.append((time.perf_counter_ns()-start)/16000)
            torch.npu.synchronize()
        return dict(samples_us=values, median_us=statistics.median(values))

    result = dict(original=measure(),
                  compiled_instances=len(mod.fa_perf_tkv_preload_dn_kernel._compiled_by_signature),
                  scope='host diagnostic; private API introspection is not a backend contract')
    if args.diagnostic_cache:
        dump = importlib.import_module('pypto_pro.runtime.exception_dump')
        original_builder = dump._build_debug_compile_cmd
        commands = {}

        def cached_builder(compiled):
            key = id(compiled)
            if key not in commands:
                commands[key] = original_builder(compiled)
            return commands[key]

        try:
            dump._build_debug_compile_cmd = cached_builder
            fn()
            result['cache_static_command_only'] = measure()
        finally:
            dump._build_debug_compile_cmd = original_builder
        result['restored'] = measure()
    (out / 'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(text.getvalue())
    print('HOST_JSON '+json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
