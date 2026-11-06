#!/usr/bin/env python3
"""Validate the official four-stage Cube/Vector pipeline against CPU FP64."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', type=int, default=0)
    args = p.parse_args()
    source = Path(args.source).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)
    import torch
    import torch_npu
    torch.npu.set_device(args.device)
    device = f'npu:{args.device}'
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location('pipeline_probe', source)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    torch.manual_seed(42)
    a = torch.randn(256, 64)
    b, d = torch.randn(64, 64), torch.randn(64, 64)
    expected = (torch.relu(a.double()@b.double())@d.double()).float()
    a, b, d = [t.to(device) for t in (a, b, d)]
    y = torch.empty(256, 64, device=device)
    kernel = mod.pipeline_demo_kernel[None, 1]
    errors = []
    previous = None
    started = time.perf_counter()
    for _ in range(3):
        y.fill_(float('nan'))
        kernel(a, b, d, y)
        torch.npu.synchronize()
        got = y.cpu()
        torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-2)
        if previous is not None:
            assert torch.equal(got, previous), 'non-deterministic pipeline output'
        previous = got.clone()
        errors.append(float((got-expected).abs().max()))
    result = dict(source=source.name, passed=True, max_abs_samples=errors,
                  wall_s=time.perf_counter()-started,
                  finite=bool(torch.isfinite(got).all()),
                  scope='four-stage relu(A@B)@D pipeline, not GQA gate')
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out/'profile')),
    ):
        for _ in range(5):
            kernel(a, b, d, y)
        torch.npu.synchronize()
    result['pipeline_generated'] = [str(p.relative_to(out)) for p in out.rglob('pipeline_generated.py')]
    (out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print('PIPELINE_JSON '+json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
