#!/usr/bin/env python3
"""Validate native strided GQA head mapping, causal VF, and BF16/D256 Cube.

These are exact mechanism probes, NOT full GQA implementations or gate runs.
"""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--device', type=int, default=0)
    args = p.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)
    import torch
    import torch_npu
    import pro_capability_kernels as kernels
    torch.npu.set_device(args.device)
    device = f'npu:{args.device}'
    torch.set_num_threads(4)
    results = []

    def check(name, fn, expected, y, atol=0, rtol=0):
        t = time.perf_counter()
        errors = []
        previous = None
        for _ in range(3):
            y.fill_(float('nan'))
            fn()
            torch.npu.synchronize()
            got = y.cpu()
            torch.testing.assert_close(got, expected, atol=atol, rtol=rtol)
            if previous is not None:
                assert torch.equal(got, previous), 'non-deterministic capability output'
            previous = got.clone()
            errors.append(float((got-expected).abs().max()))
        row = dict(name=name, passed=True, max_abs_samples=errors,
                   finite=bool(torch.isfinite(got).all()),
                   wall_s=time.perf_counter()-t, atol=atol, rtol=rtol)
        results.append(row)
        print('CAPABILITY_JSON '+json.dumps(row), flush=True)
        (out/'results.json').write_text(json.dumps(results, indent=2)+'\n')

    for batch, seq, nq, nk in [(1, 16, 4, 1), (2, 17, 12, 3),
                                (2, 33, 12, 4), (1, 1, 8, 8)]:
        torch.manual_seed(42)
        q = torch.randn(batch, seq, nq, 64)
        k = torch.randn(batch, seq, nk, 64)
        expected = q + k.repeat_interleave(nq//nk, dim=2)
        q, k = q.to(device), k.to(device)
        y = torch.empty_like(q)
        check(f'native_BSND_B{batch}_S{seq}_G{nq//nk}',
              lambda: kernels.strided_group_add[None, 8](q, k, y), expected, y)
    for sq in (1, 17, 64):
        torch.manual_seed(42)
        x = torch.randn(sq, 64)
        i = torch.arange(sq)[:, None]
        j = torch.arange(64)[None, :]
        expected = x.masked_fill(j > i + 64 - sq, -1e30)
        x = x.to(device)
        y = torch.empty_like(x)
        check(f'right_down_causal_Sq{sq}_Sk64',
              lambda: kernels.causal_mask[None, 8](x, y), expected, y)
    for m, n in [(17, 33), (64, 64), (96, 80)]:
        torch.manual_seed(42)
        a = torch.randn(m, 256).to(torch.bfloat16)
        b = torch.randn(256, n).to(torch.bfloat16)
        expected = (a.double()@b.double()).float()
        a, b = a.to(device), b.to(device)
        y = torch.empty((m, n), dtype=torch.float32, device=device)
        check(f'BF16_Cube_{m}x256x{n}',
              lambda: kernels.bf16_matmul_d256[None, 4](a, b, y), expected, y,
              atol=1e-3, rtol=1e-3)
    print('VALIDATED', len(results), 'mechanisms, 3 poisoned-output checks each', flush=True)


if __name__ == '__main__':
    main()
