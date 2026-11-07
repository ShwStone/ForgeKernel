#!/usr/bin/env python3
"""Numerical diagnostics, explicitly not a full official evaluator gate."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import time

from probe_utils import environment, fingerprint, install_compile_listener, profile, save


SYNTHETIC = [
    # B, Sq, Sk, Nq, Nkv, D, dtype, causal, scale, noncontiguous
    (1, 17, 33, 8, 2, 128, 'float16', True, -1., False),
    (2, 34, 62, 36, 3, 128, 'bfloat16', False, -1., False),
    (1, 1, 257, 32, 8, 128, 'float16', False, -1., False),
    (1, 2, 257, 32, 8, 256, 'bfloat16', True, -1., False),
    (1, 33, 65, 32, 1, 128, 'float16', False, -1., False),
    (2, 17, 33, 8, 8, 128, 'bfloat16', True, -1., False),
    (1, 34, 62, 8, 2, 256, 'float16', True, 0.08838, True),
    (1, 17, 65, 8, 2, 128, 'bfloat16', False, 0.08838, True),
]
SEEDS = (20261106, 20261107, 20261108)
CANN_COMMIT = '059cb61d41b279ad021160517bd2a5f6b7a05e01'


def oracle(q, k, v, scale, causal):
    import torch
    # Expansion is CPU oracle work only, never part of the NPU candidate.
    g = q.shape[2]//k.shape[2]
    q, k, v = [x.double().permute(0, 2, 1, 3) for x in (q, k, v)]
    k, v = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
    score = q@k.transpose(-1, -2)*(scale if scale > 0 else 1/math.sqrt(q.shape[-1]))
    if causal:
        sq, sk = score.shape[-2:]
        keep = torch.arange(sk)[None, :] <= torch.arange(sq)[:, None]+sk-sq
        score = score.masked_fill(~keep, -torch.inf)
    return (score.softmax(-1)@v).permute(0, 2, 1, 3).contiguous()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--cache', help='explicit compilation-cache directory; default is under --out')
    p.add_argument('--cann-bench', help='optional pinned checkout; use its golden, native reference and checker')
    p.add_argument('--case-ids', default='1,2,5,6,18,20')
    p.add_argument('--tile-rows', type=int, default=16)
    p.add_argument('--tile-n', type=int, default=32)
    args = p.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ['TRITON_CACHE_DIR'] = str(Path(args.cache).resolve() if args.cache else out/'cache')
    os.chdir(out)
    import torch
    import torch_npu  # noqa: F401
    import triton
    from kernels import launch_gqa
    torch.npu.set_device(args.device)
    torch.set_num_threads(4)
    save(out/'environment.json', environment())
    compilations, old_listener = install_compile_listener(out/'compiler-artifacts')
    records = []
    cases = []
    for i, (b, sq, sk, nq, nk, d, dtype, causal, scale, strided) in enumerate(SYNTHETIC):
        cases.append(dict(name=f'synthetic-{i+1}', shapes=[[b, sq, nq, d], [b, sk, nk, d], [b, sk, nk, d]],
                          dtype=dtype, causal=causal, scale=scale, strided=strided))
    golden = checker = thresholds = None
    if args.cann_bench:
        root = Path(args.cann_bench).resolve()
        actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != CANN_COMMIT:
            raise RuntimeError(f'expected pinned CANN-Bench {CANN_COMMIT}, got {actual}')
        import yaml
        from kernel_eval.checkers import relative_error_checker as checker_module
        from kernel_eval.utils import thresholds as threshold_module
        RelativeErrorChecker = checker_module.RelativeErrorChecker
        PRECISION_THRESHOLDS = threshold_module.PRECISION_THRESHOLDS
        for module in (checker_module, threshold_module):
            if not Path(module.__file__).resolve().is_relative_to(root):
                raise RuntimeError(f'checker imported outside pinned checkout: {module.__file__}')
        save(out/'official-contract.json', dict(
            commit=actual, requested_case_ids=args.case_ids,
            files=[fingerprint(f) for f in (root/'tasks/level4/gqa/golden.py',
                   root/'tasks/level4/gqa/cases.yaml', checker_module.__file__, threshold_module.__file__)]))
        spec = importlib.util.spec_from_file_location('official_gqa_golden', root/'tasks/level4/gqa/golden.py')
        golden = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(golden)
        checker, thresholds = RelativeErrorChecker(), PRECISION_THRESHOLDS
        selected = {int(i) for i in args.case_ids.split(',')}
        all_cases = yaml.safe_load((root/'tasks/level4/gqa/cases.yaml').read_text())['cases']
        found = {c['case_id'] for c in all_cases if c['case_id'] in selected}
        if selected != found:
            raise ValueError('unknown official case selector')
        for c in all_cases:
            if c['case_id'] in selected:
                cases.append(dict(name=f'official-{c["case_id"]}', shapes=c['input_shape'],
                                  dtype=c['dtype'][0], causal=c['attrs']['is_causal'],
                                  scale=c['attrs']['scaleValue'], strided=False))
    try:
        for case in cases:
            for seed in SEEDS:
                torch.manual_seed(seed)
                dtype = getattr(torch, case['dtype'])
                cpu = [(torch.rand(s)*2-1).to(dtype) for s in case['shapes']]
                expected = native = None
                if case['name'].startswith('official-'):
                    attrs = dict(scaleValue=case['scale'], is_causal=case['causal'])
                    expected = golden.gqa(*(x.double() for x in cpu), **attrs)
                    native = golden.gqa(*cpu, **attrs)
                else:
                    expected = oracle(*cpu, case['scale'], case['causal'])
                tensors = []
                for x in cpu:
                    x = x.to(f'npu:{args.device}')
                    if case['strided']:
                        backing = torch.empty((*x.shape[:-1], x.shape[-1]*2), device=x.device, dtype=x.dtype)
                        view = backing[..., ::2]
                        view.copy_(x)
                        x = view
                    tensors.append(x)
                y = torch.empty(case['shapes'][0], device=tensors[0].device, dtype=dtype)
                fn = lambda: launch_gqa(*tensors, case['scale'], case['causal'], out=y,
                                        tile_rows=args.tile_rows, tile_n=args.tile_n)
                started = time.perf_counter()
                errors, verdicts, previous = [], [], None
                for _ in range(3):
                    y.fill_(float('nan'))
                    _, compiled = fn()
                    torch.npu.synchronize()
                    got = y.cpu()
                    assert bool(torch.isfinite(got).all()), 'non-finite output'
                    if native is not None:
                        verdict = checker.check(got, expected, case['dtype'], thresholds[case['dtype']],
                                                native_outputs=native)
                        verdicts.append(verdict.to_dict())
                        assert verdict.passed, str(verdict.to_dict())
                    else:
                        tolerance = 5e-3 if dtype == torch.float16 else 2e-2
                        torch.testing.assert_close(got.double(), expected, rtol=tolerance, atol=tolerance)
                    if previous is not None:
                        assert torch.equal(got, previous), 'non-deterministic output'
                    previous = got.clone()
                    errors.append(float((got.double()-expected.double()).abs().max()))
                record = dict(case=case, seed=seed, tile_rows=args.tile_rows, tile_n=args.tile_n,
                              passed=True, max_abs_samples=errors,
                              finite=True, deterministic_output=True, compiled_hash=compiled.hash,
                              first_three_calls_including_compile_s=time.perf_counter()-started)
                if verdicts:
                    record['official_checker'] = verdicts
                records.append(record)
                save(out/'results.json', records)
                print('GQA_JSON '+json.dumps(record), flush=True)
            if case['name'] in ('synthetic-4', 'official-6'):
                save(out/f'profile-{case["name"]}.json', profile(fn, out/f'profile-{case["name"]}'))
    except Exception as exc:
        save(out/'failure.json', dict(case=case, error=repr(exc), completed_checks=len(records)))
        raise
    finally:
        triton.knobs.compilation.listener = old_listener
    print(f'VALIDATED {len(records)} inputs, {len(records)*3} poisoned-output checks; NOT a full evaluator gate', flush=True)


if __name__ == '__main__':
    main()
