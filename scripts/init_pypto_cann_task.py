#!/usr/bin/env python3
"""Build a PyPTO or experimental Triton CANN-Bench task with a sealed reference.

The initial GQA route is explicit; no fused library call is admitted as a student
implementation. Its kernel must be authored from the curriculum in PyPTO.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
import yaml

from forgekernel import arch as arch_mod
from forgekernel import profiling
from forgekernel.paths import repo_root
from forgekernel.task import load_task


def put(root: Path, rel: str, text: str, executable=False):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    if executable:
        p.chmod(0o755)


def _agent_toml(args) -> str:
    """Only write role settings the user explicitly chose."""
    sections = []
    common = []
    if args.provider:
        common.append(f'provider = {json.dumps(args.provider)}')
    if args.student_model:
        common.append(f'model = {json.dumps(args.student_model)}')
    if args.student_effort:
        common.append(f'effort = {json.dumps(args.student_effort)}')
    if common:
        sections.append('[agent]\n' + '\n'.join(common))
    teacher = []
    if args.teacher_provider:
        teacher.append(f'provider = {json.dumps(args.teacher_provider)}')
    if args.teacher_model:
        teacher.append(f'model = {json.dumps(args.teacher_model)}')
    if args.teacher_effort:
        teacher.append(f'effort = {json.dumps(args.teacher_effort)}')
    if teacher:
        sections.append('[agent.teacher]\n' + '\n'.join(teacher))
    return '\n'.join(sections) + ('\n' if sections else '')


def problem(cases, task_dir, operator, entry, commit,
            reference_name='Huawei CANN fused-attention library'):
    rows = []
    for i, case in enumerate(cases):
        case_id = str(case.get('case_id', i + 1))
        name = f'{task_dir.removeprefix("tasks/")}_{case_id}'
        workload = {'uuid': name, 'axes': {'case_id': case_id,
                    'input_shape': case.get('input_shape'), 'dtype': case.get('dtype'),
                    'attrs': case.get('attrs', {})}, 'input_groups': []}
        rows.append({'index': i, 'name': name, 'workload': workload})
    return {'schema': 1, 'suite': 'cann-bench', 'definition': {
        'name': operator, 'axes': {}, 'inputs': {'tensor': {'dtype': 'tensor'}},
        'outputs': {'y': {'dtype': 'tensor'}}, 'reference': 'CANN-Bench golden.py'},
        'workload': rows[0]['workload'], 'group': rows,
        'meta_workloads': [r['name'] for r in rows], 'dps': False,
        'kernel_file': 'kernel.py', 'entry': entry,
        'reference_impl': {'kind': 'separate_library_reference',
                           'name': reference_name + ', separate tree'},
        'time_reference': False, 'timing': 'device',
        'tolerance': {'outputs': {}, 'rule': 'official CANN checker, no local hidden set'},
        'source': {'task': task_dir, 'commit': commit}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('task', type=Path)
    p.add_argument('--language', choices=('pypto', 'triton'), default='pypto')
    p.add_argument('--signature', default='query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, '
                   'scaleValue: float = -1.0, is_causal: bool = False')
    p.add_argument('--reference-file', type=Path,
                   help='self-contained Python reference defining --entry; default: CANN GQA library')
    p.add_argument('--reference-name', default='')
    p.add_argument('--scout-case-ids', default='', help='comma-separated official case numbers')
    p.add_argument('--autotune-max-configs', type=int, default=0,
                   help='Triton only: 0 disables autotune, 1..8 bounds explicit Configs')
    p.add_argument('--case-timeout', type=int, default=180,
                   help='Triton only: wall budget for one isolated official case, including JIT')
    p.add_argument('--mutates-inputs', action='store_true',
                   help='Triton autotune requires reset/restore hooks for input mutation')
    p.add_argument('--cann-bench-dir', type=Path, required=True)
    p.add_argument('--level', default='tasks/level4',
                   help='task root relative to the CANN-Bench worktree (default: tasks/level4)')
    p.add_argument('--operator-path', default='gqa')
    p.add_argument('--operator', default='GQA')
    p.add_argument('--entry', default='gqa')
    p.add_argument('--arch', default='ascend950pr',
                   help='target Ascend architecture, e.g. ascend910b, ascend910c, ascend950pr, ascend950dt')
    p.add_argument('--hardware', default='',
                   help='free-form hardware label (default: derived from --arch)')
    p.add_argument('--expected-cann-ref', default='',
                   help='pin the CANN-Bench commit; default: the worktree HEAD')
    p.add_argument('--forgekernel-root', type=Path, default=repo_root())
    p.add_argument('--target-ratio', type=float, default=1.05)
    p.add_argument('--hours', type=float, default=16)
    p.add_argument('--max-meta', type=int, default=4)
    p.add_argument('--oracle', choices=('blackbox', 'source'), default='blackbox')
    p.add_argument('--student-model', default='')
    p.add_argument('--provider', default='')
    p.add_argument('--student-effort', default='')
    p.add_argument('--teacher-model', default='')
    p.add_argument('--teacher-provider', default='')
    p.add_argument('--teacher-effort', default='')
    p.add_argument('--pypto-source', default='')
    p.add_argument('--pto-isa', default='')
    p.add_argument('--dsl-source-dir', action='append', default=[],
                   help='additional local source directory for the offline student (repeatable)')
    args = p.parse_args(argv)
    root = args.forgekernel_root.resolve()
    dest = args.task.resolve()
    if dest.exists():
        p.error('task already exists')
    cann = args.cann_bench_dir.resolve()
    commit = subprocess.run(['git', '-C', str(cann), 'rev-parse', 'HEAD'],
                            check=True, capture_output=True, text=True).stdout.strip()
    expected_ref = args.expected_cann_ref or commit
    if commit != expected_ref:
        p.error(f'CANN-Bench HEAD is {commit}, expected {expected_ref}')
    args.arch = arch_mod.canonical_arch(args.arch)
    if not arch_mod.is_ascend_arch(args.arch):
        p.error(f'not an Ascend architecture: {args.arch}')
    if args.language == 'triton' and args.arch != 'ascend950pr':
        p.error('experimental Triton requires independently certified ascend950pr hardware')
    if not 0 <= args.autotune_max_configs <= 8:
        p.error('--autotune-max-configs must be 0..8')
    if args.case_timeout <= 0:
        p.error('--case-timeout must be positive')
    try:
        declaration = ast.parse(f'def {args.entry}({args.signature}):\n    pass\n').body[0]
    except SyntaxError as exc:
        p.error(f'invalid entry/signature: {exc}')
    label = 'Triton' if args.language == 'triton' else 'PyPTO'
    reference_name = args.reference_name or ('custom callable reference' if args.reference_file
                                            else 'Huawei CANN fused-infer-attention library')
    hardware = args.hardware or arch_mod.hardware_label(args.arch)
    level = args.level.strip('/')
    if not level.startswith('tasks/'):
        level = f'tasks/{level}'
    task_dir = f'{level}/{args.operator_path}'
    cases = yaml.safe_load((cann / task_dir / 'cases.yaml').read_text())['cases']
    card, ref = dest / 'leaderboard/card', dest / 'leaderboard/ref'
    template = root / 'src/forgekernel/card_template'
    for tree in (card, ref):
        tree.mkdir(parents=True)
        (tree / 'eval').mkdir()
        (tree / 'cann_bench').mkdir()
        for source, target in [('eval/cann_bench.py', 'eval/cann_bench.py'),
                               ('eval/ascend_profiler.py', 'eval/ascend_profiler.py'),
                               ('pypto_gpu_entry.sh', 'gpu_entry.sh')]:
            shutil.copy2(template / source, tree / target)
        put(tree, '.gitattributes', (template / '.gitattributes').read_text())
        put(tree, '.gitignore', 'loop.toml\nsubmit.local.toml\n.jobs/\n.exec/\n.cann_results/\n'
            'build/\ndist/\n*.egg-info/\n__pycache__/\n*.pyc\ncann_bench/kernel_impl.py\n')
        put(tree, 'setup.py', 'from setuptools import setup\n'
            'setup(name="cann_bench", version="1.0.0", packages=["cann_bench"])\n')
        put(tree, 'build.sh', '#!/usr/bin/env bash\nset -euo pipefail\ncd "$(dirname "$0")"\n'
            'rm -rf build dist cann_bench.egg-info cann_bench/kernel_impl.py\n'
            'cp kernel.py cann_bench/kernel_impl.py\npython setup.py bdist_wheel\n', True)
        if args.language == 'triton':
            for name in ('triton_runtime.py', 'triton_timing.py', 'ascend_process.py'):
                shutil.copy2(template / 'eval' / name, tree / 'eval' / name)
            shutil.copy2(template / 'eval/triton_runtime.py', tree / 'cann_bench/_forge_triton.py')
            put(tree, 'cann_bench/__init__.py',
                'from ._forge_triton import install as _install, track_entry as _track_entry\n_install()\n'
                f'from .kernel_impl import {args.entry}\n{args.entry} = _track_entry({args.entry})\n')
        else:
            put(tree, 'cann_bench/__init__.py', f'from .kernel_impl import {args.entry}\n')
    # One declaration, both trees: a candidate and a reference timed in different
    # windows produce a ratio between incomparable numbers. Validated rather than
    # merely written, so a later hand-edit of either file is caught at launch too.
    cann_task = {
        'operator': args.operator, 'task_dir': task_dir,
        'warmup': 3, 'repeat': 5, 'profiler_backend': 'msprof',
        'perf_metric_strategy': 'kernel_details',
        'timing_basis': 'cann_kernel_details',
        'expected_cann_ref': expected_ref}
    if args.language == 'triton':
        cann_task.update(language='triton', arch=args.arch, perf_metric_strategy='kernel_details',
                         timing_basis='cann_logical_call', autotune_max_configs=args.autotune_max_configs,
                         mutates_inputs=args.mutates_inputs, case_timeout_s=args.case_timeout)
        declared = [f'{task_dir.removeprefix("tasks/")}_{c.get("case_id", i + 1)}'
                    for i, c in enumerate(cases)]
        cann_task['cases'] = declared
        if args.scout_case_ids:
            ids = args.scout_case_ids.split(',')
            scout = [f'{task_dir.removeprefix("tasks/")}_{n.strip()}' for n in ids]
            if len(set(scout)) != len(scout) or not set(scout) <= set(declared):
                p.error('--scout-case-ids must name unique official case numbers')
        else:
            scout = list(dict.fromkeys(declared[i] for i in (0, len(cases) // 2, len(cases) - 1)))
        cann_task['scout_cases'] = scout
    if problems := profiling.timing_basis_problems(cann_task):
        p.error('; '.join(problems))
    if why := profiling.card_backend_problem(
            cann_task, profiling.backend_for(args.language, cann_task['profiler_backend'])):
        p.error(why)
    put(card, 'eval/cann_task.json', json.dumps({**cann_task, 'source_dir': '.',
        'evaluation_role': 'candidate'}, indent=2))
    put(ref, 'eval/cann_task.json', json.dumps({**cann_task, 'source_dir': 'src',
        'evaluation_role': 'reference'}, indent=2))
    put(card, 'eval/leaderboard/problem.json', json.dumps(problem(cases, task_dir,
        args.operator, args.entry, commit, reference_name), indent=2))
    signature = args.signature
    imports = 'import triton\nimport triton.language as tl' if args.language == 'triton' else 'import pypto'
    put(card, 'kernel.py', f'import torch\n{imports}\n\n'
        f'def {args.entry}({signature}) -> torch.Tensor:\n'
        f'    raise NotImplementedError("implement {label} from the milestone")\n')
    library = (args.reference_file.read_text() if args.reference_file else
               (cann / 'scripts/baseline/refs/level4.py').read_text())
    (ref / 'src').mkdir()
    for name in ('build.sh', 'setup.py', 'cann_bench'):
        p = card / name
        if p.is_dir():
            shutil.copytree(p, ref / 'src' / name)
        else:
            shutil.copy2(p, ref / 'src' / name)
    reference = library if args.reference_file else library + f'\n\ndef {args.entry}({signature}):\n' \
        '    return gqa_ref([query, key, value],\n' \
        '        {"scaleValue": scaleValue, "is_causal": is_causal})\n'
    put(ref, 'src/kernel.py', reference)
    for tree in (card, ref):
        put(tree, 'window.toml', 'schema_version = 1\n[reference]\nwindow = '
            '"Huawei fused-attention kernel_details sum, one call per sample"\n'
            'inputs = ["query", "key", "value"]\noutputs = ["y"]\n[student]\n'
            'window = "candidate, all kernels in kernel_details; trace_view AI-Core E2E for PyPTO"\n'
            'inputs = ["query", "key", "value"]\noutputs = ["y"]\n')
    if args.language == 'triton':
        for tree in (card, ref):
            inputs = json.dumps([arg.arg for arg in declaration.args.posonlyargs
                                  + declaration.args.args + declaration.args.kwonlyargs])
            put(tree, 'window.toml', 'schema_version = 1\n[reference]\n'
                'window = "complete logical call, NPU first-task-start to last-task-end window"\n'
                f'inputs = {inputs}\noutputs = ["y"]\n'
                '[student]\nwindow = "same complete logical call; not kernel duration sum"\n'
                f'inputs = {inputs}\noutputs = ["y"]\n')
    put(card, 'loop.toml.example', f'''[loop]
language = "{args.language}"
kb_arch = "{args.arch}"
gate_window = "device"
oracle_mode = "{args.oracle}"
recon_op = ""
ref_mode = "ref"
ref_name = {json.dumps(reference_name + ' (exact version in report)')}
ref_shape = "{len(cases)} public {task_dir} cases"
ref_hardware = {json.dumps(hardware)}
ref_from_goal = "off"
ref_price_op = ""
precision_contract = "off"
correctness_monitor = "off"
confirm_jobs = 3
eval_timeout_s = 7200
profiler_backend = "msprof"
ncu_available = false
job_env_preflight = "{'refuse' if args.language == 'triton' else 'warn'}"
timeline_probe = "off"
stall_pricing = false
orch_census = false
dsl_source_dirs = {json.dumps([p for p in [args.pypto_source, args.pto_isa, *args.dsl_source_dir] if p])}
''')
    put(card, 'README.md', f'''# {args.operator} — {label} on Ascend, CANN-Bench {task_dir}
Frozen public cases are judged by official golden.py, not the library reference.
`kernel.py` is student-owned and starts from a signature stub. No vendor operator
call is permitted as its implementation. The separate strong library tree is sealed
in blackbox mode. Public local scoring is not the paper's 910C / CANN 1.1.2 hosted result.
The library uses kernel_details kernel-only time. PyPTO records both that and trace_view
AI-Core E2E. Reference comparisons are cross-job median-vs-median, not paired timing.
''')
    if args.language == 'triton':
        put(card, 'README.md', f'''# {args.operator} — experimental Triton on {hardware}
Official CANN-Bench {task_dir} cases, golden.py and checker are frozen at {commit}.
The student starts from a signature-only stub. Implement with @triton.jit; Torch
is host allocation/plumbing only. No vendor fused operator is admitted.
Both sides use the explicit cann_logical_call window (including gaps),
not a sum/filter of device tasks. Separate-job reference ratios are not paired.
Reference: {reference_name}. No PyPTO/PTO Tile dependency. Candidate-only compile
IR/metadata, cache hits, first-call and host costs are diagnostics, not gate time.
Autotune permits {args.autotune_max_configs} explicit Configs (0 means disabled);
automatic tiling is forbidden. The evaluator and its new-session descendants
share a bounded job budget. Scout jobs never clear a gate. Other targets and
hosted hidden-set scores are not certified by this task.
''')
    shutil.copy2(cann / task_dir / 'desc.md', dest / 'description.md')
    put(dest, 'shapes.json', '[]\n')
    put(dest, 'task.toml', f'''[task]
language = "{args.language}"
arch = "{args.arch}"
[leaderboard]
suite = "cann-bench"
card = "leaderboard/card"
ref = "leaderboard/ref"
[target]
ratio = {args.target_ratio}
timing = "device"
{_agent_toml(args)}[budget]
hours = {args.hours}
max_meta = {args.max_meta}
''')
    put(dest, 'experiment_protocol.json', json.dumps({
        'level': args.level.rsplit('/', 1)[-1], 'operator': args.operator, 'cases': 'public', 'judge': 'official golden.py',
        'architecture_reference': reference_name,
        'student': f'{label} authored kernel; vendor fused-operator calls forbidden',
        'oracle_mode': args.oracle, 'cann_commit': commit,
        'paper_equivalence': 'same operator family, NOT paper hardware/benchmark hosted reproduction',
    }, indent=2))
    load_task(dest, kb_root=root)
    print(dest)


if __name__ == '__main__':
    main()
