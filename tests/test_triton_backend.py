"""CPU contracts for the independent, Ascend-only Triton execution path."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import yaml

from forgekernel import launch, profiling, taskcard
from forgekernel.localenv import Machine
from forgekernel.loop import cli, driver, lang_mode
from forgekernel.paths import repo_root, template_dir
from forgekernel.task import TaskError, load_task
from test_pypto import _adapter, _report
from test_pypto_task_builder import builder


def runtime():
    spec = importlib.util.spec_from_file_location('triton_runtime', template_dir() / 'eval/triton_runtime.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def triton_task(tmp_path):
    cann = tmp_path / 'cann'
    op = cann / 'tasks/level1/exp'
    op.mkdir(parents=True)
    (op / 'cases.yaml').write_text(yaml.safe_dump({'cases': [
        {'case_id': i, 'input_shape': [[1024 * i]], 'dtype': ['float32']} for i in range(1, 4)]}))
    (op / 'desc.md').write_text('exp(x), preserving dtype and shape')
    reference = tmp_path / 'reference.py'
    reference.write_text('import torch\ndef exp(x: torch.Tensor):\n    return torch.exp(x)\n')
    subprocess.run(['git', 'init', '-q', str(cann)], check=True)
    subprocess.run(['git', '-C', str(cann), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(cann), '-c', 'user.name=test', '-c',
                    'user.email=test@local', 'commit', '-qm', 'fixture'], check=True)
    dest = tmp_path / 'tasks/exp-triton'
    builder().main([str(dest), '--language', 'triton', '--cann-bench-dir', str(cann),
        '--level', 'level1', '--operator-path', 'exp', '--operator', 'Exp', '--entry', 'exp',
        '--signature', 'x: torch.Tensor', '--reference-file', str(reference),
        '--scout-case-ids', '1,3'])
    return load_task(dest)


def test_triton_has_independent_prompts_kb_provenance_and_no_cute_fallback(tmp_path):
    assert lang_mode.mode_of('triton') == lang_mode.MODE_TRITON
    assert lang_mode.is_ascend('triton') and not lang_mode.is_pypto('triton')
    assert lang_mode.kb_root_name('triton') == 'hw_kb/triton'
    assert cli._resolve_kb_arch(None, {}, 'triton') == 'ascend950pr'
    assert lang_mode.provenance_dir(tmp_path, 'triton') == tmp_path / 'docs/kb-provenance/triton'
    for mode in ('cutedsl', 'cuda-sol', 'pypto', 'triton'):
        assert lang_mode.owns_provenance('docs/kb-provenance/triton/test.md', mode) == (mode == 'triton')
    for name in (repo_root() / 'meta/pypto').glob('*.md'):
        path = lang_mode.prompt_path(repo_root() / 'meta', name.name, 'triton')
        assert path.parent.name == 'triton'
        assert 'PyPTO' not in path.read_text() or name.name == 'student_prompt.md'
    with pytest.raises(FileNotFoundError, match='missing Triton prompt'):
        lang_mode.prompt_path(tmp_path, 'student_prompt.md', 'triton')


def test_triton_target_profiler_cannot_fall_into_ncu():
    assert profiling.backend_for('triton') == 'msprof'
    assert profiling.backend_for('triton', 'cann_trace_view', arch='ascend950pr') == 'cann_trace_view'
    assert profiling.backend_for('other-dsl', arch='ascend950pr') == 'msprof'
    with pytest.raises(ValueError, match='NCU'):
        profiling.backend_for('triton', 'ncu')
    with pytest.raises(ValueError, match='uncertified'):
        profiling.backend_for('triton', arch='sm100')
    assert driver._ncu_profiler(SimpleNamespace(language='triton', profiler_backend='msprof'),
                                'main', correctness=True) == {}
    assert launch.probe_ncu(SimpleNamespace(language='triton'), Path('/unused')) is False


def test_generated_card_uses_one_official_window_and_roles(triton_task):
    task = triton_task
    assert task.language == task.loop_language == 'triton'
    assert task.arch == 'ascend950pr' and len(task.shapes) == 3
    assert taskcard.kernel_file(task) == 'kernel.py'
    card, ref = task.leaderboard.card, task.leaderboard.ref
    for tree, role in ((card, 'candidate'), (ref, 'reference')):
        config = json.loads((tree / 'eval/cann_task.json').read_text())
        assert config['language'] == 'triton' and config['evaluation_role'] == role
        assert config['perf_metric_strategy'] == 'kernel_details'
        assert config['timing_basis'] == 'cann_logical_call'
        assert config['autotune_max_configs'] == 0
        assert config['cases'] == ['level1/exp_1', 'level1/exp_2', 'level1/exp_3']
        assert config['scout_cases'] == ['level1/exp_1', 'level1/exp_3']
        assert not profiling.timing_basis_problems(config)
        assert (tree / 'eval/ascend_process.py').is_file()
        assert (tree / 'eval/triton_runtime.py').is_file()
    assert 'import triton' in (card / 'kernel.py').read_text()
    assert 'import pypto' not in (card / 'kernel.py').read_text()
    assert 'install()' in (card / 'cann_bench/__init__.py').read_text()
    assert (ref / 'src/cann_bench/_forge_triton.py').read_bytes() == (card / 'eval/triton_runtime.py').read_bytes()
    taskcard.write_submit(card, task, Machine('local', '1'))
    submit = (card / 'submit.toml').read_text()
    assert 'FK_DEVICE_TYPE = "npu"' in submit and 'PTO_TILE_LIB_CODE_PATH' not in submit
    assert 'CUTE_DSL_ARCH' not in submit
    assert taskcard.evaluation_timeout(card, task, 1800) == 7200
    import tomllib
    run = SimpleNamespace(operators_dir=card.parent)
    loop = taskcard.write_board_loop_toml(run, task, card, deny=[], ncu=False)
    assert tomllib.loads(loop.read_text())['loop']['job_env_preflight'] == 'refuse'
    from forgekernel.window_contract import load_window_contract
    assert load_window_contract(card).declared
    from forgekernel.cannref import seal_tree
    seal_tree(ref, 'kernel.py::exp')
    assert not (ref / 'src').exists()
    assert (ref / 'eval/closed_ref.py').is_file()


def test_other_triton_targets_and_non_cann_cards_are_rejected(triton_task):
    path = triton_task.dir / 'task.toml'
    text = path.read_text()
    for target in ('sm100', 'ascend950dt', 'ascend910b'):
        path.write_text(text.replace('ascend950pr', target))
        with pytest.raises(TaskError, match='only ascend950pr'):
            load_task(triton_task.dir)
    path.write_text(text.replace('suite = "cann-bench"', 'suite = "sol-execbench"'))
    with pytest.raises(TaskError, match='frozen cann-bench'):
        load_task(triton_task.dir)


def test_card_validation_refuses_wrong_role_and_duration_sum(triton_task):
    card = triton_task.leaderboard.card
    config_path = card / 'eval/cann_task.json'
    config = json.loads(config_path.read_text())
    assert cli._pypto_card_problem(card, {}, 'triton') is None
    config_path.write_text(json.dumps({**config, 'evaluation_role': 'reference'}))
    assert 'evaluation_role=candidate' in cli._pypto_card_problem(card, {}, 'triton')
    config_path.write_text(json.dumps({**config, 'timing_basis': 'cann_kernel_details',
                                      'perf_metric_strategy': 'kernel_details'}))
    assert 'full logical-call' in cli._pypto_card_problem(card, {}, 'triton')
    config_path.write_text(json.dumps({**config, 'cases': []}))
    assert 'frozen cases list' in cli._pypto_card_problem(card, {}, 'triton')


def test_full_job_cannot_pass_on_a_different_case_set(triton_task, monkeypatch, capsys):
    adapter = _adapter()
    mod = runtime()
    monkeypatch.setitem(sys.modules, 'triton_runtime', mod)
    monkeypatch.setattr(mod, 'prepare', lambda *a, **kw: None)
    original_run = subprocess.run
    def run(cmd, *a, **kwargs):
        if '-m' in cmd and 'pip' in cmd:
            return SimpleNamespace(returncode=0, stdout='', stderr='')
        return original_run(cmd, *a, **kwargs)
    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setenv('CANN_BENCH_DIR', str(triton_task.dir.parents[1] / 'cann'))
    monkeypatch.delenv('FK_CANN_CASES', raising=False)
    def evaluate(cmd, *, log, **kwargs):
        log.write_text('the evaluator judged a different case set\n')
        (log.parent / 'wrong_eval_1.json').write_text(json.dumps(_report()))
        return adapter.RunOutcome(0, False, '', 0.1)
    monkeypatch.setattr(adapter, '_run_logged', evaluate)
    with pytest.raises(RuntimeError, match='case-set contract mismatch'):
        adapter.evaluate(triton_task.leaderboard.card)
    assert 'cann/partial_report.json' in capsys.readouterr().out


def test_cache_keys_separate_environment_source_and_role(tmp_path):
    mod = runtime()
    fp, src = mod.digest({'compiler': 'one'}), mod.digest({'source': 'one'})
    root = tmp_path / 'cache'
    candidate = mod.cache_directory(root, fp, 'candidate', src)
    reference = mod.cache_directory(root, fp, 'reference', src)
    assert candidate != reference and candidate.is_dir() and reference.is_dir()
    assert candidate == mod.cache_directory(root, fp, 'candidate', src)
    assert candidate != mod.cache_directory(root, mod.digest({'compiler': 'patched'}), 'candidate', src)
    assert candidate != mod.cache_directory(root, fp, 'candidate', mod.digest({'source': 'changed'}))
    with pytest.raises(ValueError):
        mod.cache_directory(root, fp, '../escape', src)


def test_autotune_is_explicit_bounded_and_requires_mutation_hooks():
    mod = runtime()
    with pytest.raises(ValueError, match='disabled'):
        mod.validate_autotune([1], limit=0)
    for configs in ([], [1, 2, 3], iter([1])):
        with pytest.raises(ValueError, match='explicit Configs'):
            mod.validate_autotune(configs, limit=2)
    with pytest.raises(ValueError, match='reset'):
        mod.validate_autotune([1, 2], limit=2, mutates=True)
    mod.validate_autotune([1, 2], limit=2, mutates=True, restore_value=['Out'])


def test_candidate_artifacts_include_disk_hits_ir_and_failure_costs(tmp_path):
    mod = runtime()
    out = tmp_path / 'triton'
    out.mkdir()
    (out / 'compilations-1.jsonl').write_text(json.dumps({
        'cache_hit': True, 'compiler_total_us': 20, 'lowering_us': 0}) + '\n')
    (out / 'kernel.ttir').write_text('candidate IR')
    (out / 'calls-1.json').write_text('{"failure":{"type":"OutOfResources"}}')
    (out / 'preflight-cache').mkdir()
    (out / 'preflight-cache/secret.npubin').write_text('must not collect')
    emitted = []
    evidence, artifacts = mod.collect(out, lambda name, data: emitted.append((name, data)) or {'name': name})
    assert evidence['disk_hits'] == 1 and evidence['lowering_us'] == 0
    assert any(name.endswith('kernel.ttir') for name, _ in emitted)
    assert not any('preflight-cache' in name for name, _ in emitted)
    assert artifacts == evidence['artifacts']
    assert profiling.benchmark_digest({'triton': evidence})['triton'] == evidence


def test_reference_collector_is_never_called(tmp_path, monkeypatch):
    adapter = _adapter()
    mod = runtime()
    monkeypatch.setitem(sys.modules, 'triton_runtime', mod)
    monkeypatch.setattr(mod, 'collect', lambda *a: pytest.fail('reference artifacts exposed'))
    result = adapter.report_to_eval(_report())
    assert adapter._triton_result(result, tmp_path, {'language': 'triton',
                                                   'evaluation_role': 'reference'}) is result


def test_triton_promotion_preserves_disclosure_and_role(triton_task, tmp_path):
    spec = importlib.util.spec_from_file_location('evolve_triton', repo_root() / 'scripts/self_evolve.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    best = {'sha': 'a' * 40, 'kernel': 'import torch\ndef exp(x: torch.Tensor):\n    return x\n',
            'ratio_basis': 'cross-job-median-vs-median', 'requested': 3, 'speedup': 1.2,
            'samples_ms': [1, 1, 1], 'ref_samples_ms': [1.2, 1.2, 1.2]}
    dest = tmp_path / 'exp-triton-g2'
    mod.next_generation(triton_task, best, dest, 1.05)
    task = load_task(dest)
    assert task.language == 'triton'
    assert 'return x' not in (task.leaderboard.card / 'kernel.py').read_text()
    assert 'import triton' in (task.leaderboard.card / 'kernel.py').read_text()
    assert json.loads((task.leaderboard.ref / 'eval/cann_task.json').read_text())['evaluation_role'] == 'reference'
    assert json.loads((task.leaderboard.card / 'eval/cann_task.json').read_text())['evaluation_role'] == 'candidate'
    assert mod.promotion_problem(task, {**best, 'samples_ms': [1]}) is not None


def test_triton_preflight_has_no_pypto_dependency(triton_task, monkeypatch, capsys):
    adapter = _adapter()
    mod = runtime()
    monkeypatch.setitem(sys.modules, 'triton_runtime', mod)
    monkeypatch.setattr(mod, 'environment', lambda arch: {'versions': {'triton-ascend': 'test'}})
    monkeypatch.setattr(adapter.shutil, 'which', lambda name: '/bin/true')
    tensor = SimpleNamespace(sum=lambda: SimpleNamespace(item=lambda: 4))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(ones=lambda *a, **kw: tensor))
    monkeypatch.setitem(sys.modules, 'torch_npu', SimpleNamespace(__version__='test'))
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == 'pypto':
            pytest.fail('Triton preflight imported PyPTO')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    assert adapter.main(['--workspace', str(triton_task.leaderboard.card), '--preflight']) == 0
    assert 'ASCEND-PREFLIGHT-OK triton=test' in capsys.readouterr().out


def test_prepare_replaces_ambient_cache_and_isolates_reference(triton_task, tmp_path, monkeypatch):
    mod = runtime()
    def probe(cmd, **kwargs):
        path = Path(cmd[cmd.index('--environment') + 1])
        path.write_text('{"target":"npu", "compiler":"actual"}')
        assert kwargs['env']['TRITON_CACHE_DIR'] != '/shared/cache'
        assert 'TRITON_KERNEL_OVERRIDE' not in kwargs['env']
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(subprocess, 'run', probe)
    card = triton_task.leaderboard.card
    config = json.loads((card / 'eval/cann_task.json').read_text())
    paths = []
    for role in ('candidate', 'reference'):
        reports = tmp_path / role
        reports.mkdir()
        env = {'TRITON_CACHE_DIR': '/shared/cache', 'TRITON_KERNEL_OVERRIDE': '1',
               'FK_TRITON_CACHE_ROOT': str(tmp_path / 'cache')}
        mod.prepare(card, card, reports, {**config, 'evaluation_role': role}, sys.executable, env)
        assert env['FK_TRITON_ROLE'] == role
        paths.append(env['TRITON_CACHE_DIR'])
    assert paths[0] != paths[1]


def test_triton_driver_rolls_back_numeric_failure_but_stops_on_missing_measurement(tmp_path):
    from test_loop_driver import FakeBackend, REC, _ctx
    config = {'profiler_backend': 'msprof', 'timing_basis': 'cann_logical_call',
              'perf_metric_strategy': 'kernel_details'}
    for numeric in (True, False):
        work = tmp_path / str(numeric)
        work.mkdir()
        result = _adapter().report_to_eval(_report(passed=1 if numeric else 2), config=config)
        backend = FakeBackend([REC('D9.9', None)], ['=== JSON ===\n' + json.dumps(result)])
        ctx = _ctx(work, backend)
        ctx.language, ctx.profiler_backend, ctx.gate_window = 'triton', 'msprof', 'device'
        if numeric:
            rec = driver.run_inner_iteration(ctx, 0, metric_before=1.0, last_good_ref='sha-prev')
            assert rec.verdict == 'REVERT' and rec.correctness is False
            assert ('reset', 'sha-prev') in backend.calls
        else:
            reason = driver.run_inner_loop(ctx, start_iter=0, initial_green=True,
                                           initial_good_ref='sha-prev')
            assert reason == 'measurement_unavailable'
            assert not any(call[0] == 'reset' for call in backend.calls)
        assert not any(call == ('gpu', 'ncu') for call in backend.calls)
        assert 'Triton Ascend coding agent' in backend.prompts[0]


def test_triton_ship_requires_three_fresh_complete_reference_confirmations(monkeypatch):
    task = SimpleNamespace(language='triton', target=SimpleNamespace(ms=None, ratio=1.1))
    confirmation = {'requested': 3, 'jobs': [{}, {}, {}], 'ref_samples_ms': [1, 1, 1],
                    'median_ms': 0.8, 'speedup': 1.25}
    monkeypatch.setattr(launch, 'confirmation', lambda *a: confirmation)
    assert launch.confirmed_at_target(None, task)
    confirmation['ref_samples_ms'].pop()
    assert not launch.confirmed_at_target(None, task)
    confirmation['ref_samples_ms'].append(1)
    confirmation['ref_workloads_mismatch'] = True
    assert not launch.confirmed_at_target(None, task)


def test_local_agent_cannot_read_raw_reference_compiler_cache(tmp_path, monkeypatch):
    from forgekernel.loop.agent_policy import from_config
    cache = tmp_path / 'private-cache'
    ambient = tmp_path / 'ambient-cache'
    monkeypatch.setenv('FK_TRITON_CACHE_ROOT', str(ambient))
    policy = from_config({}, oracle_mode='blackbox', repo_root=tmp_path,
                         teacher_trees=[], language='triton',
                         submit_env={'FK_TRITON_CACHE_ROOT': str(cache)})
    for role in ('student', 'teacher'):
        denied = policy.settings(role, tmp_path / 'ws')['sandbox']['filesystem']['denyRead']
        assert str(cache) in denied and str(ambient) in denied
        assert str(Path.home() / '.cache/forgekernel/triton') in denied
    # Only raw caches are private; the candidate's checked artifact archive stays readable.
    assert tmp_path / 'ws/.jobs/artifacts/triton' not in policy.deny_read('student')


def _alive(pid):
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
        return text[text.rindex(')') + 2:].split()[0] != 'Z'
    except FileNotFoundError:
        return False


@pytest.mark.skipif(sys.platform != 'linux', reason='requires Linux subreaper')
@pytest.mark.parametrize('terminate', [False, True])
def test_new_session_double_forked_workers_cannot_escape_cleanup(tmp_path, terminate):
    pidfile = tmp_path / 'escaped.pid'
    # The intermediate parent exits immediately; the new-session grandchild ignores
    # TERM and closes stdout. Pipe/group based cleanup alone cannot discover it.
    worker = ('import os,signal,time\n'
              'os.setsid()\n'
              'signal.signal(signal.SIGTERM, signal.SIG_IGN)\n'
              f'open({str(pidfile)!r}, "w").write(str(os.getpid()))\n'
              'os.close(1); os.close(2)\ntime.sleep(600)\n')
    leader = ('import subprocess,sys,time\n'
              f'subprocess.Popen([sys.executable,"-c",{worker!r}])\n'
              + ('time.sleep(600)\n' if terminate else 'time.sleep(0.5)\n'))
    proc = subprocess.Popen([sys.executable, str(template_dir() / 'eval/ascend_process.py'),
                             '--', sys.executable, '-c', leader],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + 10
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pidfile.exists()
        pid = int(pidfile.read_text())
        if terminate:
            proc.send_signal(signal.SIGTERM)
        output, _ = proc.communicate(timeout=20)
        assert proc.returncode == (143 if terminate else 0), output
        assert not _alive(pid), 'new-session worker survived evaluator cleanup'
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if pidfile.exists() and _alive(int(pidfile.read_text())):
            os.kill(int(pidfile.read_text()), signal.SIGKILL)
