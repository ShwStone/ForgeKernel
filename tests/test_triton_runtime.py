"""Exercise the real runtime hooks with CPU doubles of the supported Triton API."""
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from test_triton_backend import runtime


@pytest.fixture
def fake_triton(tmp_path, monkeypatch):
    class Config:
        pass
    class Autotuner:
        def __init__(self, fn, arg_names, configs, key, reset_to_zero=None,
                     restore_value=None, prune_configs_by=None):
            self.configs = configs
    triton = ModuleType('triton')
    triton.Config = Config
    triton.knobs = SimpleNamespace(cache=SimpleNamespace(),
                                   compilation=SimpleNamespace(listener='old observer'))
    triton_runtime = ModuleType('triton.runtime')
    triton_runtime.driver = SimpleNamespace(active=SimpleNamespace(
        get_current_target=lambda: SimpleNamespace(backend='npu')))
    triton_runtime.autotuner = SimpleNamespace(Autotuner=Autotuner)
    monkeypatch.setitem(sys.modules, 'triton', triton)
    monkeypatch.setitem(sys.modules, 'triton.runtime', triton_runtime)
    for key, value in {'FK_TRITON_ROLE': 'candidate', 'FK_TRITON_OUTPUT': str(tmp_path),
                       'FK_TRITON_AUTOTUNE_MAX_CONFIGS': '2',
                       'TRITON_CACHE_DIR': str(tmp_path / 'candidate/cache'),
                       'TRITON_DUMP_DIR': str(tmp_path / 'candidate/dump'),
                       'TRITON_OVERRIDE_DIR': str(tmp_path / 'candidate/override')}.items():
        monkeypatch.setenv(key, value)
    return triton, Autotuner, Config


def test_installed_candidate_hook_bounds_standard_tuning_and_captures_metadata(fake_triton, tmp_path):
    triton, Autotuner, Config = fake_triton
    runtime().install()
    assert triton.knobs.cache.dir.endswith('candidate/cache')
    assert callable(triton.knobs.compilation.listener)
    assert len(Autotuner(None, [], [Config(), Config()], []).configs) == 2
    with pytest.raises(ValueError, match='explicit Configs'):
        Autotuner(None, [], [], [])
    with pytest.raises(ValueError, match='dynamic config pruning'):
        Autotuner(None, [], [Config()], [], prune_configs_by={'top_k': 1})
    class AutoTilingTuner(Autotuner):
        pass
    with pytest.raises(ValueError, match='automatic tiling'):
        AutoTilingTuner(None, [], [Config()], [])
    source = tmp_path / 'kernel.ttir'
    source.write_text('candidate only IR')
    triton.knobs.compilation.listener(src=None, metadata={'name': 'probe', 'hash': 'a' * 64},
        metadata_group={'ir': str(source)},
        times=SimpleNamespace(total=12, total_lowering=0, lowering_stages={}), cache_hit=True)
    assert (tmp_path / ('a' * 64) / 'kernel.ttir').read_text() == 'candidate only IR'
    record = json.loads(next(tmp_path.glob('compilations-*.jsonl')).read_text())
    assert record['cache_hit'] is True and record['lowering_us'] == 0
    assert record['metadata']['name'] == 'probe'


def test_reference_worker_removes_observer_and_never_exports_ir(fake_triton, tmp_path, monkeypatch):
    triton, _, _ = fake_triton
    monkeypatch.setenv('FK_TRITON_ROLE', 'reference')
    runtime().install()
    assert triton.knobs.compilation.listener is None
    assert not list(tmp_path.glob('compilations-*'))
    assert not list(tmp_path.glob('observed_metadata.json'))


def test_full_gqa_fixture_preserves_dated_probe_and_uses_probability_residual():
    import importlib.util
    from forgekernel.paths import repo_root
    root = repo_root()
    original = root / 'repro/ascend/triton_backend/kernels.py'
    before = original.read_bytes()
    path = root / 'repro/ascend/triton_backend/backend_smoke.py'
    spec = importlib.util.spec_from_file_location('backend_smoke', path)
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    source = smoke.gqa_candidate_source()
    compile(source, 'kernel.py', 'exec')
    assert 'tl.dot(p_hi, v) + tl.dot(p_lo, v)' in source
    assert 'def gqa(query, key, value, scaleValue=-1.0, is_causal=False)' in source
    assert original.read_bytes() == before


def test_npu_cleanup_probe_refuses_an_unowned_spawn(monkeypatch):
    import importlib.util
    from forgekernel.paths import repo_root
    path = repo_root() / 'repro/ascend/triton_backend/cleanup_probe.py'
    spec = importlib.util.spec_from_file_location('cleanup_probe', path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    monkeypatch.setattr(probe, 'under_supervisor', lambda: False)
    with pytest.raises(RuntimeError, match='must run under'):
        probe.main(['spawn'])
