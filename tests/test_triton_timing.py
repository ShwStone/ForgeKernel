"""Logical invocation windows must include gaps/overlap without summing tasks."""
import importlib.util
import json
from pathlib import Path

import pytest

from forgekernel.baseline import BASIS_CANN_LOGICAL_CALL, HOST_FREE_BASES
from forgekernel.loop.parse import parse_eval
from forgekernel.paths import template_dir
from test_pypto import _adapter, _report


def timing():
    spec = importlib.util.spec_from_file_location('triton_timing', template_dir() / 'eval/triton_timing.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def csv_file(tmp_path, rows):
    path = tmp_path / 'kernel_details.csv'
    path.write_text('Start Time(us),Duration(us),Type\n' + '\n'.join(rows) + '\n')
    return path


def test_gaps_are_retained_and_overlapping_tasks_are_not_double_counted(tmp_path):
    path = csv_file(tmp_path, ['0,1,CannBenchCacheClean', '2,3,triton_A', '10,2,triton_B',
                               '20,1,CannBenchCacheClean', '22,10,triton_A', '25,4,triton_B'])
    result = timing().call_windows(path, repeat=2)
    assert result['samples_us'] == [10, 10]
    assert result['task_sums_us'] == [5, 14]
    assert result['device_window_us'] == 10
    assert result['tasks_per_call'] == [2, 2]
    assert 'internal gaps retained' in result['scope']
    assert len(result['sha256']) == 64


def test_large_real_profiler_timestamps_do_not_quantize_microsecond_kernels(tmp_path):
    path = csv_file(tmp_path, ['1794075192285830.001,1.007,CannBenchCacheClean',
                               '1794075192285839.518,3.481,triton_A'])
    result = timing().call_windows(path, repeat=1)
    assert result['device_window_us'] == 3.481
    assert result['task_sums_us'] == [3.481]
    assert result['samples_us'] == [3.481]


@pytest.mark.parametrize('rows,repeat', [
    (['0,1,CannBenchCacheClean', '2,3,kernel'], 2),  # partial capture
    (['0,1,CannBenchCacheClean'], 1),  # empty invocation
    (['0,1,CannBenchCacheClean', '2,20,kernel', '10,1,CannBenchCacheClean', '12,1,kernel'], 2),
    (['0,1,kernel', '2,1,CannBenchCacheClean', '4,1,kernel'], 1),  # unattributed task
    (['0,1,CannBenchCacheClean', '2,nan,kernel'], 1),
    (['0,1,CannBenchCacheClean', '2,0,kernel'], 1),
])
def test_ambiguous_missing_or_nonfinite_capture_is_unavailable(tmp_path, rows, repeat):
    with pytest.raises(ValueError):
        timing().call_windows(csv_file(tmp_path, rows), repeat)


def test_adapter_requires_explicit_logical_call_evidence_not_elapsed_sum():
    adapter = _adapter()
    config = {'profiler_backend': 'msprof', 'timing_basis': 'cann_logical_call',
              'perf_metric_strategy': 'kernel_details'}
    report = _report()
    missing = adapter.report_to_eval(report, config=config)
    assert missing['correctness']['pass'] and not missing['performance']
    assert missing['measurement']['status'] == 'unavailable'
    windows = {case['case_id']: {'schema': 1, 'device_window_us': case['elapsed_us'] * 2}
               for case in report['operators'][0]['cases']}
    result = adapter.report_to_eval(report, config=config, logical_windows=windows)
    ev = parse_eval('=== JSON ===\n' + json.dumps(result))
    assert ev.correctness and ev.device_median_ms == pytest.approx(0.4)
    assert result['performance']['device_window_basis'] == BASIS_CANN_LOGICAL_CALL
    assert BASIS_CANN_LOGICAL_CALL in HOST_FREE_BASES
    assert result['cann_bench']['cases'][0]['elapsed_us_source'] == 'logical_call.device_window_us'
    assert result['cann_bench']['official_report_timing_basis'] == 'cann_kernel_details'
    assert result['cann_bench']['cases'][0]['elapsed_us'] == 100  # original report is not rewritten
    assert result['cann_bench']['cases'][0]['selected_elapsed_us'] == 200
