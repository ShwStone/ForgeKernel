"""Full-call windows from the unchanged official CANN profiler's per-case CSVs.

The official trace_view strategy requires tilefwk events and cannot time Triton.
Keep the official kernel_details report and score unchanged; derive a separately
named ForgeKernel window, never relabel the historical task-duration sum.
"""
from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
import statistics

BASIS = "cann_logical_call"


def call_windows(csv_path: str | Path, repeat: int) -> dict:
    if repeat < 1:
        raise ValueError("positive repeat count required")
    path = Path(csv_path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    tasks = []
    for row in rows:
        try:
            start = float(row["Start Time(us)"])
            duration = float(row["Duration(us)"])
            if not math.isfinite(start) or not math.isfinite(duration) or duration <= 0:
                raise ValueError("nonfinite/nonpositive task")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"incomplete logical-call profiler row: {row}") from exc
        tasks.append((start, start + duration, str(row.get("Type", ""))))
    tasks.sort()
    markers = [t for t in tasks if "CannBenchCacheClean" in t[2]]
    if len(markers) != repeat:
        raise ValueError(f"logical-call delimiters {len(markers)} != repeats {repeat}")
    helpers = ("CannBenchCacheClean", "CannBenchWarmup")
    candidate = [t for t in tasks if not any(name in t[2] for name in helpers)]
    if not candidate:
        raise ValueError("no candidate NPU tasks; no timing claim")
    windows, sums, counts = [], [], []
    consumed = 0
    for i, (_, boundary, _) in enumerate(markers):
        next_start = markers[i + 1][0] if i + 1 < len(markers) else math.inf
        owned = [t for t in candidate if boundary <= t[0] < next_start]
        if not owned or any(t[1] > next_start for t in owned):
            raise ValueError("empty call or task crossing a cache-clean boundary")
        consumed += len(owned)
        windows.append(max(t[1] for t in owned) - min(t[0] for t in owned))
        sums.append(sum(t[1] - t[0] for t in owned))
        counts.append(len(owned))
    if consumed != len(candidate):
        raise ValueError("unattributed candidate tasks outside the declared call windows")
    return {"schema": 1, "device_window_us": statistics.median(windows),
            "samples_us": windows, "task_sums_us": sums, "tasks_per_call": counts,
            "repeat": repeat, "source": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "scope": "per-call min task start to max task end; all NPU tasks, internal gaps retained; "
                     "L2-clean eager; not a task-duration sum or hosted leaderboard score"}


def collect_windows(reports: Path, case_ids: list[str], repeat: int) -> dict:
    windows = {}
    for case_id in case_ids:
        rel, number = case_id.rsplit("_", 1)
        root = (reports / "prof_data" / rel / number).resolve()
        if not root.is_relative_to(reports.resolve()):
            raise ValueError("profiler case path escapes job reports")
        paths = list(root.rglob("kernel_details.csv"))
        try:
            if len(paths) != 1:
                raise ValueError(f"expected exactly one per-case CSV, got {len(paths)}")
            window = call_windows(paths[0], repeat)
            window["source"] = "cann/" + paths[0].relative_to(reports).as_posix()
            windows[case_id] = window
        except (OSError, TypeError, ValueError) as exc:
            windows[case_id] = {"status": "unavailable", "reason": str(exc)}
    return windows
