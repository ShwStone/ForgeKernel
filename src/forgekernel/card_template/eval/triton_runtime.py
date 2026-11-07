"""Experimental Triton Ascend runtime, also copied into the submission wheel.

Only the candidate worker observes compilations. The sealed reference has a separate
cache and never installs a listener or exports its IR/metadata. No GPU imports at
module scope: cache policy and artifact collection are testable on a CPU host.
"""
from __future__ import annotations

import argparse
import atexit
import functools
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
MAX_AUTOTUNE_CONFIGS = 8
_INSTALLED = False
_ACTIVE_SIGNATURE = None


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_record(path: str | Path) -> dict:
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Separate workers own separate files; replace keeps a killed writer's last record intact.
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    tmp.replace(path)


def environment(arch: str) -> dict:
    if os.environ.get("TRITON_INTERPRET", "0") != "0":
        raise RuntimeError("TRITON_INTERPRET is forbidden: real NPU execution is required")
    import torch
    import torch_npu
    import triton
    from triton.runtime import driver
    from triton.compiler.compiler import make_backend
    if not hasattr(triton, 'knobs') or not hasattr(triton.knobs.compilation, 'listener'):
        raise RuntimeError('Triton compilation listener API unavailable; supported Triton-Ascend 3.6 required')
    torch.npu.set_device(0)
    target = driver.active.get_current_target()
    device = torch.npu.get_device_name()
    if arch != "ascend950pr" or target.backend != "npu" or "950PR" not in device.upper():
        raise RuntimeError(f"uncertified Triton target: arch={arch}, target={target}, device={device}")
    versions = {}
    for package in ("triton", "triton-ascend", "torch", "torch-npu"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    # Hash the installed compiler/driver/cache/autotuner implementation, not just its
    # version string. Include binaries and local patches in the Triton package.
    root = Path(triton.__file__).resolve().parent
    files = [file_record(p) for p in sorted(root.rglob("*"))
             if p.is_file() and p.suffix in (".py", ".so")]
    modules = [file_record(m.__file__) for m in (torch, torch_npu)]
    for module in (getattr(torch, '_C', None), getattr(torch_npu, '_C', None)):
        if module is not None and getattr(module, '__file__', None):
            modules.append(file_record(module.__file__))
    tools = [file_record(p) for name in ("bisheng", "bishengir-compile", "ccec", "msprof")
             if (p := shutil.which(name))]
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    versions_files = [file_record(p) for name in ("version.cfg", "version.info",
                     "compiler/version.info", "opp/version.info") if (p := cann / name).is_file()]
    build_env = {k: v for k, v in os.environ.items()
                 if k.startswith(("TRITON_", "BISHENG", "ASCEND_", "LD_LIBRARY_PATH"))
                 and not any(word in k for word in ("CACHE", "DUMP", "LOG", "VISIBLE"))}
    return {"arch": arch, "target": str(target), "device": device,
            "python": {"version": sys.version, **file_record(sys.executable)},
            "versions": versions, "triton_files": files, "modules": modules,
            "torch_abi": {"git": torch.version.git_version,
                          "cxx11": getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", None)},
            "tools": tools, "cann": versions_files, "build_env": build_env,
            "compiler_defaults": vars(make_backend(target).parse_options({}))}


def cache_directory(root: Path, fingerprint: str, role: str, source_hash: str) -> Path:
    if role not in ("candidate", "reference"):
        raise ValueError("Triton evaluation_role must be candidate or reference")
    for value in (fingerprint, source_hash):
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("invalid cache fingerprint")
    path = root / fingerprint / role / source_hash
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def prepare(workspace: Path, source: Path, reports: Path, config: dict,
            python: str, env: dict) -> dict:
    """Freeze the execution environment in a bounded, fresh interpreter."""
    role = config.get("evaluation_role")
    if role not in ("candidate", "reference"):
        raise ValueError("missing trusted Triton evaluation_role")
    if config.get("arch") != "ascend950pr":
        raise ValueError("only ascend950pr is certified for experimental Triton")
    if config.get("timing_basis") != "cann_logical_call" or config.get("perf_metric_strategy") != "kernel_details":
        raise ValueError("Triton requires the explicit complete logical-call window")
    limit = config.get("autotune_max_configs", 0)
    if type(limit) is not int or not 0 <= limit <= MAX_AUTOTUNE_CONFIGS:
        raise ValueError(f"autotune_max_configs must be 0..{MAX_AUTOTUNE_CONFIGS}")
    # Ignore ambient global caches, dumps and tuning records, including in preflight.
    for key in list(env):
        if key.startswith("TRITON_") and any(word in key for word in ("CACHE", "DUMP", "OVERRIDE")):
            env.pop(key)
    out = reports / "triton"
    out.mkdir(mode=0o700)
    snapshot = out / "environment.json"
    env["TRITON_CACHE_DIR"] = str(out / "preflight-cache")
    import subprocess
    proc = subprocess.run([python, str(workspace / "eval/triton_runtime.py"),
                           "--environment", str(snapshot), "--arch", config["arch"]],
                          env=env, capture_output=True, text=True, timeout=90)
    if proc.returncode:
        raise RuntimeError("Triton NPU environment check failed: " + (proc.stdout + proc.stderr)[-4000:])
    snapshot_data = json.loads(snapshot.read_text())
    fingerprint = digest(snapshot_data)
    sources = {str(p.relative_to(source)): file_record(p)["sha256"]
               for p in sorted(source.rglob("*.py")) if "__pycache__" not in p.parts}
    root = Path(env.get("FK_TRITON_CACHE_ROOT", str(Path.home() / ".cache/forgekernel/triton"))).expanduser().resolve()
    if root.is_relative_to(workspace.resolve()) or workspace.resolve().is_relative_to(root):
        raise ValueError("Triton cache needs a dedicated directory outside the job workspace")
    cache = cache_directory(root, fingerprint, role, digest(sources))
    # The worker sets these again in Triton's knobs before importing the candidate.
    env.update(TRITON_CACHE_DIR=str(cache / "cache"), TRITON_DUMP_DIR=str(cache / "dump"),
               TRITON_OVERRIDE_DIR=str(cache / "override"), TRITON_CACHE_AUTOTUNING="0",
               FK_TRITON_ROLE=role, FK_TRITON_OUTPUT=str(out),
               FK_TRITON_AUTOTUNE_MAX_CONFIGS=str(limit),
               FK_TRITON_MUTATES_INPUTS="1" if config.get("mutates_inputs") else "0")
    save(out / "protocol.json", {"environment_fingerprint": fingerprint,
         "role": role, "source_hash": digest(sources), "cache": str(cache),
         "autotune_max_configs": limit, "timing_basis": "cann_logical_call",
         "scope": "official logical-call device window; compile/host costs reported separately"})
    return {"environment_fingerprint": fingerprint, "role": role, "cache": str(cache)}


def validate_autotune(configs, *, limit: int, mutates: bool = False,
                      reset_to_zero=None, restore_value=None) -> None:
    if limit == 0:
        raise ValueError("Triton autotune disabled: use fixed configuration or explicitly bound the search")
    if not isinstance(configs, (list, tuple)) or not 1 <= len(configs) <= limit:
        raise ValueError(f"autotune needs 1..{limit} explicit Configs; automatic/empty tiling is forbidden")
    if mutates and not (reset_to_zero or restore_value):
        raise ValueError("in-place/atomic autotune requires reset_to_zero or restore_value hooks")


def install() -> None:
    """Called by the wheel before kernel import; no reference listener is installed."""
    global _INSTALLED
    if _INSTALLED:
        return
    role = os.environ.get("FK_TRITON_ROLE")
    if role not in ("candidate", "reference"):
        raise RuntimeError("Triton package must run through its frozen CANN entry")
    import triton
    from triton.runtime import driver, autotuner
    target = driver.active.get_current_target()
    if target.backend != "npu" or os.environ.get("TRITON_INTERPRET", "0") != "0":
        raise RuntimeError(f"real NPU target required, got {target}")
    triton.knobs.cache.dir = os.environ["TRITON_CACHE_DIR"]
    triton.knobs.cache.dump_dir = os.environ["TRITON_DUMP_DIR"]
    triton.knobs.cache.override_dir = os.environ["TRITON_OVERRIDE_DIR"]
    # A reference cannot inherit a candidate observer, even in an embedding process.
    triton.knobs.compilation.listener = None
    _INSTALLED = True
    if role == "reference":
        return
    out = Path(os.environ["FK_TRITON_OUTPUT"])
    log = out / f"compilations-{os.getpid()}.jsonl"

    def listener(*, src, metadata, metadata_group, times, cache_hit):
        record = {"pid": os.getpid(), "name": metadata.get("name"),
                  "specialization": _ACTIVE_SIGNATURE,
                  "hash": metadata.get("hash"), "cache_hit": cache_hit,
                  "compiler_total_us": times.total, "lowering_us": times.total_lowering,
                  "lowering_stages_us": times.lowering_stages, "metadata": metadata}
        with log.open("a") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
        directory = out / str(metadata["hash"])
        directory.mkdir(exist_ok=True)
        # Bound collection, never recurse through the global cache or reference tree.
        for path in metadata_group.values():
            path = Path(path)
            used = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
            if path.is_file() and used + path.stat().st_size <= MAX_ARTIFACT_BYTES:
                shutil.copy2(path, directory / path.name)
        save(directory / "observed_metadata.json", metadata)

    triton.knobs.compilation.listener = listener
    original = autotuner.Autotuner.__init__
    signature = inspect.signature(original)

    @functools.wraps(original)
    def bounded(self, *args, **kwargs):
        # Ascend AutoTilingTuner subclasses this class and can expand even an
        # explicit list or compile in parallel. It is intentionally not admitted.
        if type(self) is not autotuner.Autotuner:
            raise ValueError("only the standard serial Triton Autotuner is supported; automatic tiling is forbidden")
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        validate_autotune(bound.arguments["configs"],
            limit=int(os.environ.get("FK_TRITON_AUTOTUNE_MAX_CONFIGS", "0")),
            mutates=os.environ.get("FK_TRITON_MUTATES_INPUTS") == "1",
            reset_to_zero=bound.arguments.get("reset_to_zero"),
            restore_value=bound.arguments.get("restore_value"))
        if bound.arguments.get("prune_configs_by"):
            raise ValueError("dynamic config pruning is unavailable in the fixed bounded search")
        if any(not isinstance(c, triton.Config) for c in bound.arguments["configs"]):
            raise ValueError("autotune requires explicit triton.Config objects")
        original(self, *args, **kwargs)
    autotuner.Autotuner.__init__ = bounded


def track_entry(fn):
    """Observe costs without replacing the official correctness judge or stopwatch."""
    if os.environ.get("FK_TRITON_ROLE") != "candidate":
        return fn
    out = Path(os.environ["FK_TRITON_OUTPUT"])
    path = out / f"calls-{os.getpid()}.json"
    state = {"calls": 0, "host_callable_seconds": 0.0, "specializations": {},
             "scope": "host callable only, not gate device latency; first call per specialization includes synchronize"}

    def flush():
        save(path, state)
    atexit.register(flush)

    @functools.wraps(fn)
    def tracked(*args, **kwargs):
        global _ACTIVE_SIGNATURE
        import torch
        def describe(value):
            if isinstance(value, torch.Tensor):
                return {"shape": list(value.shape), "dtype": str(value.dtype), "stride": list(value.stride())}
            if isinstance(value, (bool, int, float, str, type(None))):
                return value  # semantic attrs, never tensor contents or data pointers
            return type(value).__name__
        signature = {"args": [describe(v) for v in args],
                     "kwargs": {k: describe(v) for k, v in kwargs.items()}}
        key = digest(signature)
        first = key not in state["specializations"]
        case = state["specializations"].setdefault(key, {"signature": signature, "calls": 0})
        _ACTIVE_SIGNATURE = key
        if first:
            torch.npu.synchronize()
        started = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            if first:
                torch.npu.synchronize()
                case["first_call_sync_seconds"] = time.perf_counter() - started
                # Flush before any worker can be killed mid-evaluation.
                for name, value in fn.__globals__.items():
                    if hasattr(value, "best_config"):
                        case.setdefault("autotune", {})[name] = {
                            "best_config": str(value.best_config),
                            "configs_timings": str(getattr(value, "configs_timings", None))}
            return result
        except Exception as exc:
            state["failure"] = {"type": type(exc).__name__, "error": str(exc),
                                "traceback": traceback.format_exc()}
            raise
        finally:
            _ACTIVE_SIGNATURE = None
            case["calls"] += 1
            state["calls"] += 1
            state["host_callable_seconds"] += time.perf_counter() - started
            if first or "failure" in state:
                flush()
    return tracked


def collect(out: Path, emit) -> tuple[dict, list[dict]]:
    """Only called for a trusted candidate role, even on failure/interruption."""
    files = []
    omitted = []
    total = 0
    records = []
    for path in sorted(out.rglob("*")):
        if not path.is_file() or "preflight-cache" in path.parts or path.suffix == ".tmp":
            continue
        size = path.stat().st_size
        if size + total > MAX_ARTIFACT_BYTES:
            omitted.append({"name": path.relative_to(out).as_posix(), "bytes": size,
                            "reason": "job artifact byte budget"})
            continue
        total += size
        files.append(emit("triton/" + path.relative_to(out).as_posix(), path.read_bytes()))
        if path.name.startswith("compilations-"):
            for line in path.read_text().splitlines():
                try:
                    records.append(json.loads(line))
                except ValueError:  # a killed writer's incomplete last line
                    pass
    return {"status": "available" if records else "unavailable",
            "compilations": len(records), "disk_hits": sum(bool(r["cache_hit"]) for r in records),
            "compiler_total_us": sum(r["compiler_total_us"] for r in records),
            "lowering_us": sum(r["lowering_us"] for r in records),
            "scope": "candidate only; listener time is not first-call wall or device latency",
            "artifacts": files, "omitted_artifacts": omitted}, files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--arch", required=True)
    args = parser.parse_args()
    save(args.environment, environment(args.arch))
