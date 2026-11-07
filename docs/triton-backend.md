# Experimental Triton Ascend backend

`language = "triton"` is an independent agent mode with `meta/triton/` prompts,
`hw_kb/triton/ascend950pr/`, a Python kernel ABI and the shared CANN-Bench
adapter. It does not fall back to CuTe prompts or import PyPTO/PTO Tile for
preflight. Only Ascend950PR CANN-Bench tasks are admitted. Generic task/judge
generation, other Ascend models, CUDA and ROCm are deliberately rejected until
independently validated. This is not a default-backend change.

## Generate a task

The controller needs ForgeKernel, PyYAML and a local CANN-Bench checkout at the
same commit as the execution host. The NPU host needs Triton-Ascend 3.6 with the
compilation listener API, CANN, TorchNPU, the official evaluator and msprof.
Neither generator nor preflight installs or upgrades packages. Evaluation builds
and installs the submission through the official CANN package manager. Use a
dedicated evaluator environment and serialize jobs sharing that interpreter;
concurrent installations of the global `cann_bench` package are not supported by
this experimental path. Separate Triton cache roles do not isolate Python package
installation from other jobs.

```bash
python scripts/init_triton_cann_task.py tasks/gqa-triton \
  --cann-bench-dir /path/to/cann-bench \
  --expected-cann-ref 059cb61d41b279ad021160517bd2a5f6b7a05e01 \
  --dsl-source-dir /path/to/local/triton-api-sources
forge new tasks/gqa-triton --dry-run
forge new tasks/gqa-triton
forge run .artifacts/gqa-triton_<id>
```

The generator defaults to the official 20-case GQA task and a separate CANN
library reference. Both teacher and student are independent of that source in
blackbox mode. The candidate starts with only a signature stub. The run seals
the reference into its teacher-only tree; no payload or key rides a student job.
Point `dsl_source_dirs` at permitted local API sources on the controller because
the student is offline. No default API directory from another DSL is substituted.

Other official operators use the same builder with `--level`, `--operator-path`,
`--operator`, `--entry`, `--signature` and a self-contained `--reference-file`
defining that entry. `--reference-name` labels its actual provenance. Do not use
a ready-made candidate implementation as an agent seed.

Machine configuration stays in `local.env`, not the task:

```bash
FK_DELIVERY=ssh
FK_GPU=1
REMOTE_HOST=my-ascend-host
REMOTE_WORKDIR=/srv/forgekernel-remote
REMOTE_ENV_FILE=/srv/forgekernel/repro/ascend/remote_env.sh
FK_ASCEND_PYTHON=/srv/venv/bin/python
CANN_BENCH_DIR=/srv/cann-bench
# Optional dedicated root; never a shared global Triton cache.
FK_TRITON_CACHE_ROOT=/srv/forgekernel-triton-cache
```

The runtime selects visible device 0, confirms a real NPU target and an actual
950PR device, and rejects interpreter/CUDA fallback. The submit configuration
sets NPU visibility rather than CUDA visibility. No NCU job is submitted.

## Correctness and the new timing currency

The official CANN golden, checker, inputs, input rotation, profiler and cache
clean remain unchanged. Full jobs evaluate the entire declared case set. Scout
jobs carry `workload_scope`; they cannot clear a gate or become a confirmation.

The pinned evaluator's `trace_view` strategy requires tilefwk/PyPTO events and
cannot measure ordinary Triton. Its `kernel_details` score instead sums task
durations. Neither is silently renamed into full-call latency.

Triton therefore declares **`timing_basis = "cann_logical_call"`** while asking
the official evaluator to collect `kernel_details`. Each case has its own
profiler session. The adapter locates exactly one CSV per case, checks that its
cache-clean delimiters match the configured repeat count, and groups *all* NPU
tasks between those boundaries. Each sample is `max(task end) - min(task start)`;
the case reading is the median. Internal scheduling gaps remain; overlapping
tasks are not double-counted. Empty, missing, malformed, crossing or incomplete
captures are measurement-unavailable, never zero or a fabricated PASS.

The raw official report and score are archived unchanged. Normalized evidence
labels its `official_report_timing_basis = "cann_kernel_details"` separately
from the adapter's selected window. `cann/logical_windows.json` holds samples,
task counts, diagnostic sums, CSV paths and SHA256. These are L2-clean, eager,
public-case measurements, not hosted hidden-set scores, graph benchmarks or
hot-cache timings. Task sums, do_bench_npu, host/event measurements and compiler
listener durations do not replace the selected window.

Candidate and reference must declare the same basis. Final shipping/promotion
requires at least three complete candidate and three fresh reference jobs with
matching workloads. Separate-job median ratios are not paired same-job ratios;
promotion must also clear the two-sided noise floor. Correctness failures roll
back; missing measurement evidence stops for a harness fix without banning the
candidate combination.

## Compilation, cache and evidence

A bounded environment probe records the actual target, package versions, Python
executable, Torch/TorchNPU ABI, installed Triton Python/shared-library hashes,
compiler/driver/default options, CANN version files and tool paths/hashes.
Ambient global cache, override and dump settings are not reused.

Cache namespaces include the environment fingerprint, **candidate/reference
role** and frozen Python-source hash; Triton specializes further by signature
and options. These are private ForgeKernel directories. No global cache is
cleared. Different roles never share a cache, even for identical source. The cache
root must be outside the job workspace. Raw cache roots are denied to the student
and blackbox teacher, including for local delivery; only checked candidate job
artifacts are agent-readable.

Only the candidate worker installs the compiler listener. Artifacts include
TTIR/lowered IR, binary, final metadata, cache-hit flags, stage costs and
per-specialization synchronized first-call wall costs. Host-call observations
are separately labelled and never reported as device latency. Compiler errors
and stderr survive failures. Collection is bounded; absent evidence remains
unavailable. The reference has no listener or compilation-artifact export.
Its report/CSV timings are still visible, as required for a blackbox reference.

## Search and resource budgets

Autotune is disabled by default. `--autotune-max-configs N` admits at most N
explicit standard `triton.Config` objects, with N in 1..8. Automatic/empty tiling
and dynamic config pruning are rejected. Search is serial, under the same
frozen source/config policy as the gate; tensor values, seeds and case IDs must
not select answers. `--mutates-inputs` requires reset/restore hooks before an
in-place/atomic kernel can autotune. Selections and compilation costs are kept.

Each official worker handles one case, with `--case-timeout` (default 180 s)
covering compilation and evaluation. `eval_timeout_s` defaults to 7200 s for the
whole job; runtime preflight is capped at 120 s. Linux subreapers own
evaluator/compiler/probe descendants, including
new-session and double-forked processes. TERM is followed by bounded KILL and
reaping; a surviving process is reported as failure. A stopped job keeps cost
and partial evidence, but gets no complete correctness/performance verdict.

## Acceptance and reproduction

CPU coverage includes language/target/profiler isolation, task generation,
sealed references, cache roles, bounded tuning, unavailable measurements,
rollback, confirmation/promotion and real subprocess cleanup across sessions.
Run `pytest` from the repository root.

The maintained hardware smoke uses a fixed diagnostic kernel, not a new agent
campaign, and runs the generated card through actual SSH delivery:

```bash
python repro/ascend/triton_backend/backend_smoke.py \
  --out .artifacts/triton-exp-smoke --operator exp \
  --cann-bench-dir /path/to/local/cann-bench --jobs 2 --reference-jobs 1
python repro/ascend/triton_backend/backend_smoke.py \
  --out .artifacts/triton-gqa-smoke --operator gqa \
  --cann-bench-dir /path/to/local/cann-bench --jobs 1 --reference-jobs 1 --cleanup-check
```

`--cleanup-check` also starts a TERM-ignoring, new-session orphan holding a real
NPU allocation. After the job's supervisor kills/reaps it, a separate job checks
that process identity is gone and a fresh NPU allocation/synchronization works.
The spawn probe refuses execution outside the frozen process supervisor.

Use a fresh output directory. Both scripts preserve raw logs and checksummed
artifacts, and stop on failure. Exp uses an identical frozen diagnostic baseline
only to exercise reference isolation; it is not a vendor A/B. The GQA fixture
keeps the RFC expression but preserves the softmax probability rounding residual
in a second low-precision dot. The original dated probe and evidence remain
unchanged. Its full-case failure and the correction are retained in the acceptance
record. `--scout-cases level4/gqa_15` requests only a candidate diagnostic and
keeps its non-gate scope; it never runs a reference confirmation. GQA uses the
separate CANN library as its full-task reference. Neither script establishes an agent
optimization result or satisfies three-job campaign confirmation by itself.
See the [acceptance record](../repro/ascend/triton_backend/backend-acceptance.md)
and its checksummed JSON for actual tested scope, retained failures and the
decimal-timestamp precision correction.
