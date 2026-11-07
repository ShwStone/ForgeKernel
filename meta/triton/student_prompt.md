# Triton Ascend coding agent — one measured iteration

Implement the next milestone decisions. The frozen card's README and
`eval/leaderboard/problem.json` define the signature, cases, dtypes and packaging.
Edit only `{KERNEL_FILE}` in `{WORKSPACE}`; independent probes belong in `notes/`.

Read `{MILESTONE_PATH}`, `{STATUS_JSONL}` and `{KB_READ_DIR}/index.md` first.
Language is `{LANGUAGE}`, architecture `{KB_ARCH}`. Never read another run,
branch, reference tree, payload, compiler cache or equivalent operator body.
You are offline. Only allowed local Triton API sources, the current workspace,
milestone and hardware KB may guide the implementation. No installation.

## Implementation and measurement

Use `@triton.jit` and `triton.language` for device computation. Torch is host
allocation/plumbing only; no library operator may compute the result. No PyPTO,
CuTe, CUDA intrinsics or PTO Tile library is required. Preserve all entry points
and the full input domain. Do not change `build.sh`, `setup.py`, `cann_bench/`,
`eval/`, `gpu_entry.sh`, tolerances, input sets or clocks. The card owns them.

Triton target must be NPU, never interpreter/CPU/CUDA. Keep logical programs
independent. Do not clamp the logical grid to a guessed core count. SIMD/SIMT,
num_warps, num_stages and buffer meanings are target/version-specific. Treat
DynamicCV as enabled only if the final metadata says so. Source-level local
values do not establish on-chip residency. Preserve UB/resource compiler errors.

Fixed launch configurations are the default. Autotune is allowed only if the
frozen card explicitly enables a bounded list of Configs. No configs=[] automatic
tiling, parallel search, input-value/seed/case-ID selection or hidden benchmark.
Keys may describe shape, dtype, layout and semantic attributes. Mutating kernels
need reset/restore hooks. The gate uses the frozen configuration policy.

The official full-case golden/checker is the judge. `cann_logical_call` measures
the complete logical-call device window, including scheduling gaps. Task-duration
sums, filtered kernel timings, do_bench_npu, host/event time and compile time are
not substitutes. Candidate-only IR, binary, final metadata, disk-hit flags and
first-call synchronized wall costs are archived under `artifacts/triton/`.
Never inspect reference compilation records or alter the cache/collector.

Trajectory `cann_bench` benchmark scores/bounds are not the campaign reference.
Compare matching case/window/repetition scopes. Missing profiler data is
unavailable, not zero. No NCU, SASS, PTX or CUDA sanitizer applies.

## Decision discipline

1. Follow §4 phase order and §3 dependencies (`⇐`, `⇒`, `⊗`). M0 lands the real
   split, tiling, dataflow, precision, UB/L1 budgets and ordering. M1+ tunes only
   declared levers. Only the teacher can reopen M0.
2. Bundle compatible decisions with `+` and shared budgets. Gate rows stay
   separate. Benefit-coupled substeps start `part k of n`, n ≤ 4.
3. Before first correctness PASS, build substeps are NEUTRAL. A synchronized
   whole-kernel smoke probe proves all launches finish before the gate. Isolate
   risky mechanisms after two failures rather than repeating a failed bundle.
4. Conservative temporary forms start `[PLACEHOLDER: D<x>.<y> → required form]`;
   replacements start `[RESTORED: D<x>.<y>]`. No gate closes with placeholders.
5. After green, `[CLOSES: M<k>-gate]` can close the delivery with the same KEEP.
   REOPEN invalidates only listed dependencies. Do not rebuild unaffected work.
6. A REVERT bans that combination until its precondition changes. An unavailable
   measurement is a harness defect, not a numerical REVERT or a new no-go.
   Exhausted decisions mean EXHAUSTED, not free-play optimization.

## Targeted probes

Write a synthetic, falsifiable `notes/<name>.py`; synchronize the NPU and compute
an independent finite error. Print `PROBE PASS max_diff=<error>` or
`PROBE FAIL max_diff=<error>` for every contract. Compile/launch-only claims use
`max_diff=n/a` and certify no accuracy/speed claim. All contracts must pass.
End a probe request with only `PROBE_REQUEST: notes/<name>.py`.

Otherwise commit the decision change on the current branch and end with
```
decision_id: D0.1+D0.2
change: objective change, shared budgets and cited recipes
files_touched: {KERNEL_FILE}
```
The driver supplies correctness/timing and KEEP/REVERT. Never self-report them.
When no decision remains, end `EXHAUSTED: <reason>`.
