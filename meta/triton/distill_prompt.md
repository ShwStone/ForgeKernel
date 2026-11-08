# Triton Ascend architecture agent — initial curriculum

Read `{MILESTONES_TEMPLATE_PATH}` and `{META_MILESTONE_PATH}` for the common
skeleton, five-tuples, DAG, no-code boundary and evidence discipline. These Ascend
rules replace NVIDIA instrumentation rules. Source oracle `{ORACLE}` is readable
only in this source-mode run. Do not copy/transliterate any implementation body
into the milestone or KB; the student implements the decisions independently.

Architecture reference (the only stop anchor)
{ARCH_REF}

Reference measurements and official profiler metadata
{REF_BASELINE}

Timed-window contract
{WINDOW_CONTRACT}

Read `{KB_READ_DIR}/index.md` for `{KB_ARCH}`. Write `{MILESTONE_PATH}` using
exact `## 0.`–`## 6.` and `## Worked Example` headings. Every D<x>.<y> contains
Action / Purpose / Trigger condition / Cost / Depends on; declare explicit `⇐`
prerequisites and `⊗` benefit coupling. Every phase has a `### M<k> Gate` and a
literal `mechanism-verification line`.

M0 settles actual decomposition, logical grid, dataflow, precision, intermediate
ABI, tiles, UB/L1 capacity/lifetimes and producer/consumer ownership. Identify
compiler-managed synchronization versus explicit ordering. Never map NVIDIA
warp assumptions onto NPU SIMD. Logical grid is not physical block count; do
not truncate it. Local variables do not prove on-chip residency. DynamicCV and
multibuffering claims require final metadata/IR plus a measured contract, not
requested options. Keep compiler resource failures and failed probes as evidence.
M0 gate is correctness + measured architecture/mechanism, not an absolute time
threshold. M1+ has bounded tile/stage/unroll ranges and priced tradeoffs within
that topology. Topology changes reopen the earliest M0 dependency. No planned
placeholder architecture or free-play phase.

Only real Ascend channels can verify gates
- `[VERIFY: correctness]` — full official CANN golden/checker.
- `[VERIFY: cann-logical-call]` — complete logical-call device window, internal gaps.
- `[VERIFY: msprof-direct]` — actual CSV fields with file, job, case and repeat scope.
- `[VERIFY: msprof-derived]` — an explicit derivation from those fields.
- `[VERIFY: probe]` — a driver-run falsifiable NPU experiment.
- `[VERIFY: inferred]` / `[VERIFY: none]` — hypotheses only, never hard gates.

NCU/SASS/SYNC/GATHER/PHASE rules are inapplicable. Missing metrics stay
unavailable. Timings alone do not establish overlap/stalls. Never add op_summary
and kernel_details or divide aggregate recordings by guessed call counts.
Task-duration sum, do_bench_npu, host/event time and compile time are not the gate
window. Compiler listener costs and first-call wall must be reported separately.
Autotune is off unless the frozen card permits bounded explicit Configs; generated
automatic tiling is not allowed. Formal gates preserve that frozen policy.

Preserve `target_window: {GATE_WINDOW}` and the injected target, derived only
from the campaign reference. Every performance statement carries
`[MEASURED, job N]` or `[ESTIMATE]` and derivation. Benchmark baseline, bounds and
scores never replace reference timings. Cross-job ratios are not paired; public
local cases are not hosted hidden-set scoring. Cite certified KB recipes by name
and record dead ends with evidence and a retry precondition.
Write the complete milestone. Final reply is a one-line summary.
