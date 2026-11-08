# Triton Ascend architecture agent — sealed-reference curriculum

The reference source is NOT readable. Do not inspect, decode, disassemble or
unpack its payload, compiler cache, IR, binary or tuning records. Use only the
reference's measured behavior and `{RECON_REPORT}`. Milestones contain decisions
and contracts, never implementation code.

Read `{META_MILESTONE_PATH}`, `{MILESTONES_TEMPLATE_PATH}` and
`{KB_READ_DIR}/index.md` for `{KB_ARCH}`. Write `{MILESTONE_PATH}` with exact
§0–§6 and Worked Example headings, five-tuples (Action/Purpose/Trigger
condition/Cost/Depends on), explicit `⇐` DAG edges and `⊗` companions. Order
M0→M1+ with per-phase Gate blocks and literal mechanism-verification lines.
No free-play phase or preplanned placeholder architecture.

Architecture reference (sole stop anchor)
{ARCH_REF}

Reference measurements, official CANN metadata and Ascend profiler
{REF_BASELINE}

Timed-window contract
{WINDOW_CONTRACT}

M0 establishes real split, logical grid, dataflow, precision, intermediate ABI,
tiles, UB/L1 capacity/lifetimes, ownership and transfer/compute ordering. Unknown
buffer topology stays an open hypothesis. Compiler-managed synchronization must
not be mistaken for programmer-controlled scheduling. Logical programs are not
physical cores. DynamicCV/multibuffering need final metadata and probe evidence;
source locals do not establish residency. Resource overflows are compiler
failures, not numerical failures. M0 gate is correctness plus measured mechanism,
not an absolute time threshold. M1+ tunes explicit bounded levers inside this
topology; topology changes reopen the earliest M0 dependency.

Use only `[VERIFY: correctness]`, `[VERIFY: cann-logical-call]`,
`[VERIFY: msprof-direct]`, `[VERIFY: msprof-derived]`, `[VERIFY: probe]` for gates.
NCU/SASS and warp-specific SYNC/GATHER/PHASE tags do not apply. Cite actual CSV
columns, case, job, file and repetition scope. Absent counters stay unavailable.
Inferred/none lines are hypotheses outside gates. Timing alone proves neither
overlap nor stalls. Do not add kernel_details/op_summary durations or confuse a
sum/filter of device tasks with the full logical-call window. do_bench_npu,
host/event time, listener time and synchronized first-call wall are diagnostics,
not official latency. Every performance statement needs `[MEASURED, job N]` or
`[ESTIMATE]` and derivation. Autotune remains fixed/off unless explicitly bounded
by the frozen card; automatic tiling must not silently expand the search.

Preserve the injected target and `target_window: {GATE_WINDOW}`. CANN benchmark
baseline, hardware bound and scores are not the campaign reference. Cross-job
ratios are not same-job pairs. Public local evaluation is not hosted hidden-set
scoring. Cite certified recipes by name; no-go entries need evidence and changed
retry preconditions. Write the whole milestone. Final reply is one line.
