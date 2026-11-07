# Triton Ascend M0 architecture audit

The first whole-operator correctness PASS landed. Audit structure, not whether
latency meets the final target. Read `{MILESTONE_PATH}`, `{TRAJECTORY_PATH}`,
`{META_TRAJECTORY_PATH}`, `{META_MILESTONE_PATH}` and appended Ascend evidence.
Use `{KB_READ_DIR}` only for this architecture.
Campaign reference
{ARCH_REF}
Window contract
{WINDOW_CONTRACT}

Reconcile casewise correctness, observed task types/block dimensions, scoped
durations/gaps and actual candidate IR/final metadata with M0. Check logical
grid versus physical blocks, tiles, UB/L1 budgets/lifetimes, intermediate ABI,
precision, ownership, tails and compiler-managed ordering. Requested DynamicCV
or multibuffering is not a successful transformation. Source locals do not prove
on-chip residency. Compiler resource rejection is not a numerical error.
Only real CSV fields or falsifiable NPU probes verify hardware mechanisms.
Missing stalls/counters are not NCU-equivalent. Task-duration sums, listener
costs and host/event times are not the complete logical-call device window.
Do not substitute benchmark anchors for the campaign reference.

Assign each discrepancy to a reasonably priced M1 lever, a student/API defect
(G3 recipe), or a milestone defect (G1/G2/R1/R2). Choose the shallowest defect.
A structural reopen needs a measured current-tree price. If all measurable
contracts reconcile and unknowns are discharged explicitly, return OK; do not
repeat a clean audit.
{VERDICT_PROBE_BRIEF_AUDIT}

```
cause_class: OK | G1 | G2 | G3 | G4 | R1 | R2
deficient_locus: <D id or KB:mechanism; empty when OK>
evidence: <measured sources, cases, jobs; unresolved channels>
reopen_price: <±x% via probe:notes/<name>.py#<index>; required for R1>
phase_verdict: <only when requested>
locus_segment: <only when measured>
patch_direction: <single defect, or reconciliation when OK>
```
