# Triton Ascend architecture agent — measured diagnosis

Read `{META_MILESTONE_PATH}`, `{MILESTONE_PATH}`, `{TRAJECTORY_PATH}` and
`{META_TRAJECTORY_PATH}`. Hardware KB is `{KB_READ_DIR}/index.md`. Allowed
source `{ORACLE}` is readable only when disclosure permits it, never in
blackbox mode. Diagnose one defect; do not patch yet.

The appended Ascend evidence digest is primary evidence. Compare KEEP/REVERT,
casewise times/jitter/gaps, official scores, candidate compilation/cache costs,
final metadata and real CSV observations. Compare matching case/window/repeat
scopes. Benchmark anchors are not the campaign reference. Missing counters are
unavailable. No NCU/SASS/CUDA/per-instruction stall channel exists. Timings alone
do not prove overlap; compiler flags alone do not prove DynamicCV or residency.
A resource compile failure, a numerical failure and a measurement-unavailable
job are different outcomes. Repeated cold compiles are exploration cost, not
necessarily slow device execution.

Construct an optimistic floor using justified hardware/shape arithmetic and
measured serial costs. Unproven serial costs count as zero. `floor_opt > target
* 1.20` is BLOCKED; otherwise OPEN, not a reachability proof. A measured reference
is an existence proof. Reconcile contradictions rather than changing the target.

Classify objectively
- G1 guidance underdetermined; G2 decision too coarse; G3 unknown hardware API;
  G4 student noise; G5 scheduling defect only if independently measured.
- R1 topology ceiling, requiring a priced earliest-M0 reopen.
- R2 missing upper-level lever, requiring an explicit decision.
- INFEASIBLE only with BLOCKED arithmetic excluding tuning and structural alternatives.

Structural/split patches require a current-tree driver probe, lever table or
certified recipe price. Use the seeded `notes/ascend_probe.py`, not NVIDIA
pricing tools. Full cases are mandatory for gate prices; scouts cannot certify.
{VERDICT_PROBE_BRIEF}

Do not repeat a reverted combination without a changed precondition. Read
bundles as bundles; propose explicit additive coupling for jointly useful levers.
Autotune policy remains frozen and bounded. Do not change the evaluator/cache.
End with the exact protocol (enum values have no trailing prose)
```
target: <metric and target>
achieved: <value and iteration>
gap_pct: <gap>
reachability: floor_opt=<number> ms vs target=<number> ms -> OPEN | BLOCKED
curve_shape: <shape>
cause_class: G1 | G2 | G3 | G4 | G5 | R1 | R2 | INFEASIBLE
evidence: <jobs, measurements, derivations>
deficient_locus: <D id or KB:mechanism>
locus_depth: foundation | upper | n/a
patch_kind: additive | structural | split | kb-example | none(student-noise)
reopen_price: <±x% via probe:notes/<name>.py#<index>; empty if no reopen>
phase_verdict: <close | continue:Dx.y, only when requested>
locus_segment: <only when measured>
patch_direction: <one actionable sentence>
confidence: high | medium | low + reason
```
