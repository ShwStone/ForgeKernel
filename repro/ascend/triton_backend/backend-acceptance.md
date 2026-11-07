# Backend acceptance, 2026-11-07

Implementation and fixes are recorded through `9166dd2`. The
[machine-readable record](backend-acceptance-20261107.json) includes every case
verdict, CSV hash, raw report hash, harness hash, environment fingerprint and
compilation summary. The earlier [research evidence](evidence-20261107.json)
and `kernels.py` are unchanged.

## Completed checks

| Check | Result |
| --- | --- |
| CPU suite, `python -m pytest -q` | 2177 passed in 169.37 s |
| Generated Exp candidate, two complete jobs | 20/20 each |
| Separate sealed Exp diagnostic reference | 20/20; no compilation artifacts exported |
| Generated GQA candidate with probability residual | 20/20 |
| Separate sealed CANN library GQA reference | 20/20; no compilation artifacts exported |
| Corrected GQA case 15 scout | 1/1; explicitly rejected as an operator gate |
| Final decimal-timestamp Exp scout | 1/1; explicitly rejected as an operator gate |
| Real NPU escaped-worker cleanup | TERM-ignoring orphan killed/reaped; identity gone; fresh allocation/synchronization succeeded |

The NPU is Ascend950PR, target `npu`, with Triton-Ascend 3.6.0.post2, CANN 9.2.0,
Torch 2.13.0+cpu and TorchNPU 2.13.0.rc1. The evaluator is pinned to
`059cb61d41b279ad021160517bd2a5f6b7a05e01`. Its golden, checker, thresholds and
input rotation are unchanged. Every full job's case IDs match its frozen card.

The first complete Exp job observed 20 compiler callbacks over six unique
kernels: six cold lowerings and 14 disk hits. A second fresh-process job had
20 disk hits and **zero lowering time**, with the same source/environment
fingerprints. Listener time remained nonzero and is not first-call wall time.
Candidate IR, binaries and final metadata were recovered. Full GQA final
metadata contains both enabled and disabled DynamicCV cases; there is no
blanket pipeline/residency certification.

The cleanup worker held a real NPU tensor, started a new session, became an
orphan, ignored TERM and closed stdout/stderr. The frozen entry's subreaper
found and killed it. A separate job verified that PID/start-time identity was
gone and a new NPU sum returned 4. This checks owned-process cleanup and device
health, not total system-wide NPU utilization or memory counters.

## Failures retained and resolved

The original RFC GQA kernel passed 19/20 full cases. Case 15 failed the official
FP16 check with MERE 0.001638 and four near-zero errors out of 9,854 small-value
elements. Its complete job supplied **no accepted aggregate timing**.

The acceptance fixture preserves the probability rounding residual in a second
low-precision dot, for all cases rather than a case-ID branch. No tolerance
changed. The case-15 scout then had MERE 1.994e-7 and zero small-value errors;
only the subsequent complete 20-case job established full-task correctness.
The original dated probe/evidence was not rewritten.

Two early harness prototypes also failed: a wrapper let `eval/cann_bench.py`
shadow the installed submission, and the upstream child CLI rejected a custom
strategy name. Both were removed. The final path executes the unchanged official
CLI with `kernel_details`, then derives the separately labelled call window.

Raw profiler timestamps exceed 1e15 microseconds. Binary-float arithmetic lost
up to 0.123 microseconds when adding/subtracting these timestamps. The final
parser subtracts decimal timestamps before converting differences to floats.
All archived CSVs were checked with this parser; original official scores and
prototype normalized values remain intact in the record. A fresh final-path
NPU scout reported 3.639 microseconds, with individual task durations and call
windows agreeing exactly. This is a parser correction, not a new benchmark score.

## Reproduction and limits

Run the commands in [the backend guide](../../../docs/triton-backend.md), using
a fresh output directory. Logs and full checksummed artifacts remain in the
controller's `.artifacts/acceptance/`; they are not bundled into git. The JSON
here records their hashes and bounded summaries. `compileall`, `bash -n` and
`git diff --check` also passed.

These are fixed diagnostic kernels, not an autonomous optimization campaign.
The GQA fixture is slower than the library and is not a released performance
kernel. Inputs use the evaluator's default deterministic seeds and pointer
rotation; independent multi-seed full-task certification and actual three-plus-
three campaign confirmation/promotion remain untested. Their mandatory guards
have CPU regression coverage. No CUDA/ROCm/other-Ascend, graph, hot-cache,
hidden-set or vendor-speedup result is claimed. Jobs sharing an evaluator
interpreter must be serialized because the official package installer uses the
global `cann_bench` package.
