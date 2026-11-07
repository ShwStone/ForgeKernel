## Optional NPU contract probe

Up to {MAX_VERDICT_PROBE_ROUNDS} driver-run probes may resolve uncertain M0
contracts. Write `notes/<name>.py` with a falsifiable criterion, synchronize and
print `PROBE PASS/FAIL max_diff=<finite error>` (n/a for compile/launch-only).
End `PROBE_REQUEST: notes/<name>.py`. No CUDA/NCU/SASS tooling.

Structural reopen needs a measured current-tree base/variant price, not source
inference, requested compiler flags or coincidental speed. The seeded
`notes/ascend_probe.py` prices exact edits under the frozen CANN evaluator.
Its `"cases": "scout"` option is only a narrowed diagnostic, not certification;
a full-case run still certifies the price. Compile costs and host timing are
reported separately from the full logical-call device window.
