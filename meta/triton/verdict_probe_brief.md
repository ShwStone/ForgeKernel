## Ascend verdict experiments

Up to {MAX_VERDICT_PROBE_ROUNDS} driver-run probes can test this diagnosis.
Write `notes/<name>.py` with preregistered hypotheses/thresholds, ending only
`PROBE_REQUEST: notes/<name>.py`. Print `PROBE PASS/FAIL max_diff=<finite error>`
per contract, or n/a for compile/launch-only scope. All contracts must pass for
certification. A diagnostic FAIL is evidence; no verdict means it did not run.

Use the seeded `notes/ascend_probe.py` for topology prices. Its exact-edit JSON
spec independently builds/judges base and variant with the frozen CANN harness,
alternates repeated measurements and records casewise times/spreads/commit.
It does not modify the working kernel. Never use NVIDIA/SASS pricing helpers.
Synthetic timings, compile costs and task-duration sums are not official gates.
Cite `reopen_price: <±x%> via probe:notes/<name>.py#<index>` only for full cases.

Set `"cases": "scout"` or explicit official ids for cheaper falsification.
Narrowed experiments print PROBE SCOUT and certifies=false; they buy or kill
a full-set gate but never certify a KEEP/reopen_price themselves. Do not change
cache, evaluator, compilation policy or input set to make a price look better.
