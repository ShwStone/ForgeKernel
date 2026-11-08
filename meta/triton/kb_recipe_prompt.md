# Certify one reusable Triton Ascend mechanism

Read `{META_MILESTONE_PATH}`, `{KB_DIR}/TEMPLATE.md` and
`{KB_READ_DIR}/index.md`. Address the appended G3 diagnosis using permitted local
Triton APIs. `{ORACLE}` stays subject to disclosure. Never put an operator body,
algorithm, source paraphrase, implementation symbol or measured source footprint
in the KB. Ascend extensions are target-specific, not portable Triton facts.

Write `{KB_READ_DIR}/<mechanism>.md` with Purpose, When needed, Minimal runnable
example, Gotchas, Sources and certification scope. Use generic synthetic inputs
and real NPU Triton, not interpreter execution. Record negative controls, no-go
conditions and retry preconditions. Experiment narrative belongs only in
`{PROVENANCE_DIR}/<mechanism>.md`; agent-readable sources cite public APIs and
measured probe verdicts. DynamicCV/residency claims require actual final
metadata/IR and a discriminating probe, not default flags or variable names.

Write `{WORKSPACE}/notes/<mechanism>.py`, synchronize, compute an independent
finite error and print `PROBE PASS max_diff=<error>` or
`PROBE FAIL max_diff=<error>`. A launch/compile-only claim uses `max_diff=n/a`
and certifies no numerical/speed claim. Every printed contract must pass.
End `PROBE_REQUEST: notes/<mechanism>.py`. No verdict is a launch/build failure.
Only after a driver PASS update the index as Verified with job, probe and scope,
then end `RECIPE_CERTIFIED: hw_kb/triton/{KB_ARCH}/<mechanism>.md`.
