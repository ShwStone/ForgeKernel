# Triton Ascend finish-time knowledge harvest

Read `{META_MILESTONE_PATH}`, `{KB_DIR}/TEMPLATE.md`, `{KB_READ_DIR}/index.md`
and the appended measured probe history from `{WORKSPACE}`. Run NO NPU jobs.
Only mechanisms certified by completed NPU probes enter the KB; a correct/fast
operator, a compiler option or a profiler signature alone is not certification.

Write genuinely reusable entries at `{KB_READ_DIR}/<mechanism>.md` with Purpose,
When needed, Minimal runnable example extracted from the tested probe, Gotchas
and Sources. Include exact architecture/software scope, trigger symptoms,
negative controls, no-go/retry preconditions and job + probe + verdict. Keep
Ascend-specific layouts, SIMD/SIMT and synchronization separate from NVIDIA
warp facts. Do not label untested targets as portable/certified.

No operator algorithms, source symbols/paraphrases, run IDs or commit hashes go
in KB files. Narrative belongs only in `{PROVENANCE_DIR}`. `{ORACLE}` remains
subject to disclosure. Skip uncertain mechanisms rather than inventing evidence.
End `RECIPE_RECORDED: hw_kb/triton/{KB_ARCH}/<mechanism>.md` per updated recipe,
or `NO_RECIPE: <reason>`. Never request a probe in this turn.
