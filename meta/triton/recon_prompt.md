# Triton Ascend reference reconnaissance

The reference runs only through the teacher-only tree `{RECON_DIR}`. Never
inspect/decode sealed payloads, keys, installed reference packages, compiler
caches, IR, binaries or tuning records. In blackbox mode source is unavailable;
record uncertainty. In source mode only explicitly allowed source may be read.
Source intent is not measurement.

Use the existing archives before requesting another job
{ARCH_REF}
{REF_BASELINE}

Window contract
{WINDOW_CONTRACT}

Read `{KB_READ_DIR}/index.md`. Write both `{RECON_REPORT}` and
`{CANDIDATES_PATH}`. Record target/toolchain/case/window/job provenance, casewise
timings and jitter, observed msprof CSV fields and their recording scope,
competing architecture hypotheses and a falsifiable distinguishing experiment.
Separate campaign reference, benchmark baseline and hardware bound.

No NCU/SASS/PTX, CUDA warp or guessed stall metrics apply. A logical Triton grid
is not physical core count. Timing does not prove overlap or exact hidden buffer
topology. Never add two profiler views or infer call counts from aggregate
repeats. Missing evidence stays unavailable. Use `[MEASURED, job N]` and
`[ESTIMATE]`; only Ascend verification labels. Listener/cold compilation and
host/event costs are not the official logical-call device window.

Optional measurement is a real `notes/<name>.py` inside `{RECON_DIR}` ending
`RECON_REQUEST: notes/<name>.py`. The driver commits/runs it. It may invoke
`bash gpu_entry.sh ref` for unchanged official reference evaluation, or an
independent synthetic NPU contract, never reference implementation access.
At most {MAX_RECON_ROUNDS} rounds. After both write-ups exist, finish with a
brief summary. Never fabricate a completed job.
