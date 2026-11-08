# Real DeepSeek SwiGLU campaign, 2026-11-08

A real `forge new` / `forge run` campaign started from a signature-only
`NotImplementedError` stub. Both teacher and student used **`deepseek-flash`**
through `https://api.deepseek.com/anthropic`, authenticated with the operator's
DeepSeek API key. API preflight succeeded and all four agent-session transcripts
report that model. No hand-written candidate or reference source was supplied to
the student. The campaign used blackbox reference disclosure.

The run **finished at its target**. There was **one student optimization
delivery**, containing three shape-metadata-selected Triton kernels, followed by
independent complete evaluation jobs. These are not three independent model
proposals. The unmodified exported [candidate](kernel.py), exact
[vendor reference wrapper](reference.py), and
[checksummed evidence](evidence-20261108.json) are retained here.

## Final independent confirmation

All times below are microseconds. A job's reading is the geometric mean of its
20 case medians; each case has five L2-clean logical-call samples.

| Role | Job sample 1 | Job sample 2 | Job sample 3 | Median |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek candidate | 14.106278 | 14.018701 | 14.065340 | **14.065340** |
| `torch_npu.npu_swiglu` | 16.675475 | 16.694884 | 16.744622 | **16.694884** |

The comparison is **16.694884 / 14.065340 = 1.186952x**, exceeding the declared
1.05x target. The rows contain separate jobs, not paired observations. All jobs
ran on **one physical Ascend950PR**, so repetition does not establish
node-to-node robustness. The final confirmation rejected no jobs.

The complete history contains **13 full evaluations, 260/260 official case
verdicts passing**: one initial reference, one initial candidate, three fresh
reference/two candidate target rechecks, and the final three-plus-three
confirmation. Runtime preflight is excluded from those counts.

This is not a uniform per-case win. Comparing each case's three-job medians:

| Case | Candidate, us | Native reference, us | Reference / candidate |
| --- | ---: | ---: | ---: |
| 5, large FP32 | 613.344 | 519.385 | 0.846809x |
| 12, BF16 `[1000003, 2]` | 4.766 | 178.379 | 37.427402x |
| 20, FP32 rank-5 | 74.064 | 20.453 | 0.276153x |

Case 12 materially carries the aggregate gain; case 20 is a substantial
regression. All 20 cases are included, without altering weights or dropping
unfavorable results. The JSON includes every job's case times and CSV hashes.
The headline is a ratio of job medians, not a geometric mean of the displayed
casewise median ratios. The reference is the named default vendor API, not a
claim about the fastest possible CANN submission.

## Environment, cache and disclosure

- Ascend950PR_9579; physical NPU 1 exposed as logical device 0.
- Triton-Ascend 3.6.0.post2, module version 3.6.0; CANN 9.2.0;
  Torch 2.13.0+cpu; TorchNPU 2.13.0rc1; Python 3.12.15.
- Official CANN checkout `059cb61d41b279ad021160517bd2a5f6b7a05e01`, all 20
  `level1/swi_glu_*` cases. Golden, checker, tolerances and input rotation unchanged.
- Fixed configurations, no autotuning, no PyPTO/PTO or NCU. Torch in the candidate
  only allocates output or makes an optional contiguous input copy; arithmetic
  occurs in `@triton.jit` functions. Dispatch uses shape metadata, not tensor
  values, case IDs or cached outputs.
- `cann_logical_call`, derived with Decimal timestamp arithmetic from the official
  `kernel_details` CSVs. Original official scores are separate and unchanged.
- A dedicated virtual environment installs submission packages without changing
  an existing campaign's interpreter. Immutable Torch/Triton dependencies are
  reused from the installed toolchain.

The first candidate job observed 20 compiler callbacks, 12 cold lowerings and
8 disk hits. Each subsequent candidate job had **20/20 disk hits and zero
lowering time**, with the same environment and source fingerprints. Compiler
listener totals remained nonzero and are not device latency. All seven
reference jobs exported **no Triton compilation artifacts**.

The initial stub, exported candidate, raw reports, CSVs, compiler kernels,
environment/source fingerprints and transcripts have hashes in the JSON. Raw
logs/IR/CSVs/transcripts remain in the controller's
`.artifacts/deepseek-swiglu/`; no API key or reasoning transcript is published.

## Retained failure and harness fix

A teacher turn exceeded its 600-second limit after productive tool calls. The
CLI's benign custom-model warning also appeared in stderr, which caused the
old retry classifier to label this timeout an API outage. Automatic retry
completed the milestone. The live parent was not hot-patched or manually
steered, and the student's implementation was not edited by the observer.

Commit `d926818` makes launcher exit 124 consume the ordinary bounded retry
budget even when stderr also contains a model warning or timeout text. Actual
HTTP overload/stream-cut failures retain their separate outage budget.

```text
python -m pytest -q tests/test_loop_cli.py tests/test_triton_backend.py tests/test_agent_policy.py
176 passed in 13.37s

python -m pytest -q
2179 passed in 180.88s
```

The client did not recognize DeepSeek's pricing/context configuration. Its USD
field has `costBasis = unknown` and is not DeepSeek billing evidence. The JSON
retains deduplicated client token counters, not a dollar-cost claim.

## Reproduction

Use the [backend guide](../../../../docs/triton-backend.md) to configure
`local.env`, a dedicated evaluator interpreter, the pinned CANN checkout and
permitted local Triton API sources. The actual run used a 45-minute inner-loop
budget, one meta step, 600-second teacher/student turns and 1200-second full
jobs. The iteration budget is a soft admission limit: a started iteration's
full gate/reference recheck and post-stop confirmation retain their own job
bounds. Total controller time was about **158.3 minutes**.

Route the standard Claude Code backend explicitly to DeepSeek. Preserve the
run's project/local safety policy while excluding unrelated user-level provider
settings. Credentials stay in the process environment.

```bash
# Activate the controller's ForgeKernel environment first.
: "${DEEPSEEK_API_KEY:?set your own key}"
export DEEPSEEK_API_KEY
export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
export FK_PROVIDER_KEY_ENV=DEEPSEEK_API_KEY
unset CLAUDE_CODE_OAUTH_TOKEN
export ANTHROPIC_DEFAULT_HAIKU_MODEL=deepseek-flash
export ANTHROPIC_DEFAULT_SONNET_MODEL=deepseek-flash
export ANTHROPIC_DEFAULT_OPUS_MODEL=deepseek-flash
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=16384
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
mkdir -p .artifacts
printf '%s\n' '#!/usr/bin/env bash' \
  'exec claude --setting-sources project,local "$@"' > .artifacts/claude-deepseek
chmod 700 .artifacts/claude-deepseek
export FK_CLAUDE_BIN="$PWD/.artifacts/claude-deepseek"

P="$PWD/repro/ascend/triton_backend/deepseek_swiglu"
TASK="$PWD/.artifacts/deepseek-reproduction/tasks/swiglu-triton-deepseek"
python scripts/init_triton_cann_task.py "$TASK" \
  --cann-bench-dir "$CANN_BENCH_DIR" \
  --expected-cann-ref 059cb61d41b279ad021160517bd2a5f6b7a05e01 \
  --level level1 --operator-path swi_glu --operator SwiGlu --entry swi_glu \
  --signature 'input: torch.Tensor, dim: int = -1' \
  --reference-file "$P/reference.py" \
  --reference-name 'torch_npu.npu_swiglu, unmodified vendor API' \
  --dsl-source-dir "$TRITON_API_DIR" \
  --student-model deepseek-flash --teacher-model deepseek-flash \
  --student-effort high --teacher-effort high \
  --hours 0.75 --max-meta 1 --target-ratio 1.05
```

Before `forge new`, set `timeout_s = 600` in both `[agent]` and
`[agent.teacher]` of the generated task. In its card's `loop.toml.example`, set
`eval_timeout_s = 1200` and add the **absolute path of this entire dossier** to
`ref_source_paths`. This prevents a new blackbox campaign from reading the
published candidate/reference/report as a seed or hint. Do not copy `kernel.py`
into the student task. Then run `forge new "$TASK"` and `forge run <new-run-dir>`.
Model generation and timings are nondeterministic; the case set, judge and
measurement protocol are the reproducibility contract.

This run does not establish independent multi-seed, hidden-set, other-device,
node-to-node, graph or hot-cache certification. Its two unverified reference
structure hypotheses remain in the
[gap queue](../../../../docs/kb-provenance/gaps/swiglu-triton-deepseek.md), not the
hardware KB.
