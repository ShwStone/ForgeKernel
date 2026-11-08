# Triton hardware knowledge

Use one tree per measured target. The experimental harness currently accepts
only `ascend950pr`. Other Ascend variants, CUDA and ROCm require independent
certification. Syntax portability is not hardware certification.

Only driver-run probes may certify a recipe. Generic syntax/documentation is
not a measured scheduling, residency or performance fact. Do not transfer
NVIDIA warp knowledge to Ascend SIMD/SIMT. Keep operator implementations out of
this tree; experiment narrative belongs in docs/kb-provenance/triton/.
