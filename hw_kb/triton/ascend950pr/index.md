# Triton Ascend950PR

Experimental target. No hardware recipes are certified in this tree yet.
Use the installed Triton Ascend API and driver-run synthetic probes. Do not
infer final DynamicCV, buffer residency, physical block count or overlap from
source-level syntax/default options. A logical grid must cover all work; it
is not a physical-core cap. Other targets are untested.

Candidate compilation metadata/IR and cold/host costs are diagnostic evidence.
Only the frozen official CANN evaluator supplies full-case correctness and the
complete logical-call device timing window. A scout or a mechanism probe does
not certify an operator gate or a performance claim.
