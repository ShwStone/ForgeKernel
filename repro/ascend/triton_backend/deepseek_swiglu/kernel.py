"""Triton SwiGLU (`cann_bench.swi_glu`) for Ascend950PR -- milestone M0.

Architecture freeze (milestone decisions D0.1-D0.9):

* D0.1 one logical launch per call: one NPU task, no prologue/epilogue kernel,
  no workspace, no atomics, no cross-program handoff.
* D0.2 the wrapper normalizes ``dim``, derives ``O = prod(S[:a])`` and
  ``L = S[a]//2 * prod(S[a+1:])`` from shape metadata only, allocates the
  output with the input dtype/device and passes the geometry to the kernel.
* D0.3 pairing identity ``out[o,l] = silu(in[2o,l]) * in[2o+1,l]`` with the
  closed forms ``l = n mod L``, ``a_off = 2n - l``, ``b_off = a_off + L``.
  There is no sequential loop over outer groups anywhere.
* D0.4 wide-L mode: a paired-row 2D tile (BO row pairs x BL columns) with
  ``BL <= L`` and both tails masked.
* D0.5 flat mode: a 1D grid over ``n`` with the closed-form addressing above.
* D0.6 contiguous deinterleave mode: one contiguous run of ``2*BN`` input
  elements is loaded as a ``(BN, 2)`` block and split along the pair axis;
  used for ``L == 1``, where the run is exactly the row-pair stream.
* D0.7 regime selection from ``(O, L, dtype)`` only, fixed launch constants,
  no ``autotune``, no ``Config`` list, no grid clamping to a core count.
* D0.8 one pass, product in registers, every program owns a disjoint output
  tile, masked loads use ``other = 0`` and stores are masked.
* D0.9 fp16/bf16 operands are promoted to fp32, the literal
  ``x * sigmoid(x)`` is evaluated in fp32, the product is formed in fp32 and
  the single cast back to the input dtype happens at the store.  FLOAT32
  stays fp32.  No piecewise rewrite of SiLU (the golden's -Inf -> NaN
  behaviour is preserved).
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fixed launch constants (D0.7: fixed values, never autotuned).
# ---------------------------------------------------------------------------
# L at or above this selects the paired-row 2D tile (D0.4); below it the row
# pair is not a wide enough contiguous run and the flat closed-form mapping
# (D0.5) is used.  L == 1 always uses the contiguous deinterleave (D0.6).
_WIDE_MIN_L = 64
# Widest column run per program in the wide mode (keeps BL a power of two and
# BL <= L for every selected case).
_WIDE_BL_MAX = 4096
# Target elements per program in the wide mode: BO = clamp(BLOCK // BL, 1, 16).
_WIDE_BLOCK = 4096
_WIDE_BO_MAX = 16
# Outputs per program in the flat narrow mode.
_FLAT_BN = 4096
# Row pairs per program in the contiguous deinterleave mode (2*BN loaded elems).
_CONTIG_BN = 2048


@triton.jit(do_not_specialize=["O", "L"])
def _swiglu_wide(x_ptr, y_ptr, O, L, BO: tl.constexpr, BL: tl.constexpr):
    """Paired-row 2D tile (D0.4): BO row pairs x BL columns.

    ``o`` indexes the row pair, ``l`` the column inside the free (O, L) view.
    Grid axis 0 walks adjacent ``l`` tiles of one row pair so neighbouring
    programs touch adjacent addresses (D2.2 locality order); axis 1 walks the
    row-pair tiles.
    """
    pid_l = tl.program_id(0)
    pid_o = tl.program_id(1)

    o = pid_o * BO + tl.arange(0, BO)          # (BO,)
    l = pid_l * BL + tl.arange(0, BL)          # (BL,)
    mask = (o < O)[:, None] & (l < L)[None, :]

    out_off = o[:, None] * L + l[None, :]
    a_off = 2 * o[:, None] * L + l[None, :]
    b_off = a_off + L

    a = tl.load(x_ptr + a_off, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x_ptr + b_off, mask=mask, other=0.0).to(tl.float32)
    y = a * tl.sigmoid(a) * b
    tl.store(y_ptr + out_off, y, mask=mask)


@triton.jit(do_not_specialize=["N"])
def _swiglu_flat(x_ptr, y_ptr, N, L: tl.constexpr, BN: tl.constexpr):
    """Flat closed-form mapping (D0.5) for narrow, non-unit L.

    ``n`` is the flat output index; ``l = n mod L`` and ``a_off = 2n - l`` are
    the addressing identities of D0.3.  ``L`` is a constexpr so the modulo is
    a constant-divisor reduction rather than a runtime vector divide.
    """
    pid = tl.program_id(0)
    n = pid * BN + tl.arange(0, BN)
    mask = n < N

    l = n % L
    a_off = 2 * n - l
    b_off = a_off + L

    a = tl.load(x_ptr + a_off, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x_ptr + b_off, mask=mask, other=0.0).to(tl.float32)
    y = a * tl.sigmoid(a) * b
    tl.store(y_ptr + n, y, mask=mask)


@triton.jit(do_not_specialize=["N"])
def _swiglu_contig(x_ptr, y_ptr, N, BN: tl.constexpr):
    """Contiguous deinterleave (D0.6) for L == 1.

    One contiguous run of ``2*BN`` input elements starts at ``2*pid*BN``: the
    ``(BN, 2)`` block's last axis is the row-pair axis, so ``tl.split`` gives
    the two halves without any stride-2 gather.  A masked lane is a real zero
    and the store is masked.
    """
    pid = tl.program_id(0)
    n = pid * BN + tl.arange(0, BN)
    mask = n < N

    pair = tl.arange(0, 2)
    offs = 2 * n[:, None] + pair[None, :]
    v = tl.load(x_ptr + offs, mask=mask[:, None], other=0.0)
    a, b = tl.split(v)
    a = a.to(tl.float32)
    b = b.to(tl.float32)
    y = a * tl.sigmoid(a) * b
    tl.store(y_ptr + n, y, mask=mask)


def _prod(values) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def swi_glu(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """SiLU-gated split along ``dim`` (D0.2 wrapper contract).

    The output has the input's rank and dtype with ``S[dim]`` halved.  The
    input and output are viewed as ``(2O, L)`` and ``(O, L)`` respectively, and
    for every ``o in [0, O)`` and ``l in [0, L)``::

        out[o, l] = silu(in[2o, l]) * in[2o+1, l]

    Only host integer geometry is derived here; no Torch operator computes an
    output element.
    """
    rank = input.dim()
    axis = int(dim)
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise IndexError(f"Dimension out of range for dim={dim} and rank={rank}")

    shape = input.shape
    split = int(shape[axis])
    if split % 2 != 0:
        raise ValueError(f"split dimension {axis} of size {split} is not even")

    outer = _prod(shape[:axis])
    half = split // 2
    inner = _prod(shape[axis + 1:])
    rows = half * inner                      # L
    out_shape = (tuple(int(s) for s in shape[:axis]) + (half,)
                 + tuple(int(s) for s in shape[axis + 1:]))
    out = torch.empty(out_shape, dtype=input.dtype, device=input.device)

    n_out = outer * rows
    if n_out == 0:
        return out

    # The frozen cases are ND contiguous; a non-contiguous input needs the
    # copy inside the logical-call window to keep the (2O, L) view valid.
    src = input if input.is_contiguous() else input.contiguous()

    if rows == 1:
        # D0.6: contiguous deinterleave, one 2*BN-element run per program.
        grid = ((n_out + _CONTIG_BN - 1) // _CONTIG_BN,)
        _swiglu_contig[grid](src, out, n_out, BN=_CONTIG_BN)
    elif rows < _WIDE_MIN_L:
        # D0.5: flat closed-form mapping; the row pair is not a wide run.
        grid = ((n_out + _FLAT_BN - 1) // _FLAT_BN,)
        _swiglu_flat[grid](src, out, n_out, L=rows, BN=_FLAT_BN)
    else:
        # D0.4: paired-row 2D tile, BL <= L.
        bl = min(_WIDE_BL_MAX, 1 << (rows.bit_length() - 1))
        bo = max(1, min(_WIDE_BO_MAX, _WIDE_BLOCK // bl))
        grid = ((rows + bl - 1) // bl, (outer + bo - 1) // bo)
        _swiglu_wide[grid](src, out, outer, rows, BO=bo, BL=bl)
    return out
