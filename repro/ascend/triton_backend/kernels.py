"""Independent RFC probes, not a ForgeKernel backend or an optimized GQA.

Native BSND strided input addressing, shared K/V, online FP32 softmax state.
No attention library call, K/V expansion, case-ID dispatch, or autotuning here.
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def add_probe(X, Y, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=0)
    tl.store(Y + i, x + 1., i < N)


@triton.jit
def gqa_probe(Q, K, V, Y,
              SQ: tl.constexpr, SK: tl.constexpr, NQ: tl.constexpr,
              NK: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
              q0: tl.constexpr, q1: tl.constexpr, q2: tl.constexpr, q3: tl.constexpr,
              k0: tl.constexpr, k1: tl.constexpr, k2: tl.constexpr, k3: tl.constexpr,
              v0: tl.constexpr, v1: tl.constexpr, v2: tl.constexpr, v3: tl.constexpr,
              SCALE: tl.constexpr, CAUSAL: tl.constexpr,
              ST: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    u = tl.program_id(0)
    nqtiles = tl.cdiv(SQ, ST)
    s0 = (u % nqtiles) * ST
    kh = (u // nqtiles) % NK
    b = u // (nqtiles * NK)
    rm = tl.arange(0, BM)
    rd = tl.arange(0, D)
    rn = tl.arange(0, BN)
    s = s0 + rm // G
    h = kh * G + rm % G
    valid_q = (rm < ST * G) & (s < SQ)
    q = tl.load(Q + b*q0 + s[:, None]*q1 + h[:, None]*q2 + rd[None, :]*q3,
                mask=valid_q[:, None], other=0.)
    m = tl.full((BM,), -1.0e30, tl.float32)
    l = tl.zeros((BM,), tl.float32)
    o = tl.zeros((BM, D), tl.float32)
    for j in range(0, SK, BN):
        t = j + rn
        k = tl.load(K + b*k0 + t[None, :]*k1 + kh*k2 + rd[:, None]*k3,
                    mask=t[None, :] < SK, other=0.)
        v = tl.load(V + b*v0 + t[:, None]*v1 + kh*v2 + rd[None, :]*v3,
                    mask=t[:, None] < SK, other=0.)
        score = tl.dot(q, k) * SCALE
        keep = t[None, :] < SK
        if CAUSAL:
            keep = keep & (t[None, :] <= s[:, None] + (SK - SQ))
        score = tl.where(keep, score, -1.0e30)
        new_m = tl.maximum(m, tl.max(score, axis=1))
        p = tl.exp(score - new_m[:, None])
        alpha = tl.exp(m - new_m)
        l = l * alpha + tl.sum(p, axis=1)
        o = o * alpha[:, None] + tl.dot(p.to(q.dtype), v)
        m = new_m
    out = o / l[:, None]
    tl.store(Y + ((b*SQ + s[:, None])*NQ + h[:, None])*D + rd[None, :],
             out.to(Y.dtype.element_ty), mask=valid_q[:, None])


def launch_gqa(query, key, value, scale=-1., causal=False, out=None,
               tile_rows=16, tile_n=32, **options):
    """Research-only forward; intentionally reject unvalidated semantics."""
    if any(t.ndim != 4 for t in (query, key, value)):
        raise ValueError('expected BSND tensors')
    b, sq, nq, d = query.shape
    sk, nk = key.shape[1:3]
    if key.shape != value.shape or key.shape[0] != b or key.shape[-1] != d:
        raise ValueError('incompatible Q/K/V shapes')
    if min(b, sq, sk, nq, nk) < 1 or nq % nk:
        raise ValueError('nonempty inputs and integral head grouping required')
    if d not in (128, 256) or nq // nk > 32 or (causal and sq > sk):
        raise NotImplementedError('probe scope: D128/256, G<=32, causal Sq<=Sk')
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError('probe scope: FP16/BF16 only')
    if any(t.dtype != query.dtype or t.device != query.device for t in (key, value)):
        raise ValueError('Q/K/V must share dtype and device')
    if out is None:
        out = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    if out.shape != query.shape or out.dtype != query.dtype or out.device != query.device or not out.is_contiguous():
        raise ValueError('output must be a contiguous tensor matching query')
    g = nq // nk
    if tile_rows not in (16, 32, 64) or tile_n not in (32, 64):
        raise ValueError('probe tiles must use rows=16/32/64 and KV=32/64')
    st = max(1, tile_rows // triton.next_power_of_2(g))
    bm = triton.next_power_of_2(st*g)
    compiled = gqa_probe[(b*nk*triton.cdiv(sq, st),)](
        query, key, value, out, sq, sk, nq, nk, g, d,
        *query.stride(), *key.stride(), *value.stride(),
        float(scale) if scale > 0 else 1/math.sqrt(d), bool(causal), st, bm, tile_n,
        **options)
    return out, compiled
