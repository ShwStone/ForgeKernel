"""Bounded PyPTO Pro expression probes; not production operator implementations."""
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf


@pl.jit(auto_mutex=True)
def strided_group_add(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, 64], pl.DT_FP32],
    k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, 64], pl.DT_FP32],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, 64], pl.DT_FP32],
):
    """q[b,s,h,:] + k[b,s,h//G,:] with native BSND strides and no KV expansion."""
    n, sq, nq, nk = q.shape[0], q.shape[1], q.shape[2], k.shape[2]
    g = nq // nk
    ntiles = (sq + 15) // 16
    tt = pl.TileType(shape=[16, 64], dtype=pl.DT_FP32,
                     target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    ag = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
    bg = pl.make_tile_group(type=tt, addrs=4096, mutex_ids=[1])
    cg = pl.make_tile_group(type=tt, addrs=8192, mutex_ids=[2])
    with pl.section_vector():
        for u in pl.range(pl.get_block_idx(), n*nq*ntiles, pl.get_block_num()):
            t = u % ntiles
            h = (u // ntiles) % nq
            b = u // (ntiles*nq)
            qv = pl.make_tensor(pl.addptr(pl.make_ptr(q), (b*sq*nq+h)*64),
                                [sq, 64], [nq*64, 1])
            kv = pl.make_tensor(pl.addptr(pl.make_ptr(k), (b*sq*nk+h//g)*64),
                                [sq, 64], [nk*64, 1])
            yv = pl.make_tensor(pl.addptr(pl.make_ptr(y), (b*sq*nq+h)*64),
                                [sq, 64], [nq*64, 1])
            a, c, d = ag.next(), bg.next(), cg.next()
            rows = pl.min(16, sq-t*16)
            pl.set_validshape(a, [rows, 64])
            pl.set_validshape(c, [rows, 64])
            pl.set_validshape(d, [rows, 64])
            pl.load(a, qv, [t*16, 0])
            pl.load(c, kv, [t*16, 0])
            pl.add(d, a, c)
            pl.store(yv, d, [t*16, 0])


@pl.vector_function
def causal_mask_vf(src, out, row, sq):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT32)
    keys = vf.arange(0, dtype=pl.DT_INT32)
    boundary = vf.full(row + 64 - sq, preg, dtype=pl.DT_INT32)
    keep = vf.le(keys, boundary, preg)
    values = vf.load_align(src, 0)
    fill = vf.full(-1.0e30, dtype=pl.DT_FP32)
    masked = vf.select(values, fill, keep)
    mask_float = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    vf.store_align(out, masked, mask_float)


@pl.jit(auto_mutex=True)
def causal_mask(
    x: pl.Tensor[[pl.DYNAMIC, 64], pl.DT_FP32],
    y: pl.Tensor[[pl.DYNAMIC, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ag = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
    bg = pl.make_tile_group(type=tt, addrs=256, mutex_ids=[1])
    with pl.section_vector():
        for row in pl.range(pl.get_block_idx(), x.shape[0], pl.get_block_num()):
            a, b = ag.next(), bg.next()
            pl.load(a, x, [row, 0])
            causal_mask_vf(a, b, row, x.shape[0])
            pl.store(y, b, [row, 0])


@pl.jit(auto_mutex=True)
def bf16_matmul_d256(
    a: pl.Tensor[[pl.DYNAMIC, 256], pl.DT_BF16],
    b: pl.Tensor[[256, pl.DYNAMIC], pl.DT_BF16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    at = pl.TileType(shape=[64, 256], dtype=pl.DT_BF16,
                     target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1])
    bt = pl.TileType(shape=[256, 64], dtype=pl.DT_BF16,
                     target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1])
    lt = pl.TileType(shape=[64, 256], dtype=pl.DT_BF16,
                     target_memory=pl.MemorySpace.Left, compact=1, valid_shape=[-1, -1])
    rt = pl.TileType(shape=[256, 64], dtype=pl.DT_BF16,
                     target_memory=pl.MemorySpace.Right, compact=1, valid_shape=[-1, -1])
    ct = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
                     target_memory=pl.MemorySpace.Acc, compact=1, valid_shape=[-1, -1])
    ag = pl.make_tile_group(type=at, addrs=0, mutex_ids=[0])
    bg = pl.make_tile_group(type=bt, addrs=65536, mutex_ids=[1])
    lg = pl.make_tile_group(type=lt, addrs=0, mutex_ids=[2])
    rg = pl.make_tile_group(type=rt, addrs=0, mutex_ids=[3])
    cg = pl.make_tile_group(type=ct, addrs=0, mutex_ids=[4])
    with pl.section_cube():
        mtiles = (a.shape[0] + 63)//64
        ntiles = (b.shape[1] + 63)//64
        for u in pl.range(pl.get_block_idx(), mtiles*ntiles, pl.get_block_num()):
            mi, nj = u//ntiles, u%ntiles
            vm, vn = pl.min(64, a.shape[0]-mi*64), pl.min(64, b.shape[1]-nj*64)
            al1, bl1, left, right, acc = ag.next(), bg.next(), lg.next(), rg.next(), cg.next()
            pl.set_validshape(al1, [vm, 256])
            pl.set_validshape(bl1, [256, vn])
            pl.set_validshape(left, [vm, 256])
            pl.set_validshape(right, [256, vn])
            pl.set_validshape(acc, [vm, vn])
            pl.load(al1, a, [mi*64, 0])
            pl.load(bl1, b, [0, nj*64])
            pl.move(left, al1)
            pl.move(right, bl1)
            pl.matmul(acc, left, right)
            pl.store(y, acc, [mi*64, nj*64])
