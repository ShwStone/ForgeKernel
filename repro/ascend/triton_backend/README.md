# Triton Ascend 后端可行性探针

本目录为 [RFC](../../../docs/rfcs/triton-backend.md) 提供独立研究代码，没有接入 ForgeKernel 的任务、评分或 agent 路径。固定日期数据见 `evidence-20261107.json`。

实测环境为 Ascend950PR、CANN 9.2.0、Triton-Ascend 3.6.0.post2、Torch 2.13.0+cpu、TorchNPU 2.13.0.rc1。CUDA、ROCm 和其他 Ascend 型号尚未验证。

## 先检查环境

进入自己已经配置好的 CANN/TorchNPU 环境，使用对应的 Python。本目录不会安装或替换软件包。

```bash
PYTHON="${FK_ASCEND_PYTHON:-python3}"
"$PYTHON" -c 'import triton, importlib.metadata as m; print(m.version("triton-ascend"), triton.__version__, triton.__file__)'
```

探针先选择指定设备，再确认运行时 target 为 NPU。实际导入路径、模块 hash、默认编译选项和最终 metadata 都会保存。不要仅凭模块自报的 Triton 版本判断安装内容。

无 NPU 的机器也能运行统计工具的 CPU 测试。

```bash
python3 -m unittest discover -s repro/ascend/triton_backend -p 'test_*.py'
```

## 独立执行

从项目根目录运行。先确认设备健康且空闲，设置当前可见设备中的编号；容器的逻辑编号可能不同于物理编号。锁只能协调使用同一锁文件的进程，不能代替设备独占管理。

```bash
P=repro/ascend/triton_backend
OUT="$PWD/.artifacts/triton-backend-$(date +%s)"
NPU_DEVICE=1
LOCK="${TMPDIR:-/tmp}/forge-ascend-npu-$NPU_DEVICE.lock"
run_probe() {
    timeout -s TERM -k 20 600 flock "$LOCK" "$PYTHON" "$@"
}
```

每个输出目录应当是新的。脚本保留 CSV、编译产物和失败记录，不将失败伪装成零耗时。超时后仍需检查编译器或设备作业是否存活。

### 编译缓存、grid、graph 与有界调优

```bash
CACHE="$OUT/shared-cache"
run_probe "$P/runtime_probe.py" --device "$NPU_DEVICE" --cache "$CACHE" \
    --out "$OUT/cold" --advanced
run_probe "$P/runtime_probe.py" --device "$NPU_DEVICE" --cache "$CACHE" --out "$OUT/reuse"
run_probe "$P/runtime_probe.py" --device "$NPU_DEVICE" --cache "$CACHE" \
    --out "$OUT/tile-change" --block 512
run_probe "$P/runtime_probe.py" --device "$NPU_DEVICE" --cache "$CACHE" \
    --out "$OUT/simd" --mode simd
run_probe "$P/runtime_probe.py" --device "$NPU_DEVICE" --cache "$CACHE" \
    --out "$OUT/simt" --mode simt_only
```

脚本显式设置本进程的 `TRITON_CACHE_DIR`，不使用环境里可能已有的共享路径，也不清除其他作业的缓存。`--cache` 未指定时使用输出目录下的 cache。

`--advanced` 检查 73/70,001 个逻辑 program 的结果及实际启动规模、改变输入后的 graph replay、L2-clean 与热缓存的 `do_bench_npu`，以及两个手写 Config 的选型和复用。各编译模式的测试仅为 Vector add。

### GQA 数值诊断

默认执行 8 组补充形状，3 个 seed，每组输出先填 NaN 后检查 3 次，共 72 次。包含尾块、causal、decode、FP16/BF16、D128/D256、非连续输入与不同 GQA 分组。

```bash
run_probe "$P/validate_gqa.py" --device "$NPU_DEVICE" --out "$OUT/gqa-synthetic"
```

CPU FP64 oracle 只在验证侧展开 K/V；NPU 候选直接读共享 K/V。Synthetic 使用 FP16 的 rtol/atol=0.005、BF16 的 rtol/atol=0.02，不能将这些诊断阈值沿用到正式任务。

可选地加入固定 CANN-Bench 的 6 个官方 case。需要对应 checkout 的 `kernel_eval` 和 PyYAML 已在当前环境可导入。

```bash
# CANN_BENCH_DIR 指向 059cb61d41b279ad021160517bd2a5f6b7a05e01
run_probe "$P/validate_gqa.py" --device "$NPU_DEVICE" --out "$OUT/gqa-with-official-subset" \
    --cann-bench "$CANN_BENCH_DIR" --case-ids 1,2,5,6,18,20
```

这会运行全部 14 组形状、42 组输入、126 次检查。官方子集使用其 golden、CPU 同精度输出、原阈值与 checker；脚本验证 checkout SHA 和实际模块位置，并保存文件 hash。它没有执行完整官方 evaluator，不产生 gate PASS。

默认目标行块为 16，KV tile 为 32，stride 为 constexpr 特化。可用新的输出目录和 `--tile-rows 64 --tile-n 64` 复现本机第一组尾块的 UB 编译拒绝；这应当记录为失败，不能据此删掉该形状。

## 结果边界

编译 listener 的耗时、首次调用墙钟、host callable、event stream 与 profiler task duration 属于不同范围。多任务或部分捕获的 CSV 会被统计工具拒绝，不能通过时长求和冒充延迟。

Graph 与 benchmark 只在单独的 Vector add 上验证。GQA 的两个 profiler 样本只用于核对任务数，不构成正式性能认证，也没有本轮厂商库 A/B。

这些脚本只用于独立研究。正式后端仍需角色隔离的缓存与 artifact 收集、完整 case 门禁、计时协议、失败成本和进程回收。
