# PyPTO Pro 表达能力探针

本目录为 [RFC](../../../docs/rfcs/pypto-pro-backend.md) 提供独立研究工具，不是 ForgeKernel 后端实现。固定日期结果见 `evidence-20261106.json`。

## 条件

需要健康且独占的 Ascend 950、与 CANN 匹配的 PyPTO Pro、Torch、TorchNPU，以及官方示例导入所需的 pytest。本次实测组合为 CANN 9.2.0、PyPTO 0.3.0、Torch 2.13.0+cpu、TorchNPU 2.13.0.rc1。

先进入对应 Python 环境并加载匹配的 CANN 环境。不同安装目录可能导入不同版本，本目录不会安装或替换软件包。

```bash
PYTHON="${FK_ASCEND_PYTHON:-python3}"
"$PYTHON" -c 'import pypto_pro, importlib.metadata as m; print(m.version("pypto"), pypto_pro.__file__)'
```

在无 NPU 的机器上也可以运行源文件校验工具的单元测试。

```bash
python3 -m unittest discover -s repro/ascend/pypto_pro -p 'test_*.py'
```

## 下载已经检查过的官方示例

从项目根目录运行。下载工具只下载并校验，不执行代码。FlashAttention 示例保留上游许可；自动流水示例来自官方文档代码块。源码 hash 变化时会拒绝继续，不自动更新固定值。

```bash
OUT="$PWD/.artifacts/pypto-pro-$(date +%s)"
SRC="$OUT/sources"
P=repro/ascend/pypto_pro
python3 "$P/fetch_sources.py" --out "$SRC"
```

下载结果包含官方 FlashAttention、preload=2 自动流水示例，以及只将装饰器改为 preload=0 的对照版本。请检查源码后再执行。`sources.json` 记录 hash 与来源。

## 硬件验证

先确认设备健康且没有其他任务。`NPU_DEVICE` 是当前可见设备中的编号，需按自己的环境设置。共享锁只协调使用同一锁文件的调用，不能替代外部设备独占管理。

```bash
NPU_DEVICE=1
LOCK="${TMPDIR:-/tmp}/forge-ascend-npu-$NPU_DEVICE.lock"
run_probe() {
    timeout -s TERM -k 20 300 flock "$LOCK" "$PYTHON" "$@"
}
run_probe "$P/validate_official_fa.py" --device "$NPU_DEVICE" \
    --source "$SRC/official_flash_attention.py" --out "$OUT/attention"
run_probe "$P/validate_capabilities.py" --device "$NPU_DEVICE" --out "$OUT/primitives"
run_probe "$P/validate_auto_pipeline.py" --device "$NPU_DEVICE" \
    --source "$SRC/auto_pipeline_serial.py" --out "$OUT/pipeline0"
run_probe "$P/validate_auto_pipeline.py" --device "$NPU_DEVICE" \
    --source "$SRC/official_auto_pipeline.py" --out "$OUT/pipeline2"
```

共有 5 个非 causal FP16 attention 形状、4 个原生 BSND 分组寻址形状、3 个 causal predicate、3 个 BF16/D256 Cube 形状，以及 2 个流水配置。每项先将输出填为 NaN，再重复计算检查 3 次，并要求重复输出逐元素相同。

Attention 与矩阵运算使用 CPU FP64 参考。寻址和 causal 探针精确比较。它们不构成完整 GQA 验收，不修改公开任务、精度阈值或 case 集。

## 主机调用归因

默认只做 cProfile 和原始调用计时。

```bash
run_probe "$P/profile_host.py" --device "$NPU_DEVICE" \
    --source "$SRC/official_flash_attention.py" --out "$OUT/host"
```

若需要复现 RFC 中的 A/B/恢复实验，显式增加 `--diagnostic-cache` 并换一个新输出目录。该选项仅在当前探针进程内缓存静态调试编译命令，保留每次调用的 tensor/ABI 转储元数据，结束时恢复原函数；不修改安装目录。这使用私有接口，只供指定版本研究，禁止作为生产后端补丁。

输出保存原始样本、CPU profile、生成代码和 profiler CSV。Event 时间可能包含主机供给空隙，不能解释为纯 kernel 时间。正式基准应另行统一缓存协议、参考调用边界与逻辑任务分组，并重新测量。
