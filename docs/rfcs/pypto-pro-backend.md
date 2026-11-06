# RFC 为 Ascend 950 增加实验性 PyPTO Pro 后端

状态为提议。建议接受实验性后端接入，保留当前 PyPTO Tensor 后端。本 RFC 不宣告完整 GQA 已通过验收，也不宣告达到厂商库性能。

## 决策依据

调研核对了官方 Pro 文档、本机 0.3.0 前端及生成的 C++，随后在 Ascend950PR 上执行独立探针。PyPTO Pro 已具备表达流式 GQA 所需的主要能力，包括显式 SPMD 分工、运行时 shape/stride、片上 Cube/Vector 交接、在线统计量和可控缓冲流水。

真实硬件验证共 17 项，每项 3 次先填 NaN 再计算，共 51 次输出检查通过。最终附带的可复现脚本又完整重跑了一遍，并检查重复输出逐元素一致。

完整结果见 [机器可读证据](../../repro/ascend/pypto_pro/evidence-20261106.json)，复现方法见 [探针说明](../../repro/ascend/pypto_pro/README.md)。这些研究工具不接入 ForgeKernel 的运行路径。

## 为什么需要独立的 Pro 后端

当前 `language = "pypto"` 使用 PyPTO Tensor 的 MPMD 执行模型。它将 Tensor 图切分成设备任务，由运行时处理控制流、依赖与任务下发。源代码一次 JIT 调用，不能证明整个算法已经融合到单一片上流水。

Pro 使用面向 AI Core 的 SPMD 编程模型。`section_cube()` 和 `section_vector()` 分别描述 Cube 与 Vector 程序，同一混合 kernel 的一个逻辑执行组包含 1 个 AIC 与 2 个 AIV。开发者显式安排 Tile、缓存层次及同步，生成的 launcher 通过 `<<<>>>` 下发混合 kernel。[1][2]

本次实际生成的 FlashAttention launcher 为 `__global__ __mix__(1, 2)`。设备代码里，QK/PV 使用 L0C→UB 的 `TMOV`，P 使用 UB→L1 的 `TINSERT`，`TSTORE` 只写最终输出。函数参数没有 score/P 的 GM workspace。

采集 5 次逻辑调用得到 5 条计算任务，没有 Tensor 模式中的两条 AI_CPU 控制任务。这个结果支持将 Pro 作为新的实现选择，不能证明它对所有形状都更快。

## 已验证的表达能力

环境为 CANN 9.2.0、PyPTO 0.3.0、Torch 2.13.0+cpu、TorchNPU 2.13.0.rc1，设备为 Ascend950PR_9579。平台返回 28 个 Cube 核、56 个 Vector 核，本次最多使用 4 个混合执行组。

| 能力 | 真实 NPU 验证 | 结果与边界 |
| --- | --- | --- |
| 完整片上 attention 数据流 | 官方 FP16 非 causal FlashAttention，D=128 | 5 个形状通过，包括 34×62、162×190 尾块、512×512 多执行组、1×2048 decode |
| 在线 softmax 与循环状态 | 上述 kernel 的多个 KV tile、running max/sum/output | 与独立 CPU FP64 attention 对照，最大绝对误差不超过 2.4414e-4 |
| 原生 BSND 与分组 KV 寻址 | 指针视图、运行时 stride、`h//G`，G=1/3/4 | 4 个形状精确通过，kernel 不展开 K/V；这是寻址算子，尚非完整 GQA |
| right-down causal predicate | VF 索引、比较和 select，Sq=1/17/64、Sk=64 | 3 项精确通过，尚未合入完整 attention |
| BF16 与 256 维矩阵运算 | 17×256×33、64×256×64、96×256×80 | 3 项通过，最大绝对误差约 1.15e-5；不代表完整 BF16/D256 attention 已通过 |
| 自动 CV 流水 | 四 stage 的 `relu(A@B)@D`，preload=0 与 2 | 两种形式通过，均生成 `pipeline_generated.py`，最大绝对误差约 4.58e-5 |
| 动态 shape 编译复用 | 同一个官方 attention kernel 依次接受上述 5 个形状及不同 block_dim | 实测进程内始终只有一个已编译实例 |

官方 FlashAttention 代码保留完整预取、正反向跨核同步与多缓冲，没有改写 kernel 来迎合测试。源文件及自动流水示例通过 SHA256 固定，文档变化时复现脚本拒绝运行新内容，要求先人工检查。

本次未做完整 20-case GQA 认证。还需要把原生分组寻址、causal、BF16 与 D256 组合起来，并按原任务规则验收。17 项探针只能证明接入的表达基础成立。

## 能力的实际边界

### 存储与跨核交接

`TileType` 区分 UB、L1、L0A、L0B、L0C；`make_tile_group` 显式分配轮转槽位。Pro 可以直接表达 Q 驻留、K/V 预取、QK/PV 的 L0C→UB 和概率的 UB→L1。[2][3]

UB→L1 不会自动修正任意 layout。ND 数据需要先变成正确的 NZ/ZN 物理布局，才能供 Cube 消费。两个 AIV 写入共享 L1 时也需要约定拆分方式，不能把 `move` 当成自动完成所有转换的抽象拷贝。

### 同步与自动流水

`auto_mutex=True` 处理带 mutex 元数据的核内 Tile 依赖。它不能单独代替 AIC/AIV 的跨核同步。手工流水需要匹配的 `set_cross_core` / `wait_cross_core`，循环复用槽位必须同时保证数据就绪和消费者使用完毕。

自动 CV 流水可以根据 `stage`、`fwd_ids`、`bwd_ids` 和 `preload` 插入这些依赖，本机的 preload=0/2 已实测通过。[4]

自动流水有适用条件。stage 必须按 C/V 交替，跨核 buffer 需满足相邻 stage 的生产者消费者关系，流水循环与动态分支也有限制。复杂 causal 调度可能仍需手工流水，后端不能强迫所有 kernel 都走自动变换。

### 动态参数与编译成本

`DYNAMIC` shape、运行时 scalar、dataclass TilingData 可以复用编译实例。静态 Tile 候选和算法分支使用有限的 TilingKey/datatype 特化。Tile 的片上 shape 仍然是编译期信息。[5]

本次动态复用已经验证；TilingData/TilingKey 的组合、离线二进制和 SIMT 只核对了文档及本机接口，未做硬件验收。

JIT 缓存只保证进程内复用。跨进程会重新生成与编译，不能将 build 目录直接当成可靠的持久缓存。新的任务卡应分别记录首次调用和稳态成本，不为每个普通运行时 shape 建一个特化。

## 主机下发开销是已知问题，但尚非否决项

原生 Pro eager wrapper 的调用成本约 0.5 ms，而本次 1×2048、单头、单执行组的 FP16 attention 计算任务约 10.52 µs。两者属于不同计时范围，不能相加或混标为纯 kernel 时间。

cProfile 将大量主机时间定位到异常转储路径中的 `_build_debug_compile_cmd`。该函数每次调用都会执行构建目录 glob/scandir，并重建大多不变的调试编译命令。

最终复现脚本中的 A/B/恢复实验为以下结果。

| 路径 | 主机 callable 中位数 |
| --- | --- |
| 原始运行时 | 501.14 µs |
| 仅在独立进程里缓存静态调试命令 | 69.42 µs |
| 恢复原函数 | 482.69 µs |

每次调用的 tensor/ABI 转储元数据仍然保留，已安装包没有修改。这个实验说明相当一部分开销可以具体定位。它不构成生产缓存策略，不能将私有函数 monkeypatch 直接塞进 ForgeKernel。

接入时应把该问题登记为上游运行时改进，并核对缓存的失效条件，包括编译产物、目标架构与相关环境变化。Pro 的离线算子交付路径也可研究，但涉及额外 Host tiling/CMake 工程，初版不依赖它。[6]

前期 add 实验还验证过 NPUGraph 能减少 Pro 的 Python 路径成本，但初版必须继续报告 eager 延迟。Graph 不应被用于掩盖 eager 的限制，也不能只对候选开启而不给参考同样条件。

## 提议的 ForgeKernel 接口

新增独立语言标识，不改变旧任务的行为。

```toml
[task]
language = "pypto-pro"
arch = "ascend950pr"
```

任务目标、agent 模型配置、reference 及 budget 仍沿用当前格式。初版候选保持 `kernel.py`，封装现有任务 ABI，在内部使用 `pypto_pro.language`。`language = "pypto"` 继续表示 Tensor，不能隐式切到 Pro。

### 最小接入面

| 位置 | 拟议改动 |
| --- | --- |
| `task.py`、`taskcard.py` | 注册 `pypto-pro`、kernel 文件与 CANN 卡片生成路径 |
| `loop/lang_mode.py` | 增加独立 Pro mode、prompt overlays 与 KB 路径 |
| `profiling.py` | 按 Ascend target/capability 选择 msprof，避免与 `language == pypto` 硬绑定 |
| CANN adapter / preflight | 检查真实导入路径、发行包版本、Pro API、3510 目标和当前 stream 的核数限制 |
| build/entry | 冻结候选源码与包装 ABI，复用已有 CANN correctness/perf/scout/full-case 入口 |
| evidence | 归档生成源码、launcher、pipeline 代码、CSV、编译日志和测量范围 |

这些是计划，本 RFC 不实现上述业务改动。不需要先重写所有后端架构，再交付一个最小 Pro 路径。

初版只对通过本机能力检查的 950PR 开放。950DT 在官方支持列表中，但需要自己的硬件验证后再宣称已验证。910B/910C 不在本次 Pro 支持范围，必须明确拒绝，不能回退到 Tensor 或 CUDA。

### 知识与提示词

使用独立的 `meta/pypto-pro/` 与 `hw_kb/pypto-pro/<arch>/`。可共享硬件容量与 CANN 评测规则，不继承 Tensor 模式的任务调度经验。

最初认证的知识应覆盖以下内容。

- 混合执行组、AIV subblock 编号与一致的任务映射。
- Tile 地址、容量、有效区域和 NZ/ZN 格式转换。
- 核内 mutex 与核间正反向同步的区别。
- 在线 softmax 的 FP32 状态、尾块、causal 与最终输出边界。
- 动态编译签名、首次编译成本和稳态调用成本。

Agent 必须能读取实际生成的 `kernel.cpp` 与 `pipeline_generated.py`，不能仅靠源代码注释宣称中间量驻留片上。

## 验收与计时协议

继续复用 CANN-Bench 的任务、FP64 golden、同精度对照、checker 与完整 case 集。Scout 只能用于定价，不能产生 gate PASS；缺少测量应保留 unavailable 语义，不伪装成数值失败。

设备计时应按逻辑调用建立任务边界。需要调用延迟时，使用从首任务开始到末任务结束的设备时间窗，保留内部调度与等待。多个重叠任务的时长之和只能单列为工作量字段。

另行报告 host enqueue、同步墙钟、首次编译调用、eager/graph 与 L2-cache 策略。原始 CSV 与统计口径版本一同保留，历史 run 不追溯改分。

插桩诊断与正式计时分开。Pro sanitizer 会增加约 1 GB 日志缓冲并改变执行开销，且不覆盖全部跨核或 SIMT 访问，不能把 sanitizer 通过当成完整同步正确性的证明。[7]

每个作业都需要严格的结束清理。父进程停止后，独立 session 的评测子进程仍可能存活，须验证进程和设备占用已经回收。调试死锁不得无限占用下一轮测量设备。

## 分阶段交付

### P0 固定协议与探针

本 RFC 的机制探针已完成。进入后端编码前，还需把计时字段、进程清理和环境 preflight 的回归测试写好，冻结所用版本与实际模块指纹。

### P1 最小实验后端

支持独立 `pypto-pro` mode，交付一个完整 Vector 算子和一个 mixed Cube/Vector 算子的端到端任务卡。覆盖任务生成、远端提交、正确性、计时、证据回收及失败停止。

P1 不承诺 GQA 全量通过，也不将 Pro 设为默认后端。旧 PyPTO Tensor 的回归测试必须继续通过。

### P2 完整 GQA

将分组寻址、causal、FP16/BF16、D128/D256 合并成原生 BSND 实现，保持 K/V 不扩展。按完整 20 case、原阈值、多种输入与确认协议验收。

必须覆盖短 decode、长 prefill、causal 边界与尾块。某些 case 不通过时，保留失败记录并继续标注实验状态，不能缩小 case 集换取完成状态。

### P3 性能与生产调用

同机重新测量厂商库，保持相同缓存、预处理和分配边界。分别报告 eager 与 graph，对 Pro 的 host 调用问题采用经过验证的上游修复或正式 AOT 路径。

测量结果决定是否继续投入或替换默认后端。17 项机制验证和一个单头 attention 的低耗时，不足以批准生产替换。

## 非目标与保留判断

- 不用本 RFC 替换已停止 campaign 的成绩，也不将研究者探针计入 agent 的自主收益。
- 不保证优于 Triton Ascend 或手写 Ascend C。它们仍是有价值的独立比较对象。
- 不声称 950PR 本地 CANN-Bench 2.0.0 成绩与论文的 910C、旧版任务及托管评分等价。
- 不把示例的正确性阈值沿用到正式 GQA。正式验收始终使用冻结任务自己的规则。

表达能力已足以支持一个值得实现的实验后端。主要剩余风险在于组合后的数值正确性、同步工程、主机 wrapper 成本与完整任务集表现，均应通过上述阶段验证。

## 资料

1. [PyPTO Tensor 与 Pro 的执行模型](https://pypto.gitcode.com/guide/introduction.html)
2. [CV 融合与完整 FlashAttention 示例](https://pypto.gitcode.com/guide/programming_guide/pro/development/cube_vector_fusion.html)
3. [片上 move 与 layout 约束](https://pypto.gitcode.com/api/pro_api/SIMD-API/memory_data_movement/move.html)
4. [自动 CV 流水及使用限制](https://pypto.gitcode.com/guide/programming_guide/pro/advanced_programming/auto_parallel_pipeline.html)
5. [JIT 签名、动态维度与缓存范围](https://pypto.gitcode.com/guide/programming_guide/pro/development/compilation_and_execution/JIT_compilation.html)
6. [离线二进制交付](https://pypto.gitcode.com/guide/programming_guide/pro/development/compilation_and_execution/offline_binary_compilation.html)
7. [Pro sanitizer 的开销与覆盖范围](https://pypto.gitcode.com/guide/programming_guide/pro/debug/sanitizer.html)

文档会更新，本文硬件结论以附带证据中的 0.3.0 模块指纹与固定示例 hash 为准。
