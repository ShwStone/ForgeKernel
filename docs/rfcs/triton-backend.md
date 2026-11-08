# RFC 增加 Triton 后端，先支持 Triton Ascend

状态为实验实现。`language = "triton"` 已有独立 agent 与运行路径，首个目标为 Ascend950PR。当前接口、计时口径与验收命令见[实现说明](../triton-backend.md)。其他 Ascend 型号、CUDA 和 ROCm 尚未认证，原有后端保持独立。

2026-11-08 已完成一个[真实 DeepSeek SwiGLU campaign](../../repro/ascend/triton_backend/deepseek_swiglu/README.md)。候选从空白 stub 生成，三加三独立完整确认得到单节点公开 case 几何均值 1.187 倍加速，逐 case 退步与超时记录均保留。

下文保留原始调研与提案，包括当时的验证范围。研究探针的通过不替代完整后端验收，也不产生优化成绩。

调研核对了官方文档、上游源码和本机实际安装的编译器及 launcher，并执行了独立 NPU 探针。判断是可以接入。主要工作在于补齐任务生成、环境检查、编译证据、预算与计时协议，现成的 Python kernel 能运行还不足以构成完整后端。

## 已有基础与当前缺口

仓库已有 `repro/ascend/public_triton/` 的 Exp、SwiGLU 实现，以及 `run_public.sh` 的 CANN-Bench 提交入口。这说明 Python 包装和官方 evaluator 可以复用。

Agent 路径还没有 Triton。`task.py` 未注册该语言，`taskcard.py` 没有对应生成分支，`loop/lang_mode.py` 会将陌生语言归入 CuTe mode。Ascend profiler 的选择只识别 `pypto`，CANN preflight 也硬编码导入 PyPTO。仅增加语言字符串，会让 Triton agent 继续读取错误的提示词并走错运行路径。

与 [PyPTO Pro RFC #3](https://github.com/ShwStone/ForgeKernel/pull/3) 的关系也需要明确。Triton 提供较高层的张量块程序，主要依赖编译器安排 Cube/Vector 计算、存储和同步。Pro 允许开发者显式安排 Tile 与跨核流水。两条路径可以共用 CANN 评测设施，性能结论分别认证。

## 调研与实测环境

本次实测为 CANN 9.2.0、Triton-Ascend 3.6.0.post2、模块自报 Triton 3.6.0、Torch 2.13.0+cpu、TorchNPU 2.13.0.rc1，硬件为 Ascend950PR_9579。运行时确认 target 为 `npu`，没有使用 CUDA 或 CPU interpreter 执行候选。

参考上游源码固定为 `ae07d3d13917a53300bb74127604242f7bf02f2e`。这个 SHA 用于所查资料，不冒充已安装 wheel 的构建 SHA。安装包源码另存 SHA256，版本、路径、默认选项及实际 metadata 见 [机器可读证据](../../repro/ascend/triton_backend/evidence-20261107.json)。

探针及复现命令见 [研究工具说明](../../repro/ascend/triton_backend/README.md)。本次不安装软件、不改已停止的 campaign，也不修改 ForgeKernel 业务代码。

## Triton Ascend 实际提供什么

### 程序与执行模型

标准 `@triton.jit`、`tl.program_id`、带 mask 的访存、归约、`tl.dot` 和 kernel 内循环足以表达在线 softmax GQA。普通路径大致为 TTIR→Linalg/adapter IR→Ascend NPU IR→设备二进制，launcher 调用 CANN 的 `aclrtLaunchKernelWithHostArgs`。[1]

本机实际保存了 `ttir`、`ttadapter`、`mlirbc`、`bcmlir` 和 `npubin`。GQA 的 TTIR 中有一个 KV 循环、两处 dot 和一次最终 store。采集的两组 GQA 均为 5 次调用对应 5 条计算任务。

这些证据支持循环在 kernel 内执行。它们没有证明所有中间量都留在片上，也没有证明运行时没有 workspace、重排或其他成本。后端必须让 agent 能读取实际编译产物，不能用源代码中的局部变量名判断驻留位置。

### 逻辑 grid 与物理核心

旧经验中的 grid 限制不能直接套到当前版本。本次默认路径以 73 和 70,001 个逻辑 program 执行 Vector add，均精确覆盖全部输出，profiler 的 `Block Num` 均为 56。

这验证了当前工具链的自动工作分配。grid 大小既不等于物理参与核数，也不能证明满利用率。后端不能统一在 host 把 grid 截成核数上限，否则可能直接丢失工作。逻辑 program 应保持相互独立，跨 program 的执行顺序或全局同步需要另行验证。[2]

### 编译模式和流水选项

950 提供 `simd`、默认 `simd_simt_template` 和 `simt_only`。默认混合模式主要让规则访问走 SIMD，在适用的离散访问处使用 SIMT 模板。本次 Vector add 在三种模式均通过；GQA 只验证了默认模式。[1][3]

`num_warps` 的纯 SIMT 含义不能直接当成 SIMD 的并行度。当前 `num_stages=1` 关闭自动多缓冲，其他值使用自动策略，也不等价于精确指定各层 buffer 数。`multibuffer_mode` 与显式 `num_stages` 不能同时设置。[3]

DynamicCV 默认会尝试优化适用的 Cube/Vector 程序，失败时可以回退。本次两组 GQA 的最终 metadata 均为 `enable_dynamic_cv_pipeline=false`，尽管目标默认选项会尝试启用。因此不能以传入 True 或默认值为依据，宣称流水变换已经成功。

### 编译缓存与可观测性

Triton 有进程内 JIT 缓存和磁盘编译缓存。本机 `triton.knobs.compilation.listener` 能观察磁盘命中，并报告分阶段耗时。它不会替代完整 driver 初始化、launcher 构建和首次加载的墙钟测量。[3]

下表为同一独立 Vector add 的新进程实验。缓存目录在本研究内隔离。

| 条件 | 磁盘命中 | lowering 总耗时 | 首次调用含加载与同步 |
| --- | --- | --- | --- |
| 首次编译，BLOCK=1024 | 否 | 459.8 ms | 1.998 s |
| 新进程复用同一缓存 | 是 | 0 | 0.650 s |
| BLOCK 改为 512 | 否 | 468.3 ms | 1.042 s |
| 编译模式改为 simd | 否 | 455.7 ms | 1.160 s |
| 编译模式改为 simt_only | 否 | 275.7 ms | 2.042 s |

复用命中时 hash 相同，tile 或 mode 改变后 hash 不同。表中首调用包含填 NaN 和同步，不含进程启动及前置环境检查。缓存有助于控制重复编译，但不能据此宣称首次调用免费，或推断完整 20-case 的编译成本。

## GQA 表达能力验证

研究用 kernel 直接读取原生 BSND 和共享 K/V，没有展开 K/V、调用库 attention、依据 case ID 返回答案或自动搜索配置。stride 按编译期参数特化，输出预分配。

| 验证集 | 覆盖内容 | 结果 |
| --- | --- | --- |
| 8 个补充形状 | FP16/BF16、D128/D256、G=1/4/12/32、非对称序列、尾块、decode、causal、显式 scale、非连续输入 | 3 个 seed，每输入 3 次 NaN 污染检查，72/72 通过 |
| 官方 case 1/2/5/6/18/20 | 含 D256 causal、G12、MQA 和 BF16 显式 scale | 3 个 seed，每输入重复 3 次，原 checker 54/54 通过 |

合计 14 个形状、42 组输入、126 次输出检查，重复输出逐元素一致。官方子集使用固定 CANN-Bench SHA 的 golden、CPU 同精度参考、原阈值与 `RelativeErrorChecker`，并检查实际导入路径属于该 checkout。

补充形状使用独立 CPU FP64 oracle 和显式 allclose 阈值。它们是机制诊断，不能取代正式任务规则。完整 GQA 仍需官方 20-case evaluator、多输入与确认流程；上述六个官方 case 的通过不能产生 gate PASS。

### 保留一次真实失败

首个尾块形状为 B=1、Sq=17、Sk=33、Nq=8、Nkv=2、D=128。使用 64 行、KV tile=64 时，编译器报告 UB 需要 5,527,552 bits，可用 1,769,472 bits。仅将 stride 特化并未修复它。

把目标行块改为 16、KV tile 改为 32 后，同一 kernel 表达通过了上述验证集。G=32 时实际行块至少为 32，以容纳分组头。

这说明较小 tile 能避开本次资源超限。它不证明当前表达已调优，更不支持把所有编译失败归因于 DSL 表达能力不足。编译错误必须保留资源报告，并计入探索成本。

## 主机、设备和 benchmark 语义

28×1024 FP32、预分配输出的 `y=x+1`，本次得到以下诊断中位数。

| 路径 | host callable | event stream | profiler 单任务 |
| --- | --- | --- | --- |
| eager | 25.27 µs | 27.11 µs | 1.139 µs |
| NPUGraph replay | 5.86 µs | 14.88 µs | 1.243 µs |

Graph 在两次改变输入、输出先填 NaN 的检查中均精确正确。单节点 graph 有自己的执行成本，不能承诺比 eager 的设备任务更快。Event 还可能包含主机供给空隙。

`do_bench_npu` 的源码和实测也需要一起看。它默认返回设备任务时长的均值，可用 mspti 或 torch_npu profiler；设置保留目录、结果或精确 kernel 名会改变收集路径。本次强制 profiler 时，热缓存均值约 1.424 µs，逐次 L2 清理约 2.274 µs，两种协议必须分开报告。[4]

该 helper 对多任务调用会按任务数倍率累计时长，不能自动变成逻辑调用的首尾时间窗。重叠任务的 sum、内部有空隙的多 kernel 调用、只过滤一个 kernel 名的结果，都不能直接替代完整算子的延迟。附带统计工具只接受已验证的每调用单任务数据，遇到部分捕获或多任务会拒绝。

本次没有新做厂商库 A/B，也没有宣布 GQA 性能收益。后端接入后的正式计时应按逻辑调用取完整设备起止窗，保留调度等待，另存任务时长 sum、host、同步 wall、首次编译及缓存协议，不追溯改写历史分数。

## 提议接口与接入边界

预期新增的任务写法如下，目前尚未实现。

```toml
[task]
language = "triton"
arch = "ascend950pr"
```

语言标识描述 DSL，实际硬件和冻结环境确定 compiler/launch backend。初版只认证 Ascend950PR。文档支持的 A2/A3、950DT 和未来 CUDA 目标，需要各自的真实验证；不因语法相近就自动标为支持。[5]

| 接入位置 | 拟议改动 |
| --- | --- |
| `task.py`、`taskcard.py` | 注册 triton 与 Python kernel ABI，增加任务卡生成分支 |
| `loop/lang_mode.py` | 独立 Triton mode，避免落入 CuTe 默认路径 |
| CANN entry / preflight | 提取共享 Ascend 路径，按语言检查 Triton 和实际 NPU target，不强制安装或导入 PyPTO |
| `profiling.py`、`arch.py` | 依据 target/capability 路由 profiler 与 KB，NPU 不得进入 NCU；未来 CUDA 另行认证 |
| build / cache | 打包冻结 Python 源码，首次调用 JIT；保存环境、签名、选项与缓存命中信息 |
| evidence | 回收阶段耗时、最终 metadata、IR、二进制、CSV、失败 stderr 与逐 case 成本 |

复用已有 CANN golden、checker、scout/full-case 门禁、密封 reference 和测量 unavailable 语义，不复制一套宽松 evaluator。不要让 Triton 依赖 PTO Tile 库才能通过 preflight。

使用 `meta/triton/` 和 `hw_kb/triton/<arch>/`。通用 Triton 语法可以共享，NPU 的存储、布局、流水和同步经验与 NVIDIA warp 经验分开。Ascend 专有扩展只能进入对应 target 的知识与能力表，不能伪装成可移植 API。

## 缓存与 autotune 的约束

缓存按环境指纹、目标、源码和特化参数隔离。指纹应包含实际 Triton/driver/compiler 路径与 hash、CANN/BiSheng、Python/TorchNPU ABI 和编译选项。不能假定同一个版本字符串已经覆盖所有本地补丁与工具链变化，也不清空其他作业的全局缓存。

候选与密封 reference 使用不同角色目录。Reference 的 TTIR、二进制和调优记录不得因共享缓存或统一收集编译产物而暴露给学生。Collector 只接候选 worker 的 listener，不能把同进程里的 reference 编译也收进学生可读目录。

初版默认固定配置，显式开启有界搜索后才允许 autotune。本次两个手写 Config 均被 benchmark，第二次调用复用选型，数值正确。这个实验没有验证自动 tiling 生成的质量。

Ascend 的 `configs=[]` 自动 tiling 需要额外 runtime 扩展导入，能生成候选并并行编译；这会显著改变成本和行为，不能无声开启。[6]

搜索要限制候选数、并行度、逐 case 与全作业墙钟，计入所有候选编译和 benchmark。选型 key 包含必要的 shape、dtype、layout 和属性，不得读取输入值、随机 seed 或 case ID 来选择答案。正式 gate 使用已经冻结的配置策略。

带原地写入、atomic 或累加副作用的 kernel 需要 reset/restore hook，之后还要按原规则检查最终输出。Autotune 的重复执行不能污染下一次参考比较。

## 验收顺序

### P0 先固定运行协议

补齐 target preflight、缓存角色隔离、逻辑调用计时与跨 session 子进程回收测试。TERM→KILL 后验证进程及设备占用消失，避免预算耗尽后留下工作。

编译 UB 超限保留为编译失败，数值失败保留为 correctness failure。环境缺失、profiler 缺证据和预算中断各存自己的状态，不生成零耗时或伪造 PASS。

### P1 最小实验后端

将现有 Exp/SwiGLU 或等价 Vector 任务接入完整 agent 路径，验证任务生成、远端提交、完整正确性、计时、证据、确认及回滚。增加一个含 dot/归约的 mixed kernel，验证编译产物可供 agent 阅读。

### P2 完整 GQA

用原生 BSND、共享 K/V，通过冻结的 20-case 官方 evaluator、多输入和确认任务。完整记录 cold/warm 编译与执行成本，覆盖长 prefill、大 batch decode、causal 和 D256，不能把本次小 tile 的有限验证外推到所有 case。

### P3 公平性能与其他目标

同机重测库参考与候选，交错执行，统一 dtype、输入、布局、准备和分配边界。Eager/graph、热缓存/L2-clean 分表报告。再决定是否投入调优，并为其他 Ascend 型号或 CUDA 增加独立测试。

本 RFC 只提议接入和上述验收路线，不启动新的自主 campaign，不替换默认后端，也不宣称论文同口径成绩或已达到厂商库性能。

## 资料

1. [架构、编译模式与 DynamicCV](https://github.com/triton-lang/triton-ascend/blob/ae07d3d13917a53300bb74127604242f7bf02f2e/docs/en/architecture_design_and_core_features.md)
2. [逻辑 grid 与自动 block 分配](https://github.com/triton-lang/triton-ascend/blob/ae07d3d13917a53300bb74127604242f7bf02f2e/docs/en/migration_guide/architecture_difference.md)
3. [编译选项、cache 和 listener](https://github.com/triton-lang/triton-ascend/blob/ae07d3d13917a53300bb74127604242f7bf02f2e/docs/en/environment_variable_and_compiler_options_reference.md)
4. [do_bench_npu 文档](https://triton-ascend.readthedocs.io/en/v3.6.0/debug_guide/do_bench_npu_guide.html)
5. [支持环境与安装说明](https://github.com/triton-lang/triton-ascend/blob/ae07d3d13917a53300bb74127604242f7bf02f2e/docs/en/installation_guide.md)
6. [Ascend autotune 扩展](https://github.com/triton-lang/triton-ascend/blob/ae07d3d13917a53300bb74127604242f7bf02f2e/docs/en/autotune_guide.md)
7. [官方 fused attention 示例](https://github.com/triton-lang/triton-ascend/blob/ae07d3d13917a53300bb74127604242f7bf02f2e/third_party/ascend/tutorials/06-fused-attention.py)

滚动文档的概括不能替代实测。本文关于 cache 命中、UB 拒绝、DynamicCV 回退及计时的判断，以附带证据中的安装包指纹与原始观测为准。
