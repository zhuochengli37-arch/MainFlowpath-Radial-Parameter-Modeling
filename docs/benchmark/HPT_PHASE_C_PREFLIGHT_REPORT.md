# HPT 正式第一轮运行前定向检查

日期：2026-10-22（Asia/Shanghai）。范围：三个特殊模型各一个真实任务、seed 42；Ma 定义核对；
2600 次正式拟合的保存与恢复检查。完整第一轮未启动，代表筛选与样本效率实验未启动。

最新结论：三个模型的真实数据链路均通过；Ma 定义已确认并关闭核查。
用户随后授权的最小续跑补丁已完成，36 项针对性测试及完整 246 项回归均通过。
原始审查事实保留如下，已由 `HPT_FORMAL_RESUME.md` 中的实现和验证结论取代；完整正式运行仍待授权。

## 三个模型的真实数据检查

三个模型统一使用 `HPTB:S2:RO:Cpt`，沿用冻结实例的 37/12/12 个工况和
1480/480/480 行。各只拟合一次，seed=42；不改变模型算法、超参数、搜索预算或随机性分类。
所有五指标均成功计算、保存，训练参数快照与阶段 B 冻结实例逐项一致。

| 模型 | 训练/上下文准备墙钟秒 | Validation 预测秒 | Test 预测秒 | 新进程恢复后 Val/Test 最大差异 |
|---|---:|---:|---:|---:|
| PySR | 559.64 | 8.88 | 8.58 | 0 / 0 |
| TabPFN | 9.95 | 14.58 | 15.57 | 0 / 0 |
| TabICLv2 | 5.69 | 100.32 | 110.54 | 0 / 0 |

这些是兼容检查记录，部分进程并行运行，不能用作论文耗时排名。
PySR 559.64 秒包含独立进程启动和 Julia 首次加载/编译；官方 runner 内记录的 fit 为 44.72 秒。
搜索配置仍为 niterations=2、populations=2、population_size=10、ncycles=20、timeout=120 秒、
maxsize=12、maxdepth=6、`+ - * /`、serial、deterministic=True，未改预算。
保留公式为 `n / (w * n)`；桥接中的兼容符号 `w` 本次明确代表 PI，不是流量。

- PySR：`.venv-pysr`，PySR 2.5.2、Julia 1.11.9、JuliaCall 0.9.36、SymbolicRegression.jl 2.4.2。
  验证前临时将原有 Julia depot 映射到短路径，完成训练和新进程恢复后已移除映射。
  版本化 Project/Manifest、模型参数和主环境均未改变。
- TabPFN：`.venv-tabpfn`，TabPFN 9.0.0、torch 2.14.0+cpu，原 v3.5-fast checkpoint；
  n_estimators=1、fit_preprocessors、CPU、离线加载。官方元数据确认 Train 为 1480 行/37 工况。
  权重 SHA-256：`14e0b81285bca7a6d1ef121169f27f651f76dff49cad6c0b8779d5c836018550`。
- TabICLv2：主 `.venv`，tabicl 2.2.0、torch 2.2.1；实际对象确认 n_estimators=8、seed=42、CPU、
  tabicl-regressor-v2-20260212.ckpt，未降低 ensemble 数量。使用已缓存权重，检查进程设 HF_HUB_OFFLINE=1。
  权重 SHA-256：`0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a`。

结果：`data/output/benchmark/preflight/hpt_stage_c/{pysr,tabpfn,tabiclv2}/`。
每个目录包含一个原始 run record、模型和 metadata、Val/Test NPZ、五指标 CSV、runtime CSV，
以及 `reports/restore_check.json`。恢复在新主进程中执行；PySR/TabPFN 又通过其隔离进程加载官方产物。
恢复后重新计算五指标，均与保存记录一致。实际预测差异均为 0；检查容差仅用于序列化复核，
没有成为模型精度筛选门槛。

汇总及补充产物 SHA-256：`data/output/benchmark/preflight/hpt_stage_c/preflight_summary.json`。
这些单 seed 记录明确标为 `preflight_single_seed`，不充当正式三 seed 比较，不用于模型筛选。

## 本轮最小代码支持

原正式 runner 仅按协议运行完整 1/3/3 repeats，不能直接满足本轮“每个特殊模型只跑 seed 42 一次”。
因此增加兼容检查专用的 `compatibility_seed` API 参数；它只允许显式小范围检查使用协议中已有 seed。
默认正式调度仍为 2600 次，冻结实例中的 model_seeds、参数和统计协议均未变。

调用脚本：

```powershell
.\.venv\Scripts\python.exe workbase/tools/regression/preflight_hpt_special_model.py --model tabpfn
.\.venv\Scripts\python.exe workbase/tools/regression/preflight_hpt_special_model.py --model tabiclv2
# PySR 需要下述已验证的短 depot 环境，再使用 --model pysr。
```

脚本复用 `run_formal_benchmark`，没有另一套训练框架。每次训练后自动启动新进程恢复。
已存在输出目录仍受原保护，不覆盖、不自动重试。可用 `--restore <原输出目录>` 单独复核保存产物。
该次特殊模型检查未实现续跑；后续授权补丁已在同一正式路径实现，见 `HPT_FORMAL_RESUME.md`。

## Ma 正式数据定义（已确认）

采用用户于 2026-10-22 明确给出的数据库定义：**Ma = CA / a**，即绝对坐标系下的有符号轴向马赫数。
CA 为绝对坐标系下的轴向速度分量，a 为当地声速；Ma 无量纲，符号随 CA。
论文结果、统计表和径向分布图统一使用此物理含义。

第二级 RO 的 66 个负值（涉及 13 工况）属于允许数值；该截面 Ma 全范围约为 -0.057123—0.506358。
源 Ma → canonical MA 原值传递，不取绝对值、不截断、不筛除、不重新计算。
本次实验唯一数据来源为正式径向分布数据库 DATABASE_HPTB；不再追溯上游 CFD 预处理或查找第三章正文。
本项已关闭，不再作为正式运行的待确认项。

机器可读数据说明：`data/output/benchmark/preflight/hpt_stage_c/ma_definition_audit.json`，
关联原冻结实例、数据和划分哈希；以补充说明方式保存，不改写冻结实例或历史拟合记录。

## 中断与恢复检查（补丁前历史记录）

| 项目 | 补丁前结论 |
|---|---|
| 模型/级/截面/目标/seed 唯一标识 | 通过：枚举得到 2600 个不同 run_id |
| 保存已完成任务的结果、模型、预测 | 支持；三个特殊模型新进程恢复已通过 |
| 根据 run records 重建 CSV、summary、manifest | 支持 `rebuild_formal_reports()` |
| 已有实验目录重新执行 | 实测抛出 FileExistsError；不覆盖原结果，但不会继续余下拟合 |
| 断点续跑 | **尚不支持**；没有读取成功记录后执行未完成项的路径 |
| 模型/metadata/NPZ 原子写入与 canonical record 中的校验和 | 尚未实现；仅原始 JSON record 使用 tmp+replace |
| 失败记录的显式重试及历史保存 | 尚无续跑语义，不能把失败项默默当作已完成 |

现有代码足以保留许多检查点并恢复查看结果，但不能据此承诺“额度耗尽或进程异常退出后可安全续跑”。
中断发生在拟合内部时，也不支持从算法内部继续训练；最小方案应以一个完整 fit 为恢复单位。

用户后续已授权并落实以下最小方案，沿用现有循环/JSON/结果重建机制：

1. runner 和 CLI 增加显式 resume 入口；核对原 experiment scope、frozen instance、数据与 split、
   model seeds/参数/环境/权重身份，禁止恢复时改变既定计划。
2. 验证成功记录的身份和产物完整性，仅复用已验证成功项；不重写其结果和模型。
3. 对失败、不完整或损坏项明确列出并保留历史，由用户显式决定重试；不自动跳过或换模型/预算。
   同一 seed 的重试不增加实验重复数，不能挑最佳结果。
4. 正式保存路径使用临时文件 + rename，记录模型/metadata/预测 SHA-256，最后提交 canonical run record。
   复用既有 JSON 原子写入函数，不引入数据库或新调度系统。
5. 恢复前确认旧进程已结束，并避免两个进程同时写同一目录；增加最小单写入保护。
6. 从每个唯一 run_id 的 canonical 记录重建统计，继续原 scheduled repeats 和 ddof=1 规则。
   增加中断后恢复、已成功项不再 fit、产物损坏、失败显式重试和统计不重复的最小测试。

涉及现有 `benchmark_runners.py`、CLI/current 脚本、正式产物保存处和测试即可。
最初检查仅报告方案；后续补丁没有添加自动跳过失败项、降采样、回退或大调度系统。

审查证据：`data/output/benchmark/preflight/hpt_stage_c/recovery_audit.json`。

## 正式命令与位置

以下为未来首次启动命令，**本轮未执行**，执行前仍需用户授权首次正式运行：

```powershell
.\.venv\Scripts\python.exe -m project1 current benchmark-3d --formal-config config/hpt_formal_benchmark.json
```

本机 PySR 验证使用临时短 depot；正式命令须在相同已验证的环境条件下运行。
可复用以下 Windows 运行环境准备方式；短盘符仅是 depot 缓存别名，不改变算法和版本化 Julia Project：

```powershell
$depot = (Resolve-Path '.venv-pysr/julia_depot').Path
$occupied = @([System.IO.DriveInfo]::GetDrives() | ForEach-Object { $_.Name.TrimEnd('\') })
$shortDrive = @('Z:', 'Y:', 'X:', 'W:') | Where-Object { $_ -notin $occupied } | Select-Object -First 1
if (-not $shortDrive) { throw 'No unused short drive' }
subst $shortDrive $depot
if ($LASTEXITCODE -ne 0) { throw 'Julia depot alias failed' }
$oldDepot = $env:JULIA_DEPOT_PATH
$oldOffline = $env:HF_HUB_OFFLINE
$oldTransformersOffline = $env:TRANSFORMERS_OFFLINE
try {
    $env:JULIA_DEPOT_PATH = $shortDrive + '\'
    $env:HF_HUB_OFFLINE = '1'
    $env:TRANSFORMERS_OFFLINE = '1'
    # 确认授权后在此执行正式命令，或本轮单模型 preflight 命令。
} finally {
    $env:JULIA_DEPOT_PATH = $oldDepot
    $env:HF_HUB_OFFLINE = $oldOffline
    $env:TRANSFORMERS_OFFLINE = $oldTransformersOffline
    subst $shortDrive /D
}
```

首次运行结果目录：`data/output/benchmark/runs/hptb_pi_first_round_v1`。
其中将有 records、models、predictions、results、reports 和 frozen_instance 快照。
后续补丁已提供 `--resume`、`--retry-runs` 和 `--recover-lock`；具体规则见 `HPT_FORMAL_RESUME.md`。
报告重建命令仍只负责重建统计，不执行训练恢复。

## 最终核验

- 全量回归 226 passed、25 warnings、0 skipped（56.03 秒）；新增两项单 seed 检查的边界测试。
- 数据/实例/split 哈希仍为阶段 B 冻结值；instance SHA-256：
  `b21082665190d5a14121de9d319d20092eef3b372abdeab93ac0bfc432f0a81a`。
- 模型实现、工厂、taxonomy、metric/repeat/sample-efficiency 协议未改。
- PySR 临时 depot 映射已清理；正式实验目录没有 experiment.json。
- Ma 定义已由用户确认并记录，原数据库、冻结实例及历史拟合记录未改。
- 定义补充后重新执行 prepare-only 通过（未训练）；三模型 12 个保存产物的 SHA-256 与原记录一致，
  数据说明/汇总/冻结实例的哈希引用一致。本次仅更新说明与补充元数据，未重跑模型或完整测试套件。
- 后续续跑补丁验证：36 项针对性测试通过，完整 246 passed、25 warnings、0 skipped（47.60 秒）。
- 下一步：等待用户授权首次正式运行。完整 2600 次拟合未启动。
