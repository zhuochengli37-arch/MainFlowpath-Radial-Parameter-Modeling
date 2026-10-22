# HPT 正式 benchmark：中断恢复补丁

日期：2026-10-22。补丁只扩展现有正式 runner、CLI、产物提交和报告重建路径。
没有新调度框架或任务数据库；完整 2600 次正式拟合尚未启动，仍等待用户授权。

## 修改文件

| 文件 | 本次续跑修改 |
|---|---|
| `workbase/src/project1/experiments/benchmark_runners.py` | 原正式循环内增加恢复、计划一致性、显式重试和事务提交顺序 |
| `workbase/src/project1/experiments/benchmark_instance.py` | 原子 JSON 工具旁补充文件哈希、运行环境身份、锁、记录校验及历史归档 |
| `workbase/src/project1/experiments/benchmark_reporting.py` | 校验 canonical 记录与产物，拒绝重复计数，明确全组/全实验完整状态 |
| `workbase/src/project1/cli.py`、`workbase/current_scripts/benchmark_3d.py` | 显式转发恢复、精确重试和残留锁确认参数 |
| `tests/test_formal_benchmark.py` | 小规模续跑、保存中断、损坏、失败、锁、身份和统计测试 |
| `.gitignore`、`.gitattributes` | 忽略运行锁；canonical 记录保留原始字节，防止 Git 换行转换改变审计哈希 |
| 本文、`HPT_PHASE_C_PREFLIGHT_REPORT.md` | 当前恢复规则、命令和已完成检查 |
| `data/output/benchmark/preflight/hpt_resume/` | 验收与本地运行环境身份；此前 preflight_summary 链接新验收 |
| 工作区 `05_DOCS/CURRENT_PROJECT_STATUS.md` | 跨会话交接与 Git/测试事实（位于 DEV Git 仓库之外） |

一并整理的阶段 B 变更（PI 解析、原 GPR 限制移除、正式配置/实例、结果字段和先前兼容证据）
详见 `HPT_PHASE_B_REAL_DATA_INTEGRATION.md`；本次没有再次修改数据库或模型配置。

## 首次运行与恢复命令

工作目录：`01_DEV/Project1`，使用主 `.venv`。本机 PySR 仍需先按
`HPT_PHASE_C_PREFLIGHT_REPORT.md` 准备已验证的临时短 Julia depot 和离线环境。
以下训练命令仅为使用说明，本轮不执行。

```powershell
# 只读核对冻结实例，不训练
.\.venv\Scripts\python.exe -m project1 current benchmark-3d --formal-config config/hpt_formal_benchmark.json --prepare-only

# 首次运行：没有 --resume，不覆盖已有实验
.\.venv\Scripts\python.exe -m project1 current benchmark-3d --formal-config config/hpt_formal_benchmark.json

# 恢复：读取原实验的 scope、run_plan 和原 seed，不传 --models/--cases
.\.venv\Scripts\python.exe -m project1 current benchmark-3d --formal-config config/hpt_formal_benchmark.json --resume

# 根据 reports/resume_issues.json 人工检查后，明确授权指定 run_id 重试
.\.venv\Scripts\python.exe -m project1 current benchmark-3d --formal-config config/hpt_formal_benchmark.json --resume --retry-runs HPTB_S2_RO_Cpt__random_forest__seed_2026
```

固定正式输出目录：`data/output/benchmark/runs/hptb_pi_first_round_v1`。
小范围兼容测试使用独立 `--output`；恢复该测试时用相同 `--output --resume`，不重新传选模或 seed。

## 一致性与运行清单

原冻结实例不改写。每次运行重新检查原始文件哈希、61 工况、37/12/12 划分、40 任务、
35 模型工厂参数、taxonomy、指标和重复协议，必须与原实例完全相同。
`experiment.json` 新增以下运行侧元数据，独立于原数据实例哈希：

- `resume_format=1`，明确区分补丁前仅支持查看结果的历史兼容记录；不自动迁移历史记录。
- 有序 `run_plan`：每项含 run_id、case_id、model_key、model_seed、repeat_index。
  正式模式仍为 800 + 1680 + 120 = 2600 个唯一 run_id。
- `runtime_identity`：主环境 Python 和全部安装包版本、PySR/TabPFN 独立环境版本，
  Julia 可执行文件及 Project/Manifest 哈希，TabPFN/TabICLv2 本地权重 SHA-256，
  相关模型/benchmark 源文件哈希与线程环境变量。
- `contract_sha256` 关联上述配置、清单和运行环境；每个 canonical record 引用相同标识。

恢复时重建并逐项比对完整实验信息和实验内 frozen_instance 快照。数据、split、scope、
参数、重复次数、环境或权重变化均在 fit 前报错；不接受恢复命令重新选模型、任务或 seed。
本地权重身份检查只读取缓存，不训练、不联网、不自动下载。
运行期间应保留原环境、源码和权重，不进行升级或替换。
源码身份只规范化 CRLF/LF 换行后计算哈希；模型、metadata 和预测产物仍按完整原始字节校验。

## 单次拟合、提交与恢复规则

每个 run_id 是一个完整 fit + Val/Test 预测 + 五指标 + 保存任务，不恢复算法内部步骤。

1. fit 前原子写入 `status=running` 的 canonical record，标识任务已经开始。
2. 沿用原 Train-only fit 和 Val/Test 分离评价；不使用 Test 选择模型或重新拟合全量数据。
3. 复用原模型保存函数，先保存 `.staging.pkl/.staging.json`，再 rename 到最终模型和 metadata。
   预测先保存 `.staging.npz`，再 rename；记录三份最终产物的完整 SHA-256。
4. 最后原子提交 `status=passed` 的 canonical record，包含自身内容校验和。
   产物未全部保存时不会提交成功；恢复时再次验证身份、记录和所有必要产物哈希。
5. 验证成功的 run_id 直接复用，不调用 fit，不重写其记录或模型。完全没有开始记录的任务按原顺序执行。

可捕获的异常保留 `failed`、阶段、异常类型、错误原因、已有评价和计时。
KeyboardInterrupt/SystemExit 等保留 `interrupted` 后退出；强杀留下的 `running` 视为未完成。
恢复发现任何 failed/interrupted/running/corrupt 记录时，在新拟合前报错，并写入
`reports/resume_issues.json`。损坏成功产物在报表中标为 corrupt，不再计作成功。

这些任务只有通过精确 `--retry-runs` 才会重新执行；不能授权重跑已经验证成功的任务。
重试前将原 canonical record、模型、metadata、预测和遗留 staging 文件复制到
`history/<run_id>/<attempt-id>/`，并保存人工授权记录，然后执行同一个 run_id、同一个 seed。
尝试编号递增；原失败信息和旧结果保留，统计只采用当前 canonical 记录，不增加重复数、不选择最佳结果。
原有首次运行对其他尚未开始任务的处理顺序不变；没有新增自动重试、降采样、模型替换或预算回退。

无法解析的记录、非 canonical/重复文件或不明实验身份需要人工检查；不自动删除或修补历史。
首次输出目录已有不明文件时拒绝启动。补丁前的兼容验证仍可查看、恢复模型和重建报告，
但不会被自动升级为可续跑实验。

## 写入锁与强制中断

同一实验目录的 `.benchmark.lock` 使用操作系统独占锁；Windows 为 msvcrt，其他平台为 flock。
持锁进程存活时，第二个正式 writer 无法启动，`--recover-lock` 也不能绕过。
锁记录 PID、主机和开始/关闭时间，正常退出时关闭标记，保留文件。

强杀或崩溃后操作系统释放锁，但 active 标记仍在。必须先人工确认原 PID 和相关子进程
已经结束，再显式使用 `--resume --recover-lock`；原锁记录归档到 `history/locks/`。
若有未完成 run_id，还需同时提供 `--retry-runs`。例如：

```powershell
# 仅在确认原进程及其子进程已结束后使用
.\.venv\Scripts\python.exe -m project1 current benchmark-3d --formal-config config/hpt_formal_benchmark.json --resume --recover-lock --retry-runs HPTB_S2_RO_Cpt__random_forest__seed_2026
```

锁状态无法解析时明确报错并要求人工检查，不自动删除锁文件。
不在不支持本机操作系统文件锁语义的共享文件系统上宣称同等保证。

## 结果重建和验证

`rebuild_formal_reports(output_dir)` 仍是原有报告入口。按计划校验并读取每个 run_id 的唯一
canonical record；history 不参与统计。重建原四张 CSV、accuracy aggregate、manifest、
benchmark_summary.json 和 failures.json。

只有组内全部既定 seed 成功且产物完整时才产生完整组的均值与样本标准差（ddof=1）；
少一个 seed 就保持 incomplete，不用剩余 seed 冒充完整实验。全实验 `complete=true`
要求计划中所有 run_id 均成功。中断后的表格可以重建；报告重建本身不训练或自动续跑。

验证结果和环境身份保存在 `data/output/benchmark/preflight/hpt_resume/`。
针对性测试：36 passed（其中新增续跑测试 20 项），23.36 秒。
完整回归：246 passed、25 条既有 warnings、0 skipped，47.60 秒；pip check 无依赖冲突。
覆盖正常首次提交、部分完成后中断、成功项不再 fit、三类产物和 canonical 记录损坏、
保存中断、失败人工重试、数据/split/参数/环境/清单/快照变化拒绝、重复记录拒绝、
真实子进程持锁/强杀后的人工确认，以及 ddof=1 和统计不重复。
自动化测试使用生成的小规模 fixture 和轻量模型，不依赖正式数据库或 04_TEST_DATA。
真实库检查仅执行 prepare-only，未启动正式拟合；Ma=CA/a 的有符号定义与 66 个负值保持不变。
