# HPT 真实数据接入与正式 benchmark：阶段 B

核查日期：2026-10-22，Asia/Shanghai。阶段 A 审查由用户确认，本阶段已授权实现。
本记录只证明工程兼容性；阶段 C、类内筛选和样本效率实验均未启动。

## 数据与冻结实例

- 正式库：`D:/Myprogram/Python/MasterThesis/DataProcessing/reference/DATABASE_HPTB`，始终只读。
- 244 个 DAT：两级 MAIN 各 61 个，INLET/OUTLET 各 61 个；共享 61 个工作状态。
- MAIN 共 19,520 条 section-aware 样本；8 个 partition、40 个 stage-section-target 任务。
- 每个任务 61 × 40 = 2440 行，所有任务共享工况划分。
- `PI_` 文件名原值进入第二输入列；HPC 的 W*COR/PHI 接口及物理定义不变。
- ξ 网格为 40 个非均匀点，范围 0.01—0.99，不增加端点或重采样。
- 源 `Ma` 规范为内部 `MA`，数值不变。第二级 RO 保留 13 个工况的 66 个负值。

数据身份、逐文件 SHA-256、61 条工况、40 个任务、split、模型参数、软件版本和协议快照：

`data/output/benchmark/config/hptb_pi_real_v1_instance.json`

可直接查看的工况清单：同目录 `hptb_pi_real_v1_instance.csv`。
ID 使用既有 `.12g` 工况标识规范；X 中的 PI 仍为完整浮点原值，不按 ID 精度舍入。

| 内容 | SHA-256 |
|---|---|
| 数据库存量（相对路径 + NUL + 文件 SHA-256 原始字节，按路径排序） | `c7d324cf4c09995397732204289285f84e5b3bee61fa9e49c6ab493dda970ebb` |
| 共享工况划分 | `663a0a13704e64256e735bbaaa00912f651f066aabe27efe87bddaf0390bf415` |
| 完整实验实例 | `b21082665190d5a14121de9d319d20092eef3b372abdeab93ac0bfc432f0a81a` |

split seed 42，沿用 `condition_train_validation_test_split()` 的 60%/20%/20% 及整数规则：
Train/Validation/Test = 37/12/12 个工况，即每任务 1480/480/480 行。
只在一个完整任务上调用既有划分函数，再按工作状态映射到其余全部任务。

## 实现位置与行为

| 文件 | 修改 |
|---|---|
| `services/meta_parser.py` | 原有第二变量解析增加 PI 前缀和来源名，保留所有旧接口 |
| `experiments/benchmark_instance.py` | 复用 loader/partition/split；冻结和核验数据、工况、任务与哈希 |
| `experiments/benchmark_runners.py` | 增加 `run_formal_benchmark`、逐模型 seed 注入；移除旧 3D/2D CV 的 GPR 800 行截断 |
| `experiments/benchmark_reporting.py` | 五指标、逐次记录导出、均值/样本标准差、报告恢复 |
| `current_scripts/benchmark_3d.py`、`project1/cli.py` | 原入口显式增加正式模式和准备/兼容检查参数，原 CV 默认路径保留 |
| `config/hpt_formal_benchmark.json` | 本次 HPT 输入、实例、输出和语义配置 |
| `data/output/benchmark/results/*.csv` | 四个原有空模板补齐实验、case、seed、Val/Test、运行状态等字段 |
| `.gitignore` | benchmark 下的模型和预测数组继续排除，轻量记录保留 |
| `tests/test_formal_benchmark.py` | PI、40 任务、split、指标、seed、泄漏防护、恢复、失败统计与 GPR 全样本测试 |
| `workbase/tools/regression/validate_hpt_stage_b.py` | 真实库与保存产物的无训练复核，可重复执行 |

未修改模型工厂、35 个模型、六类 taxonomy、依赖或隔离运行时。正式模式不调用旧 CV
或全库重拟合路径，不使用全局 max_samples 截取；所有模型只收到固定 Train 的 X/y。
Pipeline 的 scaler 和基础模型 context/search 同样只在 Train 上拟合。
本次不调超参数，Test 不参与训练、预处理或选择。Validation/Test 分别预测、分别计分。

NRMSE_range 按原协议使用各自评价集的目标范围；满足协议近常量条件时写 null 和原因，
不钳制分母。MAPE 分母仍为 max(abs(y), 1e-6)。近常量 Cpt/Ctt 及带符号 Ma 不被筛除。

随机性分类和 seeds 直接读取冻结协议；不从模型类别猜测。Pipeline 的随机参数、lazy
adapter 的 model_kwargs、KAN 和外部桥接的 random_state 分别传递；split seed 保持 42。

## GPR 授权变更

原 800 行逻辑是 runner 中 CV 训练折的计算开销限制，不是核函数或优化超参数。
阶段 B 中用户进一步明确授权：正式和旧 CV 都取消该限制。

- 删除 3D CV 的 gpr_matern / ard_gpr 完整工况采样上限。
- 删除 2D CV 的 gpr_matern 行采样上限（ARD 原本不属于旧 2D 候选集）。
- 正式模式两个 GPR 都使用 1480 行；旧模式使用其训练折的全部有效行。
- 保留工厂原核函数、优化器、优化重启次数、normalize_y 和其他算法参数。
- 全面检索未发现其他 GPR 专属样本截取。旧 CV 的所有模型共用 max_samples 配置保持原样；
  它不是额外的 GPR 限制。正式模式不走该全局采样路径。
- 不因耗时增加自动跳过、降采样或回退。

## 命令

以下均在 `01_DEV/Project1` 下执行。

只准备/核验实例，不训练：

```powershell
.\.venv\Scripts\python.exe -m project1 current benchmark-3d --formal-config config/hpt_formal_benchmark.json --prepare-only
```

显式少量模型兼容检查，要求独立输出目录（已运行的目录保留，不覆盖）：

```powershell
.\.venv\Scripts\python.exe -m project1 current benchmark-3d --formal-config config/hpt_formal_benchmark.json --models ridge_deg2 --output data/output/benchmark/preflight/hpt_stage_b/ridge_coverage
.\.venv\Scripts\python.exe -m project1 current benchmark-3d --formal-config config/hpt_formal_benchmark.json --models random_forest gpr_matern ard_gpr --cases HPTB:S2:RO:MA --output data/output/benchmark/preflight/hpt_stage_b/representative_smoke
```

直接脚本入口也支持同样参数。未传 `--formal-config` 时继续使用原 YAML CV 配置。
阶段 C 的以下命令已准备好，**本阶段没有执行，须由用户另行授权启动**：

```powershell
.\.venv\Scripts\python.exe -m project1 current benchmark-3d --formal-config config/hpt_formal_benchmark.json
```

它将使用全部 35 模型、40 任务，共 2600 次拟合：20×40 + 14×3×40 + 1×3×40。
结果写入 `data/output/benchmark/runs/hptb_pi_first_round_v1`。
PySR 和基础模型沿用当前工厂工程设置并已逐参数快照；不将其宣称为调优后的论文最优预算。
PySR 仍使用独立 Julia 环境，Windows 长路径/depot 问题仍按既有文档处理，不新增超时或回退。

## 记录、恢复与统计

每次拟合写独立原子 JSON：`records/<case>__<model>__seed_<seed>.json`。
记录工况哈希、两套 taxonomy 身份、seed、参数、Train 数、Val/Test 五指标、墙钟耗时和失败原因。
模型文件名包含 seed；模型 metadata 中的 training_sources 仅包含 Train 源文件。
预测数组单独保存 Val/Test 的 X、truth、prediction 和 condition IDs。

四类 CSV 分别记录 accuracy/runtime/metadata/summary，另有
`model_accuracy_aggregate.csv`。只在同一 case/model/evaluation_subset 内对既定 repeats
计算均值及 ddof=1 样本标准差；单次确定性运行的样本标准差留空。
重复缺失或失败时保留原始记录，聚合标为 incomplete，不拿成功子集冒充完整重复结果。
不进行加权总分、跨尺度原始 RMSE 平均、最佳 seed 选择或代表模型筛选。

`rebuild_formal_reports(output_dir)` 可从 records 重建表、summary 和 manifest，不训练、不自动续跑。
`models/models_manifest.json` 保留现有模型身份结构，并增加 model_seed；恢复某个 repeat 时应使用
记录中的精确 model_path，不依赖默认自动选模来选择 seed。

耗时是主进程 fit/predict 调用的墙钟时间，外部桥接包含其进程开销；官方内部时间如有则保留在
symbolic/foundation 元数据中。公共 runner 未测峰值内存，memory 明确为空并记录原因，不伪造数值。

## 验证证据与边界

- 修改前完整回归：210 passed，25 warnings。
- 最终完整回归：224 passed，25 warnings，26.96 秒，无跳过；pip check 无依赖冲突。
- 新增 14 项测试覆盖：PI/旧流量、40 任务共享 split、五指标、全部随机模型 seed 字段、
  Train-only scaler/source、模型恢复与表重建、数据改变拒绝、失败记录，以及 GPR 超 800 行不截取。
- 真实库 Ridge 覆盖 40/40；Random Forest 三 seed + 两个 GPR 在 S2:RO:MA 共 5/5。
- 本次真实数据兼容测试共 45 次拟合、90 组 Val/Test 评价，不是正式第一轮。
- 两个真实 GPR 拟合对象的 X_train_ 均为 1480 行；实测 fit 约 23.6 秒、55.6 秒。
  这些是兼容性测量，部分检查并行进行，不用于论文计算成本排名。
- 保存模型重载预测逐元素一致；报告从逐次记录重建后 CSV 字节一致；数据哈希未变。
- 复核命令：`.\.venv\Scripts\python.exe workbase/tools/regression/validate_hpt_stage_b.py`。
- 汇总证据：`data/output/benchmark/preflight/hpt_stage_b/stage_b_acceptance.json`。

Ma 数据定义已由用户于 2026-10-22 明确确认：Ma = CA/a，为绝对坐标系下的有符号轴向马赫数；
CA 为绝对轴向速度分量，a 为当地声速。第二级 RO 的 66 个负值属于允许数值。
源 Ma → canonical MA 原样传递，不取绝对值、不截断、不筛除、不重新计算。
唯一数据来源为正式径向分布数据库，不再追溯上游代码或查找第三章正文；定义核查已关闭。
补充数据说明见 `data/output/benchmark/preflight/hpt_stage_c/ma_definition_audit.json`。

未重新执行真实 HPT 上的 PySR、TabPFN、TabICLv2 训练；既有桥接与离线测试保持通过。
全部 2600 次拟合的总耗时、全模型真实精度仍未知。
代表选择保持 pending；样本效率的整数计数、子集和取整规则均未实例化。
