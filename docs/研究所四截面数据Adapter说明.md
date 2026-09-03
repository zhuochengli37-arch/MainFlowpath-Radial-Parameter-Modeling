# 研究所四截面数据 Adapter 说明

## 阶段边界

Institute Four-Section DATABASE 通过通用 Multi-Section 核心完成识别、解析、拆分和
canonical内部表示。Four-Section模块只保留当前正式库兼容入口与展示顺序。Adapter的
输出不会自动进入现有1D/2D/3D训练或预测，本阶段也不包含代理模型或AI模型。

## 权威样例核验

实现依据 `04_TEST_DATA/DATABASE-四个截面` 中 400 个正式 `.dat` 文件的实际目录、
文件名和 header。Schema 不继承顶层目录名称，而由每个文件的实际 header 判定，
`station` 则由 `DATABASE_*`、`DATABASE_*_INLET`、`DATABASE_*_OUTLET` 路径语义判定。

| 部件 | MAIN原始section顺序 | 每个section的实际字段 |
|---|---|---|
| FAN | RI, RO, SI, SO | xi, Cpt, Cps, Ctt, Cts, Vz, Rho |
| CMP | RI, RO, SI, SO | xi, Cpt, Cps, Ctt, Cts, Vz, Rho |
| HPTB | SI, SO, RI, RO | xi, Cpt, Cps, Ctt, Cts, Ma |
| LPTB | SI, SO, RI, RO | xi, Cpt, Cps, Ctt, Cts, Ma |

FAN/CMP 的 INLET 和 OUTLET 实际 header 是：

```text
xi Cpt Cps Ctt Cts Vz Rho
```

HPTB/LPTB 的 INLET 和 OUTLET 实际 header 是：

```text
xi Cpt Cps Ctt Cts Ma
```

这些边界文件没有 `_RI/_RO/_SI/_SO` 后缀，因此其实际 Schema 是
`institute_single_section`，`section=None`。它们可识别，但因 `station` 不是 MAIN，
不参与当前代理模型。

## Schema 定义

### Legacy Validation Schema

- 标识：`legacy_validation`
- 典型字段：`xi psi tsi mai`
- 仅保留旧数据、旧模型和回归测试兼容性，不代表研究所正式字段定义。

### Institute Single-Section Schema

- 标识：`institute_single_section`
- 正式单截面 MAIN 典型字段：`xi Cpt Ctt Cps Cts MA`
- 四截面数据库中的 INLET/OUTLET 也属于无后缀单截面表，其字段以实际 header 为准。
- Single-Section 的 `section=None`。

### Institute Four-Section Schema

- 标识：`institute_four_section`
- MAIN 宽表中的字段由 `_RI`、`_RO`、`_SI`、`_SO` 后缀明确归属。
- 当前 Four-Section兼容入口验证RI/RO/SI/SO四组完整性；通用核心不限制数量和名称。
- 每组必须有自己的 `xi` 和至少一个实际输出字段。
- 分组不依赖固定列号，因此压缩部件和涡轮源文件的列组顺序可以不同。
- 通用设计详见 `docs/研究所多截面数据框架说明.md`。

## station 与 section

`station` 表示数据库文件在部件边界上的语义，只允许 `MAIN`、`INLET`、`OUTLET`。
`section` 表示MAIN级内气动截面。当前正式库使用RI/RO/SI/SO，通用核心允许其他合法名称。

```text
component=CMP, stage=1, station=MAIN, section=RO
```

INLET/OUTLET 不会根据 stage=0 或 stage=999 被伪造成任何 section；stage 只作为原始
metadata 保留，准入判断使用 station。

## 宽表到 canonical 长格式

Adapter 按 header 后缀建立字段映射。例如：

```text
xi_RO, Cpt_RO, ..., Vz_RO
    -> component=CMP, stage=1, station=MAIN, section=RO
       speed_parameter, flow_parameter, xi, outputs, schema, source_file
```

每一行的四组数据会成为四条独立逻辑记录。每条记录至少包含：

```text
component
stage
station
section
speed_parameter
flow_parameter
xi
schema
source_file
outputs
source_fields
schema_name
schema_version
speed_parameter_name
flow_parameter_name
units
```

`source_fields` 保存 canonical 字段到原字段名的映射。例如涡轮 `Ma_SI` 明确映射为
canonical `MA`，同时保留 `source_fields["MA"]="Ma_SI"`。这里只统一大小写命名，
不修改数值、物理定义或单位。

压缩部件保留 `Vz` 和 `Rho`，涡轮保留 `MA`；Adapter 不进行 Vz/Ma 换算、不补造
Rho，也不强行统一两类部件的输出目标。

## 当前模型准入

| 数据 | 可识别/读取 | 独立Adapter拆分 | 当前模型训练/预测 |
|---|---:|---:|---:|
| Legacy Validation | 是 | 不适用 | 是（兼容路径） |
| Institute Single MAIN | 是 | 不适用 | 是 |
| Institute INLET/OUTLET | 是 | 不适用 | 否 |
| Institute Four MAIN | 是 | 是 | 否 |

1D 工作线仍只按 `component + stage` 分区，section 不参与 1D partition。现有 2D/3D
训练和预测加载器仍拒绝 `institute_four_section`；后续接入四截面模型必须另行设计和
验证。

## 完整正式库 Adapter 审计

2026-09-03 对权威样例执行了逐文件扫描：共 400 个文件；FAN 100、CMP 160、HPTB 60、
LPTB 80；MAIN 240、INLET 80、OUTLET 80。240 个 MAIN 均识别为 Four-Section 并成功
拆分，160 个边界文件均识别为 Single-Section。全部 MAIN 均得到 RI/RO/SI/SO，未发现
拒绝文件、缺失或重复 section、header 异常、同行字段数量异常或 section 字段行数不一致。
