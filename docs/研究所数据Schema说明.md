# 研究所数据 Schema 说明

## Schema 分类

### Legacy Validation Schema

- 标识：`legacy_validation`
- 典型表头：`xi psi tsi mai`
- 用途：旧验证数据、旧测试和旧模型流程兼容。
- 旧 `psi/tsi/mai` 不再代表研究所正式数据定义。

### Institute Single-Section Schema

- 标识：`institute_single_section`
- 正式单截面 MAIN 表头：`xi Cpt Ctt Cps Cts MA`
- 四截面 DATABASE 中无 section 后缀的 INLET/OUTLET 表也保持单截面 Schema。
- 工况参数从 `CNC_*` 和 `WnCOR_*` 路径解析，不改变原物理定义或单位。
- 当前模型只准入 `station=MAIN`；Single-Section 的 `section=None`。

### Institute Four-Section Schema

- 标识：`institute_four_section`
- MAIN 宽表字段带 `_RI`、`_RO`、`_SI`、`_SO` 后缀。
- 独立 Adapter 可按字段名拆分为四组 canonical 数据。
- 当前 1D/2D/3D 训练入口准入该 Schema；Multi-Section预测入口尚未准入。
- 完整规则和正式样例核验见 `docs/研究所四截面数据Adapter说明.md`。

### Institute Multi-Section Schema

- 标识：`institute_multi_section`
- 版本：`v1`
- 从 `xi_<section>` 动态发现一个或多个section，不限定section数量或名称。
- `institute_four_section` 继续作为当前研究所RI/RO/SI/SO数据的兼容标识，其canonical
  Schema身份是 `institute_multi_section / v1`。
- 每个section独立保留自己的xi、动态outputs和原字段映射。

## 统一内部 metadata

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
schema_name
schema_version
speed_parameter_name
flow_parameter_name
outputs
source_fields
units
```

`rpm`、`wcor` 和 `source_path` 继续作为兼容别名。单位只有在数据源或调用方明确提供时
才记录，不根据变量名称猜测。

## station 与 section

- `station`：`MAIN`、`INLET`、`OUTLET`。
- `section`：MAIN级内由header发现的截面名称；RI/RO/SI/SO是当前正式库实例，不是
  通用核心允许的固定全集。
- Single-Section 和 INLET/OUTLET 的 `section=None`。
- 准入判断使用 station，不以 stage=0/999 代替 station 语义。

## 当前模型边界

1D `speed_parameter -> flow_parameter` 仍只按 `component + stage` 分区，不加入
section。2D/3D训练对Single-Section保持 `component + stage`，对Multi-Section使用
`component + stage + section`。Multi-Section预测和结果输出尚未接入。

`dev-v1.1-single-section` 的正式单截面功能和 Legacy Validation Schema 兼容能力继续
保留。当前训练烟测只验证数据加载、交叉验证、拟合、保存及元数据链路可运行，不作为
代理模型精度验证；本阶段不包含Multi-Section预测。
