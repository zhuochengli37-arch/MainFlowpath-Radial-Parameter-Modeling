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
- 当前 1D/2D/3D 训练和预测入口仍不准入该 Schema。
- 完整规则和正式样例核验见 `docs/研究所四截面数据Adapter说明.md`。

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
```

Four-Section Adapter 另提供当前 section 实际存在的 `outputs` 和原字段追踪
`source_fields`。`rpm`、`wcor` 和 `source_path` 继续作为兼容别名。

## station 与 section

- `station`：`MAIN`、`INLET`、`OUTLET`。
- `section`：MAIN 级内的 `RI`、`RO`、`SI`、`SO`。
- Single-Section 和 INLET/OUTLET 的 `section=None`。
- 准入判断使用 station，不以 stage=0/999 代替 station 语义。

## 当前模型边界

1D `speed_parameter -> flow_parameter` 仍只按 `component + stage` 分区，不加入
section。2D/3D 仍按单截面逻辑运行。四截面当前只到
`raw file -> schema -> adapter -> canonical data` 为止。

`dev-v1.1-single-section` 的正式单截面功能和 Legacy Validation Schema 兼容能力继续
保留。本阶段不包含四截面训练、预测或精度验证。
