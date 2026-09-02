# 研究所数据 Schema 说明

## 当前阶段边界

当前DEV仅适配研究所正式单截面数据库。四截面数据库只用于确认未来接口边界，
不进入1D、2D或3D训练与预测。

## Schema分类

### Legacy Validation Schema

- 标识：`legacy_validation`
- 典型表头：`xi psi tsi mai`，也兼容历史代理模型已经支持的表格。
- 用途：保留旧验证数据、旧测试和旧模型流程兼容性。
- 说明：旧 `psi/tsi/mai` 不再代表研究所正式数据定义。

### Institute Single-Section Schema

- 标识：`institute_single_section`
- 正式表头：`xi Cpt Ctt Cps Cts MA`
- 工况参数：从目录 `CNC_*` 与文件名 `WnCOR_*` 解析。
- 当前训练范围：仅 `station=MAIN`。
- `DATABASE_*_INLET` 和 `DATABASE_*_OUTLET` 可以识别、检查，但不进入当前模型训练。
- 数据未提供权威RI/RO/SI/SO含义，因此内部 `section` 保持未指定。

### Institute Four-Section Schema

- 标识：`institute_four_section`
- 特征：表头字段带 `_RI`、`_RO`、`_SI`、`_SO` 后缀。
- 当前行为：能够识别Schema，但加载器明确不将其交给当前训练或预测流程。
- 后续实现不得假设压气机和涡轮四截面的输出字段、物理定义或单位相同。

## 统一内部metadata

领域样本至少暴露：

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

`speed_parameter` 与 `flow_parameter` 是不带单位假设的内部名称。现有模型代码中的
`rpm`、`wcor` 和 `source_path` 继续作为兼容别名，避免破坏Legacy模型与测试。

## 当前模型分区

1D `speed_parameter -> flow_parameter` 模型仍只按：

```text
component + stage
```

分区。`station` 负责训练准入，`section` 本阶段不参与1D分区。2D和3D模型仍按
现有单截面逻辑运行。
