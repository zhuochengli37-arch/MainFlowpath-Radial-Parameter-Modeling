# AIR2 Project1

AIR2 Project1 是一个用于气动性能预测的机器学习框架。

当前项目以 `workbase/` 为核心，支持两类工作流：

- `workbase/current_scripts`：领域专用的 1D / 2D / 3D 流程（含 workline 一键串联）
- `workbase/generic_scripts`：通用 1D / 2D / 3D 表格流程

## 主要目录结构

```text
Project1/
|- config/
|- data/
|  |- current/
|  |- generic/
|  |- input/    (停用，仅保留说明)
|  `- output/   (停用，仅保留说明)
|- docs/
|- tests/
`- workbase/
   |- common/
   |- current_scripts/
   |- generic_scripts/
   `- src/
```

## 数据案例映射

- `DATACASE1`: 当前项目 3D 数据格式，同时也是 2D + 工作线预测输入格式
- `sample6`: 当前项目 1D 数据格式，也可作为通用 1D 示例
- `DATACASE3`: 通用 2D 表格格式
- `DATACASE4`: 通用 3D 表格格式

研究所正式数据与Legacy验证数据的边界见 `docs/研究所数据Schema说明.md`。当前正式
单截面模型只加载 `station=MAIN`；INLET/OUTLET可识别但不训练。四截面可由独立
Adapter识别并拆分为canonical数据，但仍不会进入现有1D/2D/3D训练或预测入口。

训练数据通常位于 `data/current/3d/<dataset>/train/`。  
预测数据通常位于 `data/current/3d/<dataset>/predict/`。

如果你不喜欢命令行，也可以直接在 IDE 中运行：

- `workbase/current_scripts/*.py`
- `workbase/generic_scripts/*.py`

但数据和输出目录仍然应统一使用 `data/current` 与 `data/generic`。

## 领域专用工作流

这些脚本使用 `config/benchmark_config.yaml`：

- `python workbase/current_scripts/benchmark_1d.py`
- `python workbase/current_scripts/predict_1d.py`
- `python workbase/current_scripts/benchmark_2d.py`
- `python workbase/current_scripts/workline_pipeline.py`
- `python workbase/current_scripts/benchmark_3d.py`
- `python workbase/current_scripts/predict_3d.py`

对应的统一入口是：

- `python -m project1 current benchmark-1d`
- `python -m project1 current predict-1d`
- `python -m project1 current benchmark-2d`
- `python -m project1 current predict-2d`
- `python -m project1 current benchmark-3d`
- `python -m project1 current predict-3d`

## 通用工作流

这些脚本使用 `config/generic_config.yaml`：

- `python workbase/generic_scripts/generic_benchmark_1d.py`
- `python workbase/generic_scripts/generic_predict_1d.py`
- `python workbase/generic_scripts/generic_benchmark_2d.py`
- `python workbase/generic_scripts/generic_predict_2d.py`
- `python workbase/generic_scripts/generic_benchmark_3d.py`
- `python workbase/generic_scripts/generic_predict_3d.py`

对应的统一入口是：

- `python -m project1 generic benchmark-1d`
- `python -m project1 generic predict-1d`
- `python -m project1 generic benchmark-2d`
- `python -m project1 generic predict-2d`
- `python -m project1 generic benchmark-3d`
- `python -m project1 generic predict-3d`

## 回归测试

- `python workbase/tools/regression/run_regression_suite.py`
- `python workbase/tools/regression/run_regression_suite.py --group current`
- `python workbase/tools/regression/run_regression_suite.py --group generic`

## 通用表格规则

- 输入文件必须包含表头
- 前 `N` 列视为输入列
- 剩余列视为输出列
- 支持 `.txt`、`.csv`、`.dat`
- 同一目录下允许不同文件暴露不同输出列，但输入列必须一致

## DATACASE 说明

### `DATACASE1`

`DATACASE1` 是结构化气动案例。训练数据包含完整的 `(rpm, wcor, xi, outputs)` 信息。预测数据按目录结构组织，部分文件可能只包含单列 `xi`；这类只含 `xi` 的文件也是合法预测输入。

### `sample6`

`sample6` 是当前项目统一后的 1D 示例数据集，也可直接用于通用 1D 工作流。典型文件示例如下：

```text
xi psi tsi
0.0 1.23 0.91
0.5 1.28 0.95
1.0 1.31 0.99
```

### `DATACASE3`

`DATACASE3` 是通用 2D 表格格式，通常为两列输入加一列或多列输出。

### `DATACASE4`

`DATACASE4` 是通用 3D 表格格式，通常为三列输入加一列或多列输出。

## 自动化测试

在项目根目录运行：

```bash
python -m pytest tests -v
```

当前测试集已经与 `DATACASE1`、`sample6`、`DATACASE3`、`DATACASE4` 的实际格式对齐。

## 离线部署

当前仓库已经提供完整的离线部署流程：

- 联网环境准备离线包：`powershell -ExecutionPolicy Bypass -File "workbase/tools/deployment/prepare_offline_packages.ps1"`
- 离线环境安装依赖：`powershell -ExecutionPolicy Bypass -File "workbase/tools/deployment/deploy_offline.ps1"`

离线机器建议使用项目自带的 `.venv` 作为解释器。  
在 PyCharm 中请将项目解释器设置为：

- `.\.venv\Scripts\python.exe`

如果项目根目录不适合创建 `.venv`，也可以把虚拟环境部署到其他路径，例如：

- `powershell -ExecutionPolicy Bypass -File "workbase/tools/deployment/deploy_offline.ps1" -PythonExe "D:\python310\python.exe" -VenvPath "$env:TEMP\project1_offline"`

部署脚本会自动记录当前项目应使用的虚拟环境解释器，因此后续直接运行 `workbase/current_scripts/*.py` 与 `workbase/generic_scripts/*.py` 仍然可用。

离线依赖包建议至少包含：

- `numpy`
- `scipy`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `PyYAML`
- `tqdm`

## 文档

- `docs/文档索引.md`
- `docs/基准配置指南.md`
- `docs/迁移指南.md`
- `docs/PyCharm离线导入指南.md`
- `docs/离线部署自检清单.md`
- `docs/Python310离线部署指南.md`
- `docs/已训练模型预测指南.md`
- `docs/方法二与方法三预测指南.md`
- `docs/模型版本管理指南.md`
- `docs/模型说明.md`
- `docs/公共数据集预测报告模板.md`
