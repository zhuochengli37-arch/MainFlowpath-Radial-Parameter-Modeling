# Python 3.10 离线部署指南

## 0. 本文适用场景

本文用于指导 Windows 离线机器部署 `AIR2 Project1`，目标是让离线机同时具备以下能力：

1. 能安装匹配版本的 Python 3.10；
2. 能在离线环境创建项目虚拟环境并安装依赖；
3. 能直接运行项目脚本，例如：
   - `workbase/current_scripts/workline_pipeline.py`
   - `workbase/current_scripts/predict_3d.py`
   - `workbase/generic_scripts/generic_predict_1d.py`
4. 能使用统一入口：
   - `python -m project1 ...`

本文按“照着执行”的方式编写，建议严格按顺序操作。

## 1. 先记住两个原则

### 1.1 不要直接复制联网机器上的 `.venv`

不建议把联网机器上的 `.venv` 直接拷贝到离线机。

原因如下：

- `.venv` 与本机 Python 安装路径绑定；
- `.venv` 与本机 DLL、环境变量和系统状态绑定；
- 跨机器复制后，容易出现能打开但运行异常的隐蔽问题。

正确做法是：

1. 复制源码；
2. 复制离线依赖包；
3. 在离线机重新创建虚拟环境。

### 1.2 你的使用习惯如果是“直接运行脚本”，更要把解释器配对好

你如果习惯直接运行：

- `workbase/current_scripts/*.py`
- `workbase/generic_scripts/*.py`

那么离线部署完成后，最重要的是：

1. 使用部署脚本创建好的虚拟环境解释器；
2. 在 IDE 中把解释器指向这个虚拟环境；
3. 运行脚本时，工作目录保持为项目根目录。

当前部署脚本已经会自动记录本项目对应的虚拟环境路径，即使虚拟环境不放在项目根目录下，也能支持脚本直接运行。

## 2. 联网机器需要准备什么

在联网机器上，项目根目录至少应具备以下内容：

- 完整项目源码；
- `offline_packages/` 离线依赖包目录；
- `offline_installers/python-3.10.10.exe` 或其他与你离线包匹配的 Python 3.10 安装程序。

### 2.1 重新生成离线依赖包的方法

如果你想重新打包依赖，请在联网机器项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File "workbase/tools/deployment/prepare_offline_packages.ps1"
```

执行完成后，确认以下文件存在：

- `offline_packages/`
- `offline_packages/offline_manifest.json`
- `offline_packages/` 中的 `pip`、`setuptools`、`wheel`
- `offline_packages/` 中的核心依赖包

### 2.2 当前离线部署最关键的版本要求

离线机 Python 版本必须与 `offline_packages/offline_manifest.json` 中记录的版本一致。

当前项目实际检查结果对应版本为：

```text
Python 3.10.10
```

所以离线机建议安装：

```text
Python 3.10.10
```

## 3. 从联网机拷贝到离线机的内容

请把以下内容完整复制到离线机：

1. 整个项目目录；
2. `offline_packages/`；
3. `offline_installers/python-3.10.10.exe`；
4. 如果你有自己修改过的配置，也要确保 `config/` 已完整复制。

建议重点核对这些目录不要漏：

- `config/`
- `data/`
- `docs/`
- `workbase/`
- `offline_packages/`
- `project1/`

不建议复制的内容：

- 联网机器现成的 `.venv`

## 4. 离线机安装 Python 的详细步骤

### 4.1 安装 Python

在离线机上找到安装包，例如：

```text
offline_installers/python-3.10.10.exe
```

双击安装。

建议安装到你容易识别的位置，例如：

```text
D:\python310\
```

安装完成后，先不要急着运行项目，先验证 Python 是否真的安装好了。

### 4.2 验证 Python

打开 PowerShell，执行：

```powershell
D:\python310\python.exe --version
```

期望看到：

```text
Python 3.10.10
```

如果你的 Python 不在这个路径，请把后续文档里的：

```text
D:\python310\python.exe
```

替换成你的真实路径。

## 5. 离线部署依赖的详细步骤

以下步骤都在项目根目录执行。

### 5.1 先进入项目根目录

示例：

```powershell
cd D:\Myprogram\Python\AIR2\Project1
```

进入后，你应该能看到：

- `workbase/`
- `config/`
- `docs/`
- `offline_packages/`

### 5.2 推荐部署命令

如果项目目录本身允许创建虚拟环境，先执行：

```powershell
powershell -ExecutionPolicy Bypass -File "workbase/tools/deployment/deploy_offline.ps1" -PythonExe "D:\python310\python.exe"
```

这会默认把虚拟环境创建到：

```text
.\.venv
```

### 5.3 如果项目目录写权限不稳定，用这一条

如果你担心项目根目录写权限、杀毒软件占用、或系统策略限制，建议直接改用：

```powershell
powershell -ExecutionPolicy Bypass -File "workbase/tools/deployment/deploy_offline.ps1" -PythonExe "D:\python310\python.exe" -VenvPath "$env:TEMP\project1_offline"
```

这一条同样支持你直接运行脚本。

原因是部署脚本现在会自动在项目根目录写入一个解释器提示文件，脚本运行时会自动找到正确的虚拟环境。

### 5.4 部署过程中你会看到什么

正常情况下，部署脚本会依次完成：

1. 检查 `offline_packages/` 和 `offline_manifest.json`；
2. 检查 Python 版本是否匹配；
3. 创建虚拟环境；
4. 安装 `pip`；
5. 安装依赖；
6. 验证核心依赖是否可导入；
7. 验证统一入口和脚本相关模块是否可导入。

### 5.5 部署成功时重点看这几类提示

部署输出里建议重点关注这些信息：

- `Detected Python version: 3.10.10`
- `Python version matches offline manifest.`
- `Core dependencies imported successfully.`
- `Project modules imported successfully.`
- `Entrypoint check passed.`

只要这些都正常，通常就说明离线环境已经可以用了。

## 6. 部署后必须做的检查

下面这几步建议一条一条执行，不要省。

### 6.1 检查虚拟环境解释器

如果你部署到了默认位置：

```powershell
.\.venv\Scripts\python.exe --version
```

如果你部署到了临时目录：

```powershell
$env:TEMP\project1_offline\Scripts\python.exe --version
```

### 6.2 检查核心依赖是否能导入

默认 `.venv`：

```powershell
.\.venv\Scripts\python.exe -c "import numpy, scipy, sklearn, yaml, tqdm, xgboost, lightgbm; print('deps-ok')"
```

`TEMP` 虚拟环境：

```powershell
$env:TEMP\project1_offline\Scripts\python.exe -c "import numpy, scipy, sklearn, yaml, tqdm, xgboost, lightgbm; print('deps-ok')"
```

期望输出：

```text
deps-ok
```

### 6.3 检查统一入口是否可用

默认 `.venv`：

```powershell
.\.venv\Scripts\python.exe -m project1 --help
```

`TEMP` 虚拟环境：

```powershell
$env:TEMP\project1_offline\Scripts\python.exe -m project1 --help
```

只要能正常显示帮助信息，就说明统一入口已经可以用。

## 7. 你习惯直接运行脚本时，应该怎么做

这是本文最重要的一部分。

### 7.1 如果你在终端中直接运行脚本

假设你部署的是默认 `.venv`，可以这样运行：

```powershell
.\.venv\Scripts\python.exe workbase/current_scripts/workline_pipeline.py --mode predict
```

或者：

```powershell
.\.venv\Scripts\python.exe workbase/current_scripts/predict_3d.py
```

通用脚本示例：

```powershell
.\.venv\Scripts\python.exe workbase/generic_scripts/generic_predict_1d.py
```

如果你部署到 `TEMP` 路径，则把解释器前缀换成：

```powershell
$env:TEMP\project1_offline\Scripts\python.exe
```

例如：

```powershell
$env:TEMP\project1_offline\Scripts\python.exe workbase/current_scripts/workline_pipeline.py --mode predict
```

### 7.2 如果你在 PyCharm 中直接点运行

请按下面顺序设置：

1. 打开项目；
2. 进入 `Settings`；
3. 打开 `Project: ... -> Python Interpreter`；
4. 选择部署好的虚拟环境解释器；
5. 确认 `Working Directory` 是项目根目录。

解释器示例：

- 默认 `.venv`：`D:\Myprogram\Python\AIR2\Project1\.venv\Scripts\python.exe`
- `TEMP` 环境：`C:\Users\你的用户名\AppData\Local\Temp\project1_offline\Scripts\python.exe`

### 7.3 推荐先跑哪几个脚本

如果你最常用的是当前项目领域脚本，建议先检查：

```powershell
.\.venv\Scripts\python.exe workbase/current_scripts/predict_1d.py
.\.venv\Scripts\python.exe workbase/current_scripts/predict_3d.py
.\.venv\Scripts\python.exe workbase/current_scripts/workline_pipeline.py --mode predict
```

如果你还会用通用脚本，再补充检查：

```powershell
.\.venv\Scripts\python.exe workbase/generic_scripts/generic_predict_1d.py
.\.venv\Scripts\python.exe workbase/generic_scripts/generic_predict_2d.py
```

## 8. 关于 `-InstallEditable` 要不要用

大多数离线使用场景下，不必额外加：

```powershell
-InstallEditable
```

原因是：

- 你从项目根目录运行 `python -m project1 ...` 已经可用；
- 你直接运行 `workbase/current_scripts/*.py`、`workbase/generic_scripts/*.py` 也已经可用；
- 离线部署的核心目标是“能运行”，不是必须做开发态可编辑安装。

如果你确实传了 `-InstallEditable`，出现 warning，也不一定代表部署失败。  
只要依赖安装成功、统一入口可用、脚本能运行，通常就可以继续使用。

## 9. 常见问题与处理方法

### 9.1 出现 `Python version mismatch`

说明离线机 Python 版本与离线依赖包记录版本不一致。

处理方法：

1. 改用匹配版本的 Python；
2. 或在联网环境重新生成离线包。

### 9.2 出现 `No module named ...`

通常说明依赖没有完整安装。

处理方法：

1. 重新执行部署脚本；
2. 检查 `offline_packages/` 是否完整；
3. 再执行依赖导入检查命令。

### 9.3 项目根目录下创建 `.venv` 失败

如果项目根目录写入受限，或系统环境对该目录管控较严，请直接改用：

```powershell
powershell -ExecutionPolicy Bypass -File "workbase/tools/deployment/deploy_offline.ps1" -PythonExe "D:\python310\python.exe" -VenvPath "$env:TEMP\project1_offline"
```

### 9.4 脚本启动后又切换错了解释器

当前部署脚本会自动为项目写入解释器提示文件。  
如果你重新部署过虚拟环境，请重新执行一次 `deploy_offline.ps1`，让提示文件同步到最新解释器路径。

## 10. 建议你实际照着执行的顺序

如果你想要一套最稳妥的实际操作顺序，建议如下。

### 10.1 安装 Python

```powershell
D:\python310\python.exe --version
```

### 10.2 进入项目根目录

```powershell
cd D:\Myprogram\Python\AIR2\Project1
```

### 10.3 执行离线部署

优先推荐这一条：

```powershell
powershell -ExecutionPolicy Bypass -File "workbase/tools/deployment/deploy_offline.ps1" -PythonExe "D:\python310\python.exe"
```

如果根目录写权限不稳定，改用这一条：

```powershell
powershell -ExecutionPolicy Bypass -File "workbase/tools/deployment/deploy_offline.ps1" -PythonExe "D:\python310\python.exe" -VenvPath "$env:TEMP\project1_offline"
```

### 10.4 做基础检查

```powershell
.\.venv\Scripts\python.exe -m project1 --help
.\.venv\Scripts\python.exe -c "import numpy, scipy, sklearn, yaml, tqdm, xgboost, lightgbm; print('deps-ok')"
```

### 10.5 做脚本检查

```powershell
.\.venv\Scripts\python.exe workbase/current_scripts/workline_pipeline.py --mode predict
```

如果你使用的是 `TEMP` 虚拟环境，就把上面的解释器替换成：

```powershell
$env:TEMP\project1_offline\Scripts\python.exe
```

## 11. 最终结论标准

满足以下条件后，可认为离线部署已经基本完成：

1. Python 版本与离线清单一致；
2. 部署脚本执行完成；
3. 核心依赖可以导入；
4. `python -m project1 --help` 可以正常显示；
5. 你常用的脚本可以直接运行；
6. 日志和输出目录能够正常写入。
