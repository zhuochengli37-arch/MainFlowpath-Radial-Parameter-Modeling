# PyCharm 离线导入指南

## 1. 适用场景

本文用于指导你把 `AIR2 Project1` 交付到离线 Windows 机器后，在 `PyCharm` 中完成导入、解释器配置和脚本运行。

本文重点适配两类使用方式：

1. 在 `PyCharm` 中直接点运行脚本；
2. 在 `PyCharm Terminal` 中执行 `python -m project1 ...`。

## 2. 先说最重要的一件事

**不要把当前机器的 `.venv` 直接复制给甲方。**

原因如下：

- `.venv` 和当前机器的 Python 安装路径绑定；
- `.venv` 和当前机器的系统 DLL、环境变量状态绑定；
- 跨机器复制后，很容易出现解释器路径错乱、依赖可见但运行异常的问题。

正确做法是：

1. 复制项目源码；
2. 复制 `offline_packages/`；
3. 复制 `offline_installers/python-3.10.10.exe`；
4. 在甲方机器重新执行离线部署脚本创建虚拟环境。

## 3. 需要复制到离线机的内容

请把以下内容复制到甲方离线机器：

- 整个项目目录；
- `offline_packages/`；
- `offline_installers/python-3.10.10.exe`；
- `config/`；
- `data/`；
- `docs/`；
- `workbase/`；
- `project1/`。

不建议复制的内容：

- 当前机器已有的 `.venv`
- 你的本地 IDE 配置目录
- 你的 AI 辅助目录
- 你的测试缓存和临时虚拟环境

## 4. 离线机先做什么

在甲方机器上，先完成 Python 安装，再做 `PyCharm` 导入。

### 4.1 安装 Python

双击安装：

```text
offline_installers/python-3.10.10.exe
```

建议安装到：

```text
D:\python310\
```

### 4.2 验证 Python

打开 `PowerShell`，执行：

```powershell
D:\python310\python.exe --version
```

期望输出：

```text
Python 3.10.10
```

如果安装路径不同，后续命令里的 Python 路径请替换成真实路径。

## 5. 在离线机执行项目部署

这一步建议先在 `PowerShell` 中完成，不要一上来就在 `PyCharm` 里点运行。

### 5.1 进入项目根目录

例如：

```powershell
cd D:\Myprogram\Python\AIR2\Project1
```

### 5.2 推荐部署命令

优先执行：

```powershell
powershell -ExecutionPolicy Bypass -File "workbase/tools/deployment/deploy_offline.ps1" -PythonExe "D:\python310\python.exe"
```

这会默认创建：

```text
.\.venv
```

### 5.3 如果项目根目录不适合创建 `.venv`

如果甲方机器对项目目录写权限比较严格，建议改用：

```powershell
powershell -ExecutionPolicy Bypass -File "workbase/tools/deployment/deploy_offline.ps1" -PythonExe "D:\python310\python.exe" -VenvPath "$env:TEMP\project1_offline"
```

当前项目已经支持这种方式。  
即使虚拟环境不在项目根目录，只要部署脚本执行成功，后续在 `PyCharm` 里直接运行脚本也仍然可用。

## 6. 部署完成后先检查什么

部署脚本执行完成后，建议优先确认这些提示是否出现：

- `Detected Python version: 3.10.10`
- `Python version matches offline manifest.`
- `Core dependencies imported successfully.`
- `Project modules imported successfully.`
- `Entrypoint check passed.`

如果这些都正常，说明离线部署本身已经基本成功。

## 7. 在 PyCharm 中导入项目

### 7.1 打开项目

在 `PyCharm` 中选择：

1. `Open`
2. 选择项目根目录
3. 等待项目索引完成

### 7.2 设置解释器

进入：

1. `File`
2. `Settings`
3. `Project`
4. `Python Interpreter`

然后：

1. 点击 `Add Interpreter`
2. 选择 `Existing environment`
3. 选择你刚部署好的虚拟环境解释器

解释器路径示例：

- 默认 `.venv`：`D:\Myprogram\Python\AIR2\Project1\.venv\Scripts\python.exe`
- `TEMP` 虚拟环境：`C:\Users\用户名\AppData\Local\Temp\project1_offline\Scripts\python.exe`

### 7.3 工作目录一定要对

如果你要直接运行脚本，`Working Directory` 必须设置为项目根目录。

例如：

```text
D:\Myprogram\Python\AIR2\Project1
```

这个设置非常重要，因为项目里的配置、数据、日志路径都默认相对项目根目录解析。

## 8. 你习惯直接运行脚本时，推荐怎么配

既然你平时更习惯直接运行脚本，这里给你按这个习惯来写。

### 8.1 推荐建立哪些 Run Configuration

建议你在 `PyCharm` 里分别建立这些运行配置：

- `workbase/current_scripts/predict_1d.py`
- `workbase/current_scripts/predict_3d.py`
- `workbase/current_scripts/workline_pipeline.py`
- `workbase/generic_scripts/generic_predict_1d.py`

### 8.2 每个脚本运行配置建议这样设

以 `workline_pipeline.py` 为例：

1. `Script path`：`workbase/current_scripts/workline_pipeline.py`
2. `Working directory`：项目根目录
3. `Python interpreter`：离线部署后的虚拟环境解释器
4. `Parameters`：例如 `--mode predict`

### 8.3 为什么现在可以放心直接运行脚本

当前项目已经做了两层保护：

1. 脚本内部会检查当前解释器是否为项目部署解释器；
2. 离线部署脚本会记录当前项目应该使用的解释器路径。

所以即使虚拟环境部署在 `TEMP`，脚本也能识别并使用正确解释器。

## 9. 在 PyCharm 里建议先验证什么

### 9.1 先做帮助检查

在 `PyCharm Terminal` 中执行：

```powershell
python -m project1 --help
python -m project1 current --help
python -m project1 generic --help
```

### 9.2 再做直接脚本检查

建议至少运行一条你最常用的脚本：

```powershell
python workbase/current_scripts/workline_pipeline.py --mode predict
```

如果你还会用通用流程，再检查：

```powershell
python workbase/generic_scripts/generic_predict_1d.py
```

### 9.3 通过标准

满足以下条件即可认为 `PyCharm` 配置正确：

1. 没有 `ModuleNotFoundError`；
2. 没有解释器切换错误；
3. 日志文件能正常生成；
4. 输出目录能正常写入。

## 10. 常见问题

### 10.1 为什么不能直接复制 `.venv`

因为 `.venv` 不是纯项目文件，而是带有本机路径和环境状态的执行环境。  
交付给甲方时应重新创建，而不是原样复制。

### 10.2 如果 `PyCharm` 用错了解释器怎么办

优先检查：

1. `Python Interpreter` 是否指向部署好的虚拟环境；
2. `Working Directory` 是否为项目根目录；
3. 是否重新执行过一次 `deploy_offline.ps1`。

### 10.3 如果项目根目录不方便创建 `.venv`

请直接用：

```powershell
powershell -ExecutionPolicy Bypass -File "workbase/tools/deployment/deploy_offline.ps1" -PythonExe "D:\python310\python.exe" -VenvPath "$env:TEMP\project1_offline"
```

然后在 `PyCharm` 中把解释器指向这个路径下的 `python.exe`。

### 10.4 如果运行脚本时报缺依赖

优先排查：

1. 是否真的用了部署后的虚拟环境解释器；
2. `offline_packages/` 是否完整；
3. 是否已经执行过部署脚本；
4. 是否在项目根目录运行。

## 11. 给甲方交付时的推荐状态

建议你交付给甲方的目录尽量保持“干净交付”：

- 保留源码、数据、配置、文档、离线安装包；
- 删除本地测试缓存；
- 删除本地临时虚拟环境；
- 删除个人 IDE 配置目录；
- 删除 AI 辅助目录和记忆目录；
- 不附带你当前机器的 `.venv`。

如果你这样整理后再交付，甲方收到的会更像“正式交付包”，而不是“开发中工作目录”。
