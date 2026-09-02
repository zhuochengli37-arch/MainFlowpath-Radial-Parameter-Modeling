# 单元测试

本目录包含 AIR2 Project1 的单元测试。

## 测试覆盖范围

### 核心模块测试

| 测试文件 | 测试模块 | 覆盖内容 |
|---------|---------|---------|
| `test_data_reader.py` | `project1.services.data_reader` | 数据文件读取（.txt/.dat格式）|
| `test_meta_parser.py` | `project1.services.meta_parser` | 从路径解析元数据（rpm, wcor, 部件）|
| `test_validators.py` | `scripts.validators` | 数据验证（文件存在性、数组形状、数值有效性）|
| `test_model_versioning.py` | `scripts.model_versioning` | 模型版本管理（元数据创建、保存、加载、兼容性检查）|
| `test_model_factory.py` | `scripts.model_factory` | 模型工厂（1D/3D模型构建）|

## 运行测试

### 安装测试依赖

```bash
pip install pytest pytest-cov
```

### 运行所有测试

```bash
# 在项目根目录下运行
cd d:\Myprogram\Python\AIR2\Project1
python -m pytest tests/ -v
```

### 运行特定测试文件

```bash
python -m pytest tests/test_data_reader.py -v
```

### 运行特定测试类

```bash
python -m pytest tests/test_validators.py::TestValidateFileExists -v
```

### 运行特定测试方法

```bash
python -m pytest tests/test_validators.py::TestValidateFileExists::test_existing_file -v
```

### 生成测试覆盖率报告

```bash
# 生成覆盖率报告
python -m pytest tests/ --cov=src --cov=scripts --cov-report=html

# 查看报告
# 打开 htmlcov/index.html
```

### 生成简洁的覆盖率报告

```bash
python -m pytest tests/ --cov=src --cov=scripts --cov-report=term-missing
```

## 测试结构

每个测试文件包含多个测试类，每个测试类对应一个功能模块：

```python
class TestFunctionName:
    """测试特定函数"""

    def test_normal_case(self):
        """测试正常情况"""
        # 测试代码

    def test_edge_case(self):
        """测试边界情况"""
        # 测试代码

    def test_error_case(self):
        """测试错误情况"""
        # 测试代码
```

## 测试最佳实践

### 1. 测试命名

- 测试文件：`test_<module_name>.py`
- 测试类：`Test<FunctionName>` 或 `Test<ClassName>`
- 测试方法：`test_<what_is_being_tested>`

### 2. 测试内容

每个测试应该验证：
- **正常情况**：函数在正常输入下的行为
- **边界情况**：空输入、极大/极小值、特殊值
- **错误情况**：无效输入、异常处理

### 3. 使用临时文件

测试涉及文件操作时，使用 `tempfile` 模块：

```python
import tempfile
from pathlib import Path

def test_file_operation():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_path = Path(f.name)
    
    try:
        # 测试代码
        pass
    finally:
        temp_path.unlink()  # 清理临时文件
```

### 4. 使用 pytest fixtures

对于需要重复使用的测试数据或设置：

```python
import pytest

@pytest.fixture
def sample_data():
    return {"xi": [0.1, 0.2, 0.3], "psi": [2.5, 2.6, 2.7]}

def test_with_fixture(sample_data):
    assert len(sample_data["xi"]) == 3
```

### 5. 参数化测试

对于需要测试多组输入的情况：

```python
import pytest

@pytest.mark.parametrize("input,expected", [
    (0.1, 2.5),
    (0.2, 2.6),
    (0.3, 2.7),
])
def test_multiple_inputs(input, expected):
    result = some_function(input)
    assert result == pytest.approx(expected)
```

## 持续集成

测试可以集成到 CI/CD 流程中：

```yaml
# .github/workflows/test.yml 示例
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -r docs/requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest tests/ --cov=src --cov=scripts
```

## 添加新测试

添加新测试时：

1. 在 `tests/` 目录下创建 `test_<module_name>.py`
2. 导入要测试的模块
3. 创建测试类和测试方法
4. 运行测试验证

示例：

```python
"""
新模块单元测试
"""

import pytest
from pathlib import Path
import sys

# 添加项目路径
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from your_module import your_function


class TestYourFunction:
    """测试 your_function"""

    def test_basic_case(self):
        """测试基本情况"""
        result = your_function(input_data)
        assert result == expected_output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## 常见问题

### Q: 测试失败，提示找不到模块

确保在测试文件中正确添加了项目路径：

```python
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
```

### Q: 如何跳过某些测试

使用 `@pytest.mark.skip` 装饰器：

```python
@pytest.mark.skip(reason="暂时跳过")
def test_something():
    pass
```

### Q: 如何测试预期会抛出异常的代码

使用 `pytest.raises`：

```python
def test_exception():
    with pytest.raises(ValueError):
        function_that_raises_error()
```

---

*更新日期：2026-04-14*
