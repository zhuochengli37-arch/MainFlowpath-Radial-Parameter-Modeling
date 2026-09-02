"""
元数据解析模块单元测试
"""

import pytest
from pathlib import Path
import sys

# 添加项目路径
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from project1.services.meta_parser import parse_metadata_from_path


class TestParseMetadataFromPath:
    """测试 parse_metadata_from_path 函数"""

    def test_parse_standard_path(self):
        """测试解析标准路径格式"""
        path = Path("data/current/3d/CASE1/train/DATABASE_CMP/STAGE_1/RPM_0.6/0.00681.txt")
        metadata = parse_metadata_from_path(path)

        assert metadata["database"] == "DATABASE_CMP"
        assert metadata["component"] == "CMP"
        assert metadata["stage"] == 1
        assert metadata["rpm"] == pytest.approx(0.6)
        assert metadata["wcor"] == pytest.approx(0.00681)

    def test_parse_fan_component(self):
        """测试解析 FAN 部件"""
        path = Path("DATABASE_FAN/STAGE_2/RPM_0.8/0.01234.txt")
        metadata = parse_metadata_from_path(path)

        assert metadata["component"] == "FAN"
        assert metadata["stage"] == 2
        assert metadata["rpm"] == pytest.approx(0.8)

    def test_parse_plain_component_before_stage(self):
        """测试没有 DATABASE_ 前缀的部件目录"""
        path = Path("DATACASEVAR_1/CMP/STAGE_1/RPM_0.6/0.00436.txt")
        metadata = parse_metadata_from_path(path)

        assert metadata["database"] == "DATABASE_CMP"
        assert metadata["component"] == "CMP"
        assert metadata["stage"] == 1
        assert metadata["rpm"] == pytest.approx(0.6)
        assert metadata["wcor"] == pytest.approx(0.00436)

    def test_parse_turbine_component(self):
        """测试解析涡轮部件"""
        path = Path("DATABASE_HPTB/STAGE_1/RPM_0.9/0.00500.txt")
        metadata = parse_metadata_from_path(path)

        assert metadata["component"] == "HPTB"
        assert metadata["stage"] == 1

    def test_parse_plain_numeric_rpm_directory(self):
        """测试 STAGE 后使用纯数字目录表示 rpm"""
        path = Path("DATABASE_HTB/STAGE_1/0.555/0.01509.txt")
        metadata = parse_metadata_from_path(path)

        assert metadata["component"] == "HTB"
        assert metadata["database"] == "DATABASE_HTB"
        assert metadata["stage"] == 1
        assert metadata["rpm"] == pytest.approx(0.555)
        assert metadata["wcor"] == pytest.approx(0.01509)

    def test_parse_lptb_component(self):
        """测试解析低压涡轮"""
        path = Path("DATABASE_LPTB/STAGE_3/RPM_0.7/0.00800.dat")
        metadata = parse_metadata_from_path(path)

        assert metadata["component"] == "LPTB"
        assert metadata["stage"] == 3
        assert metadata["rpm"] == pytest.approx(0.7)
        assert metadata["wcor"] == pytest.approx(0.00800)

    def test_parse_path_with_different_separators(self):
        """测试不同路径分隔符"""
        # Windows 风格路径
        path = Path("D:\\data\\DATABASE_CMP\\STAGE_1\\RPM_0.6\\0.00681.txt")
        metadata = parse_metadata_from_path(path)

        assert metadata["component"] == "CMP"
        assert metadata["rpm"] == pytest.approx(0.6)

    def test_parse_path_missing_components(self):
        """测试缺少某些组件的路径"""
        path = Path("some/random/path/file.txt")
        metadata = parse_metadata_from_path(path)

        # 应该返回默认值或 None
        assert metadata.get("component") is None or metadata.get("component") == ""
        assert metadata.get("rpm") is None or metadata.get("rpm") == 0.0

    def test_parse_scientific_notation_wcor(self):
        """测试科学计数法表示的 wcor"""
        path = Path("DATABASE_CMP/STAGE_1/RPM_0.6/1.234e-03.txt")
        metadata = parse_metadata_from_path(path)

        assert metadata["wcor"] == pytest.approx(0.001234)

    def test_parse_large_stage_number(self):
        """测试大级数"""
        path = Path("DATABASE_CMP/STAGE_10/RPM_0.6/0.00681.txt")
        metadata = parse_metadata_from_path(path)

        assert metadata["stage"] == 10


    def test_parse_inlet_database_component(self):
        path = Path("DATABASE_CMP_INLET/STAGE_0/CNC_0.78/W25COR_27.998.dat")
        metadata = parse_metadata_from_path(path)

        assert metadata["database"] == "DATABASE_CMP_INLET"
        assert metadata["component"] == "CMP"
        assert metadata["station"] == "INLET"
        assert metadata["stage"] == 0
        assert metadata["rpm"] == pytest.approx(0.78)
        assert metadata["wcor"] == pytest.approx(27.998)

    def test_parse_outlet_database_component(self):
        path = Path("DATABASE_FAN_OUTLET/STAGE_999/CNC_1.00/W2COR_13.200.dat")
        metadata = parse_metadata_from_path(path)

        assert metadata["database"] == "DATABASE_FAN_OUTLET"
        assert metadata["component"] == "FAN"
        assert metadata["station"] == "OUTLET"
        assert metadata["stage"] == 999
        assert metadata["rpm"] == pytest.approx(1.0)
        assert metadata["wcor"] == pytest.approx(13.2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
