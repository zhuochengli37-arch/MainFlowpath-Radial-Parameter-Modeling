"""
Tests for the tabular data readers using current project data formats.
"""

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parent.parent
WORKBASE_SRC = ROOT / "workbase" / "src"
if str(WORKBASE_SRC) not in sys.path:
    sys.path.insert(0, str(WORKBASE_SRC))

from project1.services.data_reader import read_curve_file, read_tabular_file


class TestReadTabularFile:
    def test_read_domain_specific_1d_datacase2(self):
        file_path = ROOT / "data" / "current" / "1d" / "sample6" / "train" / "example1.txt"

        result = read_tabular_file(str(file_path))

        assert result["columns"] == ["xi", "psi", "tsi", "mai"]
        assert result["row_count"] == 50
        assert len(result["rows"]) == 50
        assert result["rows"][0]["xi"] == pytest.approx(0.0)
        assert result["rows"][0]["psi"] == pytest.approx(1.088620e05)
        assert result["rows"][-1]["xi"] == pytest.approx(1.0)

    def test_read_generic_2d_datacase3(self):
        file_path = ROOT / "data" / "generic" / "2d" / "train" / "example.csv"

        result = read_tabular_file(str(file_path))

        assert result["columns"] == ["x1", "x2", "target"]
        assert result["row_count"] == 11
        assert result["rows"][0]["x1"] == pytest.approx(0.0)
        assert result["rows"][1]["x2"] == pytest.approx(1.0)
        assert result["rows"][-1]["target"] == pytest.approx(11.9)

    def test_read_generic_3d_datacase4(self):
        file_path = ROOT / "data" / "generic" / "3d" / "train" / "example.csv"

        result = read_tabular_file(str(file_path))

        assert result["columns"] == ["x1", "x2", "x3", "target"]
        assert result["row_count"] == 11
        assert result["rows"][4]["x3"] == pytest.approx(0.8)
        assert result["rows"][-1]["target"] == pytest.approx(11.8)

    def test_read_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            read_tabular_file("nonexistent_file.txt")

    def test_empty_file_raises_value_error(self, tmp_path: Path):
        file_path = tmp_path / "empty.txt"
        file_path.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="usable tabular content"):
            read_tabular_file(str(file_path))


class TestReadCurveFile:
    def test_read_curve_file_returns_current_structure(self):
        file_path = ROOT / "data" / "current" / "1d" / "sample6" / "train" / "example1.txt"

        result = read_curve_file(str(file_path))

        assert result["columns"] == ["xi", "psi", "tsi", "mai"]
        assert result["row_count"] == 50
        assert len(result["rows"]) == 50
        assert result["rows"][1]["tsi"] == pytest.approx(2.979658e02)

    def test_read_current_3d_prediction_style_xi_only_file(self):
        file_path = (
            ROOT
            / "data"
            / "current"
            / "3d"
            / "CASE1"
            / "predict"
            / "DATABASE_CMP"
            / "STAGE_1"
            / "RPM_0.6"
            / "0.00681.txt"
        )

        result = read_curve_file(str(file_path))

        assert result["columns"] == ["xi"]
        assert result["row_count"] == 50
        assert result["rows"][0]["xi"] == pytest.approx(0.0)
        assert result["rows"][-1]["xi"] == pytest.approx(1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
