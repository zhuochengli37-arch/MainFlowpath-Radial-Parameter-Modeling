"""
Tests for validation helpers using the current interfaces.
"""

from pathlib import Path
import sys
import tempfile

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parent.parent
WORKBASE_COMMON = ROOT / "workbase" / "common"
if str(WORKBASE_COMMON) not in sys.path:
    sys.path.insert(0, str(WORKBASE_COMMON))

from validators import (
    ValidationError,
    validate_array_shape,
    validate_column_exists,
    validate_directory_exists,
    validate_file_exists,
    validate_input_output_match,
    validate_numeric_array,
)


class TestValidateFileExists:
    def test_existing_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = Path(f.name)

        try:
            validate_file_exists(temp_path, "测试文件")
        finally:
            temp_path.unlink()

    def test_nonexistent_file(self):
        with pytest.raises(ValidationError):
            validate_file_exists(Path("nonexistent.txt"), "测试文件")

    def test_directory_instead_of_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValidationError):
                validate_file_exists(Path(temp_dir), "测试文件")


class TestValidateDirectoryExists:
    def test_existing_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            validate_directory_exists(Path(temp_dir), "测试目录")

    def test_nonexistent_directory(self):
        with pytest.raises(ValidationError):
            validate_directory_exists(Path("nonexistent_dir"), "测试目录")

    def test_file_instead_of_directory(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValidationError):
                validate_directory_exists(temp_path, "测试目录")
        finally:
            temp_path.unlink()


class TestValidateArrayShape:
    def test_valid_2d_array(self):
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        validate_array_shape(arr, expected_dims=2, min_samples=2, array_name="测试数组")

    def test_invalid_dimensions(self):
        arr = np.array([1, 2, 3])
        with pytest.raises(ValidationError):
            validate_array_shape(arr, expected_dims=2, array_name="测试数组")

    def test_insufficient_samples(self):
        arr = np.array([[1, 2]])
        with pytest.raises(ValidationError):
            validate_array_shape(arr, expected_dims=2, min_samples=5, array_name="测试数组")

    def test_empty_array(self):
        arr = np.array([])
        with pytest.raises(ValidationError):
            validate_array_shape(arr, expected_dims=1, min_samples=1, array_name="测试数组")


class TestValidateNumericArray:
    def test_valid_numeric_array(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        validate_numeric_array(arr, array_name="测试数组")

    def test_array_with_nan(self):
        arr = np.array([[1.0, np.nan], [3.0, 4.0]])
        with pytest.raises(ValidationError):
            validate_numeric_array(arr, array_name="测试数组")

    def test_array_with_inf(self):
        arr = np.array([[1.0, np.inf], [3.0, 4.0]])
        with pytest.raises(ValidationError):
            validate_numeric_array(arr, array_name="测试数组")

    def test_integer_array(self):
        arr = np.array([[1, 2], [3, 4]])
        validate_numeric_array(arr, array_name="测试数组")


class TestValidateColumnExists:
    def test_existing_column(self):
        columns = ["xi", "psi", "tsi", "mai"]
        validate_column_exists(columns, "psi", "测试文件")

    def test_nonexistent_column(self):
        columns = ["xi", "psi", "tsi"]
        with pytest.raises(ValidationError, match="缺少必需的列"):
            validate_column_exists(columns, "mai", "测试文件")

    def test_empty_columns_list(self):
        columns = []
        with pytest.raises(ValidationError, match="缺少必需的列"):
            validate_column_exists(columns, "xi", "测试文件")


class TestValidateInputOutputMatch:
    def test_matching_lengths(self):
        validate_input_output_match(3, 3)

    def test_mismatched_lengths(self):
        with pytest.raises(ValidationError):
            validate_input_output_match(3, 2)

    def test_empty_lengths(self):
        validate_input_output_match(0, 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
