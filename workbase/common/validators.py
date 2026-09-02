"""
AIR2 Project1 数据校验工具。

本模块提供输入数据、文件路径和模型参数的校验函数，
用于保证数据质量并减少运行时错误。
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional


class ValidationError(Exception):
    """数据校验失败时抛出的自定义异常。"""
    pass


def validate_file_exists(file_path: Path, file_type: str = "文件") -> None:
    """
    验证文件是否存在。

    参数:
        file_path: 文件路径
        file_type: 文件类型描述（用于错误消息）

    抛出:
        ValidationError: 如果文件不存在
    """
    if not file_path.exists():
        raise ValidationError(f"{file_type}不存在: {file_path}")
    if not file_path.is_file():
        raise ValidationError(f"路径不是文件: {file_path}")


def validate_directory_exists(dir_path: Path, dir_type: str = "目录") -> None:
    """
    验证目录是否存在。

    参数:
        dir_path: 目录路径
        dir_type: 目录类型描述（用于错误消息）

    抛出:
        ValidationError: 如果目录不存在
    """
    if not dir_path.exists():
        raise ValidationError(f"{dir_type}不存在: {dir_path}")
    if not dir_path.is_dir():
        raise ValidationError(f"路径不是目录: {dir_path}")


def validate_array_shape(
    array: np.ndarray,
    expected_dims: int,
    min_samples: int = 1,
    array_name: str = "数组"
) -> None:
    """
    验证数组形状。

    参数:
        array: NumPy数组
        expected_dims: 期望的维度数
        min_samples: 最小样本数
        array_name: 数组名称（用于错误消息）

    抛出:
        ValidationError: 如果数组形状不符合要求
    """
    if array.ndim != expected_dims:
        raise ValidationError(
            f"{array_name}维度错误: 期望{expected_dims}维，实际{array.ndim}维"
        )

    if len(array) < min_samples:
        raise ValidationError(
            f"{array_name}样本数不足: 期望至少{min_samples}个，实际{len(array)}个"
        )


def validate_no_nan(array: np.ndarray, array_name: str = "数组") -> None:
    """
    验证数组中没有NaN值。

    参数:
        array: NumPy数组
        array_name: 数组名称（用于错误消息）

    抛出:
        ValidationError: 如果数组包含NaN值
    """
    if np.isnan(array).any():
        nan_count = np.isnan(array).sum()
        raise ValidationError(f"{array_name}包含{nan_count}个NaN值")


def validate_no_inf(array: np.ndarray, array_name: str = "数组") -> None:
    """
    验证数组中没有无穷大值。

    参数:
        array: NumPy数组
        array_name: 数组名称（用于错误消息）

    抛出:
        ValidationError: 如果数组包含无穷大值
    """
    if np.isinf(array).any():
        inf_count = np.isinf(array).sum()
        raise ValidationError(f"{array_name}包含{inf_count}个无穷大值")


def validate_numeric_array(
    array: np.ndarray,
    array_name: str = "数组",
    allow_nan: bool = False,
    allow_inf: bool = False
) -> None:
    """
    验证数值数组的有效性。

    参数:
        array: NumPy数组
        array_name: 数组名称（用于错误消息）
        allow_nan: 是否允许NaN值
        allow_inf: 是否允许无穷大值

    抛出:
        ValidationError: 如果数组包含无效值
    """
    if not allow_nan:
        validate_no_nan(array, array_name)

    if not allow_inf:
        validate_no_inf(array, array_name)


def validate_column_exists(
    columns: List[str],
    required_column: str,
    file_name: str = "文件"
) -> None:
    """
    验证列是否存在。

    参数:
        columns: 可用的列名列表
        required_column: 必需的列名
        file_name: 文件名（用于错误消息）

    抛出:
        ValidationError: 如果列不存在
    """
    if required_column not in columns:
        raise ValidationError(
            f"{file_name}中缺少必需的列: {required_column}\n"
            f"可用的列: {', '.join(columns)}"
        )


def validate_columns_exist(
    columns: List[str],
    required_columns: List[str],
    file_name: str = "文件"
) -> None:
    """
    验证多个列是否存在。

    参数:
        columns: 可用的列名列表
        required_columns: 必需的列名列表
        file_name: 文件名（用于错误消息）

    抛出:
        ValidationError: 如果有列不存在
    """
    missing = [col for col in required_columns if col not in columns]
    if missing:
        raise ValidationError(
            f"{file_name}中缺少必需的列: {', '.join(missing)}\n"
            f"可用的列: {', '.join(columns)}"
        )


def validate_value_range(
    value: float,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    value_name: str = "值"
) -> None:
    """
    验证值是否在指定范围内。

    参数:
        value: 要验证的值
        min_val: 最小值（None表示无限制）
        max_val: 最大值（None表示无限制）
        value_name: 值的名称（用于错误消息）

    抛出:
        ValidationError: 如果值超出范围
    """
    if min_val is not None and value < min_val:
        raise ValidationError(f"{value_name}小于最小值: {value} < {min_val}")

    if max_val is not None and value > max_val:
        raise ValidationError(f"{value_name}大于最大值: {value} > {max_val}")


def validate_array_range(
    array: np.ndarray,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    array_name: str = "数组"
) -> None:
    """
    验证数组中所有值是否在指定范围内。

    参数:
        array: NumPy数组
        min_val: 最小值（None表示无限制）
        max_val: 最大值（None表示无限制）
        array_name: 数组名称（用于错误消息）

    抛出:
        ValidationError: 如果有值超出范围
    """
    if min_val is not None:
        below_min = array < min_val
        if below_min.any():
            count = below_min.sum()
            raise ValidationError(
                f"{array_name}中有{count}个值小于最小值{min_val}"
            )

    if max_val is not None:
        above_max = array > max_val
        if above_max.any():
            count = above_max.sum()
            raise ValidationError(
                f"{array_name}中有{count}个值大于最大值{max_val}"
            )


def validate_model_name(model_name: str, valid_models: List[str]) -> None:
    """
    验证模型名称是否有效。

    参数:
        model_name: 模型名称
        valid_models: 有效的模型名称列表

    抛出:
        ValidationError: 如果模型名称无效
    """
    if model_name not in valid_models:
        raise ValidationError(
            f"无效的模型名称: {model_name}\n"
            f"有效的模型: {', '.join(valid_models)}"
        )


def validate_input_output_match(
    x_samples: int,
    y_samples: int,
    x_name: str = "输入数据",
    y_name: str = "输出数据"
) -> None:
    """
    验证输入和输出样本数是否匹配。

    参数:
        x_samples: 输入样本数
        y_samples: 输出样本数
        x_name: 输入数据名称
        y_name: 输出数据名称

    抛出:
        ValidationError: 如果样本数不匹配
    """
    if x_samples != y_samples:
        raise ValidationError(
            f"{x_name}和{y_name}样本数不匹配: {x_samples} vs {y_samples}"
        )


def validate_training_data_3d(
    rpm_values: np.ndarray,
    wcor_values: np.ndarray,
    xi_values: np.ndarray,
    y_values: np.ndarray
) -> None:
    """
    验证三维训练数据的完整性。

    参数:
        rpm_values: RPM值数组
        wcor_values: Wcor值数组
        xi_values: Xi值数组
        y_values: 输出值数组

    抛出:
        ValidationError: 如果数据无效
    """
    # 验证数组形状
    validate_array_shape(rpm_values, 1, min_samples=1, array_name="RPM数据")
    validate_array_shape(wcor_values, 1, min_samples=1, array_name="Wcor数据")
    validate_array_shape(xi_values, 1, min_samples=1, array_name="Xi数据")
    validate_array_shape(y_values, 1, min_samples=1, array_name="输出数据")

    # 验证样本数匹配
    n_samples = len(rpm_values)
    validate_input_output_match(len(wcor_values), n_samples, "Wcor数据", "RPM数据")
    validate_input_output_match(len(xi_values), n_samples, "Xi数据", "RPM数据")
    validate_input_output_match(len(y_values), n_samples, "输出数据", "RPM数据")

    # 验证数值有效性
    validate_numeric_array(rpm_values, "RPM数据")
    validate_numeric_array(wcor_values, "Wcor数据")
    validate_numeric_array(xi_values, "Xi数据")
    validate_numeric_array(y_values, "输出数据")


def validate_training_data_1d(
    x_values: np.ndarray,
    y_values: np.ndarray
) -> None:
    """
    验证一维训练数据的完整性。

    参数:
        x_values: 输入值数组
        y_values: 输出值数组

    抛出:
        ValidationError: 如果数据无效
    """
    # 验证数组形状
    validate_array_shape(x_values, 2, min_samples=1, array_name="输入数据")
    validate_array_shape(y_values, 1, min_samples=1, array_name="输出数据")

    # 验证样本数匹配
    validate_input_output_match(len(x_values), len(y_values))

    # 验证数值有效性
    validate_numeric_array(x_values, "输入数据")
    validate_numeric_array(y_values, "输出数据")
