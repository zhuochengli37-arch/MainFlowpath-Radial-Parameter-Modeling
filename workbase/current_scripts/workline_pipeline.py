"""
Workline 场景一键流程：
- 训练：1D(rpm->wcor) + 2D(rpm, xi->outputs)
- 预测：1D(只读 rpm) + 2D(只读 rpm, xi)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKBASE = PROJECT_ROOT / "workbase"
SRC = WORKBASE / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from workbase.common.runtime_env import ensure_project_venv

if __name__ == "__main__":
    ensure_project_venv(PROJECT_ROOT)

from workbase.common.config_loader import load_config
from workbase.common.logger_config import setup_logger
from workbase.common.prediction_output import TabularTemplate, detect_tabular_delimiter, write_table_like_template
from workbase.current_scripts.benchmark_1d import main as benchmark_1d_main
from workbase.current_scripts.benchmark_2d import main as benchmark_2d_main
from workbase.current_scripts.predict_1d import main as predict_1d_main
from workbase.current_scripts.predict_2d import main as predict_2d_main

logger = setup_logger("workline_pipeline")
config = load_config("config/benchmark_config.yaml")

SUPPORTED_SUFFIXES = {".txt", ".dat", ".csv"}


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value)) if value is not None else default
    except (TypeError, ValueError):
        return default


def run_train() -> None:
    logger.info("[1/2] Train 1D (rpm -> wcor)")
    benchmark_1d_main()
    logger.info("[2/2] Train 2D (rpm, xi -> outputs)")
    benchmark_2d_main()


def run_predict() -> None:
    logger.info("[1/3] Predict 1D (read rpm only)")
    predict_1d_main()
    logger.info("[2/3] Predict 2D (read rpm + xi only)")
    predict_2d_main()
    logger.info("[3/3] Merge 1D + 2D prediction outputs")
    merge_predictions()


def _normalize_column_name(name: str) -> str:
    return str(name).strip().replace("\ufeff", "")


def _schema_alias_candidates(field: str) -> list[str]:
    aliases = config.schema_aliases.get(field, [])
    names = [field, *aliases]
    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        normalized = str(name).strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(normalized)
    return result


def _match_column(columns: list[str], logical_field: str) -> str | None:
    lowered = {_normalize_column_name(col).lower(): col for col in columns}
    for candidate in _schema_alias_candidates(logical_field):
        matched = lowered.get(_normalize_column_name(candidate).lower())
        if matched is not None:
            return matched
    return None


def _list_output_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)


def _read_rows_loose(file_path: Path) -> tuple[list[str], list[dict[str, object]]]:
    raw = file_path.read_text(encoding="utf-8", errors="ignore")
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return [], []

    delimiter = detect_tabular_delimiter(file_path)
    if delimiter == ",":
        splitter = lambda text: [part.strip() for part in text.split(",")]  # noqa: E731
    elif delimiter == "\t":
        splitter = lambda text: [part.strip() for part in text.split("\t")]  # noqa: E731
    else:
        splitter = lambda text: text.split()  # noqa: E731

    columns = [token for token in splitter(lines[0]) if token]
    rows: list[dict[str, object]] = []
    for line in lines[1:]:
        values = splitter(line)
        if not values:
            continue
        if len(values) < len(columns):
            if delimiter == "whitespace":
                values = [""] * (len(columns) - len(values)) + values
            else:
                values = values + [""] * (len(columns) - len(values))
        if len(values) > len(columns):
            values = values[: len(columns)]
        rows.append({columns[index]: values[index] for index in range(len(columns))})
    return columns, rows


def _merged_output_root() -> Path:
    base = Path(config.predict_2d_output_path)
    return base.parent / f"{base.name}_merged"


def merge_predictions() -> Path:
    root_1d = Path(config.predict_1d_output_path)
    root_2d = Path(config.predict_2d_output_path)
    output_root = _merged_output_root()
    output_root.mkdir(parents=True, exist_ok=True)

    files_2d = _list_output_files(root_2d)
    if not files_2d:
        raise FileNotFoundError(f"no 2D prediction files found under: {root_2d}")

    merged_count = 0
    missing_1d_files: list[str] = []
    row_mismatch_files: list[str] = []

    for file_2d in files_2d:
        relative = file_2d.relative_to(root_2d)
        file_1d = root_1d / relative

        columns_2d, rows_2d = _read_rows_loose(file_2d)
        if not columns_2d:
            logger.warning(f"skip empty 2D file: {file_2d}")
            continue
        wcor_col_2d = _match_column(columns_2d, "wcor")
        if wcor_col_2d is None:
            wcor_col_2d = "wcor"
            columns_2d.append(wcor_col_2d)
        rows_1d: list[dict[str, object]] = []
        wcor_col_1d: str | None = None

        if file_1d.exists():
            columns_1d, rows_1d = _read_rows_loose(file_1d)
            wcor_col_1d = _match_column([str(col) for col in columns_1d], "wcor")
        else:
            missing_1d_files.append(str(relative))

        if rows_1d and len(rows_1d) != len(rows_2d):
            row_mismatch_files.append(str(relative))

        for index, row_2d in enumerate(rows_2d):
            if index < len(rows_1d) and wcor_col_1d is not None:
                wcor_value = rows_1d[index].get(wcor_col_1d)
                if wcor_value is None or str(wcor_value).strip() == "":
                    non_empty_values = [value for value in rows_1d[index].values() if str(value).strip() != ""]
                    if non_empty_values:
                        wcor_value = non_empty_values[-1]
                if wcor_value is not None and str(wcor_value).strip() != "":
                    row_2d[wcor_col_2d] = wcor_value

        template = TabularTemplate(
            source_path=file_2d,
            columns=[_normalize_column_name(col) for col in columns_2d],
            delimiter=detect_tabular_delimiter(file_2d),
            suffix=file_2d.suffix or ".dat",
        )
        output_file = (output_root / relative).with_suffix(template.suffix)
        write_table_like_template(output_file, template, rows_2d)
        merged_count += 1

    summary = {
        "source_1d": str(root_1d),
        "source_2d": str(root_2d),
        "merged_output": str(output_root),
        "merged_files": merged_count,
        "missing_1d_files": sorted(missing_1d_files),
        "row_mismatch_files": sorted(row_mismatch_files),
    }
    summary_path = output_root / "merge_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Merged prediction files: {merged_count}")
    logger.info(f"Merged output: {output_root}")
    if missing_1d_files:
        logger.warning(f"Missing 1D peers: {len(missing_1d_files)}")
    if row_mismatch_files:
        logger.warning(f"Row count mismatch files: {len(row_mismatch_files)}")
    return output_root


def _load_json(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def write_evaluation_summary() -> Path:
    output_root = _merged_output_root()
    output_root.mkdir(parents=True, exist_ok=True)

    leaderboard_1d_path = Path(config.benchmark_1d_output_dir) / "leaderboard.json"
    leaderboard_2d_path = Path(config.benchmark_2d_output_dir) / f"leaderboard_2d_{config.radial_mode_2d}.json"
    rows_1d = _load_json(leaderboard_1d_path)
    rows_2d = _load_json(leaderboard_2d_path)

    grouped_1d: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows_1d:
        partition = str(row.get("partition") or "legacy_all")
        grouped_1d[partition].append(row)

    best_1d_by_partition: list[dict[str, object]] = []
    for partition, entries in grouped_1d.items():
        best = min(entries, key=lambda item: float(item.get("rmse", float("inf"))))
        best_1d_by_partition.append(
            {
                "partition": partition,
                "component": best.get("component"),
                "stage": best.get("stage"),
                "target": best.get("target"),
                "model": best.get("model"),
                "rmse": best.get("rmse"),
                "mape": best.get("mape"),
                "mae": best.get("mae"),
                "r2": best.get("r2"),
                "folds": best.get("folds"),
                "partition_size": best.get("partition_size"),
            }
        )
    best_1d_by_partition.sort(key=lambda item: str(item["partition"]))

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows_2d:
        partition = str(row.get("partition", ""))
        target = str(row.get("target", ""))
        if partition and target:
            grouped[(partition, target)].append(row)

    best_2d: list[dict[str, object]] = []
    for (partition, target), entries in grouped.items():
        random_entries = [entry for entry in entries if str(entry.get("split_mode")) == "random"]
        candidates = random_entries or entries
        best = min(candidates, key=lambda item: float(item.get("rmse", float("inf"))))
        best_2d.append(
            {
                "partition": partition,
                "target": target,
                "model": best.get("model"),
                "split_mode": best.get("split_mode"),
                "rmse": best.get("rmse"),
                "mape": best.get("mape"),
                "mae": best.get("mae"),
                "r2": best.get("r2"),
                "folds": best.get("folds"),
                "partition_size": best.get("partition_size"),
            }
        )
    best_2d.sort(key=lambda item: (str(item["partition"]), str(item["target"])))

    payload = {
        "benchmark_1d_leaderboard": str(leaderboard_1d_path),
        "benchmark_2d_leaderboard": str(leaderboard_2d_path),
        "best_1d": best_1d_by_partition[0] if len(best_1d_by_partition) == 1 else None,
        "best_1d_by_partition": best_1d_by_partition,
        "best_2d": best_2d,
    }
    json_path = output_root / "workline_evaluation_summary.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: list[str] = ["# Workline Pipeline Evaluation Summary", ""]
    if best_1d_by_partition:
        lines.extend(
            [
                "## Best 1D by Partition (rpm -> wcor)",
                "",
                "| partition | model | rmse | mape | mae | r2 | folds | samples |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in best_1d_by_partition:
            lines.append(
                f"| {item['partition']} | `{item['model']}` | {_as_float(item['rmse']):.6f} | "
                f"{_as_float(item['mape']):.6f} | {_as_float(item['mae']):.6f} | "
                f"{_as_float(item['r2']):.6f} | {_as_int(item['folds'])} | "
                f"{_as_int(item['partition_size'])} |"
            )
        lines.append("")
    else:
        lines.extend(["## Best 1D by Partition (rpm -> wcor)", "", "- No 1D leaderboard found.", ""])

    lines.extend(["## Best 2D by Partition/Target", ""])
    if best_2d:
        lines.append("| partition | target | model | split | rmse | mape | mae | r2 | folds | samples |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|")
        for item in best_2d:
            lines.append(
                f"| {item['partition']} | {item['target']} | `{item['model']}` | {item['split_mode']} | "
                f"{_as_float(item['rmse']):.6f} | {_as_float(item['mape']):.6f} | {_as_float(item['mae']):.6f} | "
                f"{_as_float(item['r2']):.6f} | {_as_int(item['folds'])} | {_as_int(item['partition_size'])} |"
            )
    else:
        lines.append("- No 2D leaderboard found.")

    md_path = output_root / "workline_evaluation_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Evaluation summary: {md_path}")
    return md_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run current workline scenario pipeline")
    parser.add_argument(
        "--mode",
        choices=("all", "train", "predict"),
        default="all",
        help="Pipeline mode (default: all)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logger.info("=" * 70)
    logger.info(f"Workline pipeline started (mode={args.mode})")
    logger.info("=" * 70)
    if args.mode in {"all", "train"}:
        run_train()
    if args.mode in {"all", "predict"}:
        run_predict()
    write_evaluation_summary()
    logger.info("=" * 70)
    logger.info("[OK] Workline pipeline completed")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
