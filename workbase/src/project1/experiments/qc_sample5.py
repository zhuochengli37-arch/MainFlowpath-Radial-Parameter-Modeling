from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
from collections import defaultdict

import numpy as np

from project1.services.data_reader import read_curve_file
from project1.services.meta_parser import parse_metadata_from_path


TARGETS = ("psi", "tsi", "mai")


@dataclass(frozen=True)
class CurveRecord:
    component: str
    stage: int
    rpm: float
    wcor: float
    rows: list[dict[str, float]]
    columns: list[str]
    path: Path


def _iter_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name.lower().endswith(".bak"):
            continue
        if p.suffix.lower() not in {".txt", ".dat"}:
            continue
        out.append(p)
    return out


def _load_records(root_dir: str) -> list[CurveRecord]:
    root = Path(root_dir)
    records: list[CurveRecord] = []
    for p in _iter_files(root):
        meta = parse_metadata_from_path(str(p))
        component = str(meta.get("component") or "").upper()
        stage = meta.get("stage")
        rpm = meta.get("rpm")
        wcor = meta.get("wcor")
        if not component or stage is None or rpm is None or wcor is None:
            continue
        try:
            parsed = read_curve_file(str(p))
        except ValueError:
            continue
        rows = [dict(r) for r in parsed["rows"] if "xi" in r and all(t in r for t in TARGETS)]
        if not rows:
            continue
        records.append(
            CurveRecord(
                component=component,
                stage=int(stage),
                rpm=float(rpm),
                wcor=float(wcor),
                rows=rows,
                columns=[str(c) for c in parsed["columns"]],
                path=p,
            )
        )
    return records


def _stats(arr: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def _collect_values(records: list[CurveRecord]) -> dict[str, list[float]]:
    vals: dict[str, list[float]] = {"rpm": [], "wcor": [], "xi": [], "psi": [], "tsi": [], "mai": []}
    for rec in records:
        vals["rpm"].append(rec.rpm)
        vals["wcor"].append(rec.wcor)
        for row in rec.rows:
            vals["xi"].append(float(row["xi"]))
            for t in TARGETS:
                vals[t].append(float(row[t]))
    return vals


def _distribution_summary(base: list[CurveRecord], new: list[CurveRecord]) -> dict[str, object]:
    b = _collect_values(base)
    n = _collect_values(new)
    result: dict[str, object] = {}
    for key in b.keys():
        b_arr = np.array(b[key], dtype=float)
        n_arr = np.array(n[key], dtype=float)
        if len(b_arr) == 0 or len(n_arr) == 0:
            continue
        b_stats = _stats(b_arr)
        n_stats = _stats(n_arr)
        mean_shift = 0.0
        if abs(b_stats["mean"]) > 1e-12:
            mean_shift = (n_stats["mean"] - b_stats["mean"]) / abs(b_stats["mean"]) * 100.0
        result[key] = {
            "base": b_stats,
            "new": n_stats,
            "mean_shift_pct": float(mean_shift),
        }
    return result


def _global_bounds(records: list[CurveRecord]) -> dict[str, tuple[float, float]]:
    vals = _collect_values(records)
    bounds: dict[str, tuple[float, float]] = {}
    for k, arr in vals.items():
        if not arr:
            continue
        a = np.array(arr, dtype=float)
        bounds[k] = (float(np.min(a)), float(np.max(a)))
    return bounds


def _extreme_violation_rate(base: list[CurveRecord], new: list[CurveRecord]) -> dict[str, float]:
    bounds = _global_bounds(base)
    vals = _collect_values(new)
    out: dict[str, float] = {}
    for key, data in vals.items():
        if key not in bounds or len(data) == 0:
            continue
        lo, hi = bounds[key]
        arr = np.array(data, dtype=float)
        viol = np.sum((arr < lo) | (arr > hi))
        out[key] = float(viol / len(arr))
    return out


def _key(rec: CurveRecord) -> tuple[str, int]:
    return (rec.component, rec.stage)


def _interpolation_bounds_check(base: list[CurveRecord], new: list[CurveRecord]) -> dict[str, float]:
    base_by_group: dict[tuple[str, int], list[CurveRecord]] = defaultdict(list)
    for rec in base:
        base_by_group[_key(rec)].append(rec)
    for k in list(base_by_group.keys()):
        base_by_group[k] = sorted(base_by_group[k], key=lambda x: (x.rpm, x.wcor))

    checked = 0
    param_out = 0
    value_out = 0
    value_total = 0

    for rec in new:
        g = _key(rec)
        group = base_by_group.get(g, [])
        if len(group) < 2:
            continue
        # Find nearest neighbors around new point in sorted coordinate.
        coords = np.array([[it.rpm, it.wcor] for it in group], dtype=float)
        target = np.array([rec.rpm, rec.wcor], dtype=float)
        d = np.linalg.norm(coords - target, axis=1)
        idx = np.argsort(d)[:2]
        if len(idx) < 2:
            continue
        a = group[int(idx[0])]
        b = group[int(idx[1])]
        checked += 1

        rpm_lo, rpm_hi = min(a.rpm, b.rpm), max(a.rpm, b.rpm)
        w_lo, w_hi = min(a.wcor, b.wcor), max(a.wcor, b.wcor)
        if not (rpm_lo - 1e-12 <= rec.rpm <= rpm_hi + 1e-12 and w_lo - 1e-12 <= rec.wcor <= w_hi + 1e-12):
            param_out += 1

        m = min(len(a.rows), len(b.rows), len(rec.rows))
        if m <= 0:
            continue
        a_rows = sorted(a.rows, key=lambda r: float(r["xi"]))
        b_rows = sorted(b.rows, key=lambda r: float(r["xi"]))
        n_rows = sorted(rec.rows, key=lambda r: float(r["xi"]))
        for i in range(m):
            for t in TARGETS:
                lo = min(float(a_rows[i][t]), float(b_rows[i][t]))
                hi = max(float(a_rows[i][t]), float(b_rows[i][t]))
                v = float(n_rows[i][t])
                value_total += 1
                # 2% tolerance band for injected noise.
                tol = max(abs(hi - lo) * 0.02, 1e-12)
                if v < lo - tol or v > hi + tol:
                    value_out += 1

    return {
        "checked_files": float(checked),
        "param_out_of_segment_rate": 0.0 if checked == 0 else float(param_out / checked),
        "value_out_of_segment_rate": 0.0 if value_total == 0 else float(value_out / value_total),
    }


def run_qc(
    base_dir: str,
    sample5_dir: str,
    output_dir: str,
) -> dict[str, object]:
    base = _load_records(base_dir)
    new = _load_records(sample5_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result = {
        "base_dir": base_dir,
        "sample5_dir": sample5_dir,
        "base_files": len(base),
        "sample5_files": len(new),
        "distribution_summary": _distribution_summary(base, new),
        "extreme_violation_rate": _extreme_violation_rate(base, new),
        "interpolation_bounds_check": _interpolation_bounds_check(base, new),
    }

    json_path = out / "sample5_qc_report.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Sample5 QC Report",
        "",
        f"- base_dir: {base_dir}",
        f"- sample5_dir: {sample5_dir}",
        f"- base_files: {len(base)}",
        f"- sample5_files: {len(new)}",
        "",
        "## Extreme Violation Rate",
    ]
    for k, v in result["extreme_violation_rate"].items():
        lines.append(f"- {k}: {v:.6f}")
    lines.append("")
    lines.append("## Interpolation Bounds Check")
    ib = result["interpolation_bounds_check"]
    lines.append(f"- checked_files: {int(ib['checked_files'])}")
    lines.append(f"- param_out_of_segment_rate: {ib['param_out_of_segment_rate']:.6f}")
    lines.append(f"- value_out_of_segment_rate: {ib['value_out_of_segment_rate']:.6f}")
    lines.append("")
    lines.append("## Mean Shift (%)")
    ds = result["distribution_summary"]
    for key in ("rpm", "wcor", "xi", "psi", "tsi", "mai"):
        if key not in ds:
            continue
        lines.append(f"- {key}: {float(ds[key]['mean_shift_pct']):.3f}%")

    md_path = out / "sample5_qc_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result["json_report_path"] = str(json_path)
    result["markdown_report_path"] = str(md_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Quality check report for sample5 synthetic dataset")
    parser.add_argument("--base-dir", default="./data/input/sample4", help="Reference dataset directory")
    parser.add_argument("--sample5-dir", default="./data/input/sample5", help="Synthetic dataset directory")
    parser.add_argument("--output-dir", default="./data/output", help="Report output directory")
    args = parser.parse_args()

    res = run_qc(
        base_dir=args.base_dir,
        sample5_dir=args.sample5_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

