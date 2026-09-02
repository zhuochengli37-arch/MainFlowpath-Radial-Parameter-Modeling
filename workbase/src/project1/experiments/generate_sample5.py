from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import shutil

import numpy as np

from project1.services.data_reader import read_curve_file
from project1.services.meta_parser import parse_metadata_from_path


@dataclass(frozen=True)
class CurveFile:
    component: str
    stage: int
    rpm: float
    wcor: float
    columns: list[str]
    rows: list[dict[str, float]]
    path: Path


def _iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name.lower().endswith(".bak"):
            continue
        if p.suffix.lower() not in {".txt", ".dat"}:
            continue
        files.append(p)
    return files


def _load_curves(input_dir: str) -> list[CurveFile]:
    root = Path(input_dir)
    result: list[CurveFile] = []
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
        result.append(
            CurveFile(
                component=component,
                stage=int(stage),
                rpm=float(rpm),
                wcor=float(wcor),
                columns=[str(c) for c in parsed["columns"]],
                rows=[dict(r) for r in parsed["rows"]],
                path=p,
            )
        )
    return result


def _write_curve_file(path: Path, columns: list[str], rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = " ".join(columns)
    lines = [header]
    for row in rows:
        values = [f"{float(row[col]):.6e}" for col in columns]
        lines.append(" ".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _group_key(item: CurveFile) -> tuple[str, int]:
    return (item.component, item.stage)


def generate_sample5(
    input_dir: str,
    output_dir: str,
    per_pair: int = 3,
    noise_ratio: float = 0.002,
    clamp_ratio: float = 0.02,
    seed: int = 42,
    clean_output: bool = False,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    curves = _load_curves(input_dir)
    out_root = Path(output_dir)
    if clean_output and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Keep original files in sample5.
    copied = 0
    src_root = Path(input_dir).resolve()
    for p in _iter_files(src_root):
        rel = p.resolve().relative_to(src_root)
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        copied += 1

    groups: dict[tuple[str, int], list[CurveFile]] = {}
    for item in curves:
        groups.setdefault(_group_key(item), []).append(item)

    synthetic_count = 0
    used_paths: set[str] = set()
    for key, items in groups.items():
        if len(items) < 2:
            continue
        sorted_items = sorted(items, key=lambda x: (x.rpm, x.wcor))
        for i in range(len(sorted_items) - 1):
            a = sorted_items[i]
            b = sorted_items[i + 1]
            if len(a.rows) == 0 or len(b.rows) == 0:
                continue
            m = min(len(a.rows), len(b.rows))
            cols = a.columns
            if cols != b.columns:
                continue
            for j in range(1, per_pair + 1):
                alpha = j / (per_pair + 1)
                rpm_new = (1.0 - alpha) * a.rpm + alpha * b.rpm
                wcor_new = (1.0 - alpha) * a.wcor + alpha * b.wcor

                rows_new: list[dict[str, float]] = []
                for ridx in range(m):
                    row_a = a.rows[ridx]
                    row_b = b.rows[ridx]
                    mixed: dict[str, float] = {}
                    for col in cols:
                        va = float(row_a[col])
                        vb = float(row_b[col])
                        base = (1.0 - alpha) * va + alpha * vb
                        lo, hi = min(va, vb), max(va, vb)
                        if col != "xi":
                            sigma = max(abs(base) * noise_ratio, 1e-9)
                            base = base + float(rng.normal(0.0, sigma))
                            span = max(abs(hi - lo), 1e-12)
                            pad = span * clamp_ratio
                            base = max(lo - pad, min(hi + pad, base))
                        else:
                            base = max(lo, min(hi, base))
                        mixed[col] = base
                    rows_new.append(mixed)

                dir_path = out_root / f"DATABASE_{key[0]}" / f"STAGE_{key[1]}" / f"RPM_{rpm_new:.3f}"
                file_path = dir_path / f"{wcor_new:.6f}.dat"
                file_key = str(file_path).lower()
                if file_key in used_paths or file_path.exists():
                    continue
                _write_curve_file(file_path, cols, rows_new)
                used_paths.add(file_key)
                synthetic_count += 1

    return {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "noise_ratio": noise_ratio,
        "clamp_ratio": clamp_ratio,
        "copied_original_files": copied,
        "synthetic_files": synthetic_count,
        "total_files": copied + synthetic_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate richer sample5 dataset from sample4-like data")
    parser.add_argument("--input-dir", default="./data/input/sample4", help="Source dataset directory")
    parser.add_argument("--output-dir", default="./data/input/sample5", help="Target dataset directory")
    parser.add_argument("--per-pair", type=int, default=3, help="Synthetic files generated between adjacent pairs")
    parser.add_argument("--noise-ratio", type=float, default=0.002, help="Relative Gaussian noise level")
    parser.add_argument("--clamp-ratio", type=float, default=0.02, help="Allowed excursion beyond interpolation segment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--clean-output", action="store_true", help="Delete output directory before generation")
    args = parser.parse_args()

    result = generate_sample5(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        per_pair=args.per_pair,
        noise_ratio=args.noise_ratio,
        clamp_ratio=args.clamp_ratio,
        seed=args.seed,
        clean_output=args.clean_output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

