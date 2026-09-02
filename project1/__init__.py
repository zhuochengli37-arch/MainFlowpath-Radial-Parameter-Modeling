from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path


__path__ = extend_path(__path__, __name__)

_SRC_PACKAGE = Path(__file__).resolve().parent.parent / "workbase" / "src" / "project1"
if _SRC_PACKAGE.is_dir():
    src_package = str(_SRC_PACKAGE)
    if src_package not in __path__:
        __path__.append(src_package)
