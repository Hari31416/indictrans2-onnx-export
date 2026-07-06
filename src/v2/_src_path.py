"""Expose parent ``src/`` on sys.path for v2 entry-point scripts.

Scripts under ``src/v2/`` are launched as ``python src/v2/foo.py``, so Python
prepends ``src/v2/`` to sys.path — not ``src/``. Shared modules such as
``onnx_bundle_optimize``, ``it2_inference``, and ``translate`` live in ``src/``.
Append the parent directory so those imports resolve while ``src/v2/`` keeps
priority for v2-local modules (e.g. ``it2_onnx_wrappers``).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[1]
_src_root = str(_SRC_ROOT)
if _src_root not in sys.path:
    sys.path.append(_src_root)
