
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
_GATEWAY_APP_PATH = PROJECT_ROOT / "api-gateway" / "main.py"
_GATEWAY_DIR = _GATEWAY_APP_PATH.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(_GATEWAY_DIR) not in sys.path:
    sys.path.insert(0, str(_GATEWAY_DIR))

_spec = importlib.util.spec_from_file_location("api_gateway_main", _GATEWAY_APP_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load API gateway app from {_GATEWAY_APP_PATH}")

_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

app = _module.app

__all__ = ["app"]
