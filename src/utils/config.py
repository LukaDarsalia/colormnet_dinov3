from __future__ import annotations

from pathlib import Path
import copy
from typing import Any, Dict, Union

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    base = data.get("_base_")
    if base:
        base_paths = base if isinstance(base, list) else [base]
        merged: Dict[str, Any] = {}
        for base_path in base_paths:
            base_cfg = load_config(path.parent / base_path)
            merged = _deep_merge(merged, base_cfg)
        data = _deep_merge(merged, {k: v for k, v in data.items() if k != "_base_"})
    return data
