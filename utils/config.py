from __future__ import annotations

import copy
from typing import Any, Dict
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file into a nested dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base (base not modified)."""
    base = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base


def set_by_dotted_path(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    """Set cfg['a']['b']['c'] given dotted='a.b.c'."""
    cur = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value
