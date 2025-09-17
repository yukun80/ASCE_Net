import os
from pathlib import Path
from typing import Any, Dict

import yaml

MODULE_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML config and expand relative paths against the module root."""
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = (MODULE_ROOT / cfg_path).resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    project_root = _resolve_path(MODULE_ROOT, config.get("project_root", ".."))
    data_cfg = config.get("data", {})
    for key in ["sar_root", "label_5ch_root", "label_2ch_root"]:
        if key in data_cfg:
            data_cfg[key] = str(_resolve_path(project_root, data_cfg[key]))
    config["data"] = data_cfg

    training_cfg = config.get("training", {})
    for key in ["save_dir", "log_dir"]:
        if key in training_cfg:
            training_cfg[key] = str(_resolve_path(MODULE_ROOT, training_cfg[key]))
    config["training"] = training_cfg

    config["project_root"] = str(project_root)
    return config


def ensure_dir(path: str) -> str:
    """Create directory if it does not exist and return the path."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return str(path_obj)
