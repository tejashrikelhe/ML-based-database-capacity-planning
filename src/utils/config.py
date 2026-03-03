from __future__ import annotations
import yaml
from pathlib import Path
from typing import Any, Dict

def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dirs(cfg: Dict[str, Any]) -> None:
    for k in ["raw_dir","processed_dir","model_dir","reports_dir"]:
        Path(cfg["data"][k]).mkdir(parents=True, exist_ok=True)
