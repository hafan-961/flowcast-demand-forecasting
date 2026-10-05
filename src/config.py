"""
Configuration loader.
"""
import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load the YAML configuration file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "params.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, "r") as f:
        return yaml.safe_load(f)
