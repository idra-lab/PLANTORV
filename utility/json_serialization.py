from pathlib import Path
from typing import Any

import numpy as np


def to_json_compatible(value: Any) -> Any:
    """Convert pipeline-specific values for use by ``json.dump(default=...)``."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
