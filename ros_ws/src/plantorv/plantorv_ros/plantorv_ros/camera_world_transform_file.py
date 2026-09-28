"""Location and YAML format for the captured camera-to-world transform."""

import os
from pathlib import Path
from typing import Union

import yaml
from ament_index_python.packages import get_package_share_directory

DEFAULT_PACKAGE = "plantorv_bringup"
DEFAULT_RELATIVE_PATH = ("config", "camera_to_world_transform.yaml")

HEADER = (
    "# Transform from static_camera_color_optical_frame into world.\n"
    "# Recorded by save_camera_world_transform. Re-record after moving the camera or robot.\n"
)


def default_file() -> str:
    """Return the shared calibration path used by the recorder by default."""
    return os.path.join(
        get_package_share_directory(DEFAULT_PACKAGE), *DEFAULT_RELATIVE_PATH
    )


def resolve(path: Union[str, Path]) -> Path:
    """Expand a configured path into an absolute filesystem path."""
    return Path(os.path.expanduser(str(path))).resolve()


def write(path: Union[str, Path], document: dict) -> Path:
    """Write a transform document, creating its parent directory if needed."""
    output = resolve(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        handle.write(HEADER)
        yaml.safe_dump(document, handle, sort_keys=False)
    return output
