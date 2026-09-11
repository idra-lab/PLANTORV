#!/usr/bin/env python3

"""The recorded marker transform file: where it lives, how it is shaped.

``save_marker_transforms`` writes it and ``static_marker_publisher``
reads it, so the format and its default location live here on their
own.

That default is the bringup package's ``config/static_transforms.yaml``,
the copy that gets committed. In this workspace the installed share
path is a symlink back to the source tree, so recording through it
updates the file under version control instead of a throwaway copy.

The file looks like this::

    parent_frame: camera_color_optical_frame
    recorded_at: 2026-09-10T17:00:00+00:00
    transforms:
      charuco_board:
        parent_frame: camera_color_optical_frame
        child_frame: charuco_board
        translation: {x: 0.0, y: 0.0, z: 0.0}
        rotation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
        samples: 30
        translation_spread_m: 0.0
        rotation_spread_deg: 0.0
"""

import os
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory

# The committed transforms live with the bringup that replays them.
DEFAULT_PACKAGE = "plantorv_bringup"
DEFAULT_RELATIVE_PATH = ("config", "static_transforms.yaml")

HEADER = (
    "# Marker transforms recorded by save_marker_transforms.\n"
    "# Replay them with static_marker_publisher.\n"
)


def default_file() -> str:
    """Path of the transform file the nodes use unless told otherwise.

    Resolved on call rather than at import, so the package lookup
    happens where a failure can be reported against the node that
    needed it.
    """
    return os.path.join(
        get_package_share_directory(DEFAULT_PACKAGE),
        *DEFAULT_RELATIVE_PATH,
    )


def resolve(path) -> Path:
    """Expand a configured path into an absolute one."""
    return Path(os.path.expanduser(str(path)))


def write(path, document) -> Path:
    """Write a transform document, creating its directory if needed."""
    resolved = resolve(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    with resolved.open("w") as handle:
        handle.write(HEADER)
        yaml.safe_dump(document, handle, sort_keys=False)

    return resolved


def read(path) -> dict:
    """Read a transform document, with the usual mistakes reported."""
    resolved = resolve(path)

    if not resolved.is_file():
        raise FileNotFoundError(
            f"No marker transforms at {resolved}. Record them once "
            "with save_marker_transforms, or point the file parameter "
            "at the file you have."
        )

    with resolved.open() as handle:
        document = yaml.safe_load(handle) or {}

    if not document.get("transforms"):
        raise ValueError(f"{resolved} has no transforms in it")

    return document
