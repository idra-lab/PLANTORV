#!/usr/bin/env python3

"""The recorded marker transform file: where it lives, how it is shaped.

``save_marker_transforms`` writes it and ``static_marker_publisher``
reads it, and the launch files need its default path, so the format
lives here on its own. Nothing in this module imports OpenCV or ROS,
which keeps it cheap to import while a launch file is being evaluated.

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

DEFAULT_FILE = os.path.expanduser(
    "~/.ros/plantorv/marker_transforms.yaml"
)

HEADER = (
    "# Marker transforms recorded by save_marker_transforms.\n"
    "# Replay them with static_marker_publisher.\n"
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
