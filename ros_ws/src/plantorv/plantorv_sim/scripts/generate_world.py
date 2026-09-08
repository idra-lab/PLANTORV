#!/usr/bin/env python3
"""
Copyright 2025 Enrico Saccon

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

__maintainers__ = ["Enrico Saccon", "Davide De Martini", "Marco Roveri", "Davide Nardi"]

"""Regenerate ``worlds/plantorv_table.world`` from ``config/scene.yaml``.

A development tool, not a node: the world file is committed, and this is what
rewrites it after the yaml changes. It needs nothing from ROS, so it runs in any
Python with PyYAML::

    python3 scripts/generate_world.py
"""

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from plantorv_sim.scene import Scene  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=PACKAGE_ROOT / "config" / "scene.yaml")
    parser.add_argument(
        "--output", type=Path, default=PACKAGE_ROOT / "worlds" / "plantorv_table.world"
    )
    args = parser.parse_args()

    scene = Scene.from_file(args.scene)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(scene.world_sdf(), encoding="utf-8")

    print(f"wrote {args.output} from {args.scene}")
    print(f"  table top at z = {scene.table_height:.3f} m")
    print(f"  robot base_link at z = {scene.mount_height:.3f} m, xy = {scene.mount_xy}")
    for obj in scene:
        print(f"  {obj.type:<9} {obj.name:<12} at {tuple(round(v, 3) for v in obj.position)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
