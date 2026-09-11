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

__maintainers__ = ["Enrico Saccon", "Tommaso Faraci"]

"""``config/scene.yaml`` read into objects, and turned back into SDF.

The workcell is described once, in ``config/scene.yaml``. This module is the
only thing that parses it, so the Gazebo world, the MoveIt planning scene and
the pose queries the planner makes all agree on where things are by
construction rather than by three people editing three files.

Poses here are always the centre of the object's bounding box, expressed in the
frame named by ``frame_id`` in the yaml (``world`` -- on the floor, at the
centre of the table top). ``dimensions`` are full extents, never half-extents:
Gazebo's ``<box><size>`` and MoveIt's ``SolidPrimitive.BOX`` both want the full
extent, and having exactly one convention removes the factor of two that
otherwise creeps in somewhere.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

# The three kinds of thing in the scene. Cubes are the only ones that move, and
# so the only ones spawned as their own Gazebo model; the rest is furniture.
TYPE_CUBE = "cube"
TYPE_TRAY = "tray"
TYPE_FURNITURE = "furniture"


@dataclass
class SceneObject:
    """One box-shaped thing in the workcell.

    Attributes
    ----------
    name : str
        Unique, and also the Gazebo model name and the MoveIt collision object
        id. One name means one thing, everywhere.
    type : str
        ``cube``, ``tray`` or ``furniture``.
    position : tuple[float, float, float]
        Centre of the bounding box, in the scene frame.
    yaw : float
        Rotation about z, radians. Nothing in the scene is tilted.
    dimensions : tuple[float, float, float]
        Full extents of the bounding box.
    color : tuple[float, float, float, float]
        RGBA, 0-1.
    mass : float
        Only meaningful for the movable objects; 0 marks a static model.
    parts : list[tuple]
        For composite objects (a tray is a floor and four walls), the boxes it
        is made of, each ``(offset_xyz, size_xyz)`` relative to ``position``.
        Empty means the object is the single box given by ``dimensions``.
    """

    name: str
    type: str
    position: Tuple[float, float, float]
    yaw: float = 0.0
    dimensions: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    color: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    mass: float = 0.0
    parts: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = field(
        default_factory=list
    )

    @property
    def top_z(self) -> float:
        """Height of the object's upper surface, in the scene frame."""
        return self.position[2] + self.dimensions[2] / 2.0

    def boxes(self):
        """Yield ``(centre, size)`` for every box the object is made of."""
        if not self.parts:
            yield self.position, self.dimensions
            return
        for offset, size in self.parts:
            centre = (
                self.position[0] + offset[0],
                self.position[1] + offset[1],
                self.position[2] + offset[2],
            )
            yield centre, size


class Scene:
    """The parsed workcell.

    Parameters
    ----------
    config : dict
        The contents of ``scene.yaml``.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.frame_id: str = config.get("frame_id", "world")
        self._objects: Dict[str, SceneObject] = {}
        self._build()

    # -- construction ----------------------------------------------------

    @classmethod
    def from_file(cls, path) -> "Scene":
        """Read a scene from a ``scene.yaml`` on disk."""
        with open(path, "r", encoding="utf-8") as handle:
            return cls(yaml.safe_load(handle))

    @classmethod
    def from_package(cls, package: str = "plantorv_sim") -> "Scene":
        """Read the scene installed in ``<package>/share/config/scene.yaml``."""
        from ament_index_python.packages import get_package_share_directory

        return cls.from_file(Path(get_package_share_directory(package)) / "config" / "scene.yaml")

    def to_world(self, xy: Sequence[float], frame: str = "world"):
        """Convert an (x, y) measured in ``frame`` into the scene frame.

        ``world`` returns it unchanged. ``base`` treats it as the pendant does:
        the UR controller's own Base feature frame, which is where a jogged TCP
        position is read off.

        That frame is the arm's base_link turned by pi -- ur_description adds
        that rotation so `base` matches the UR controller's convention -- and
        base_link is itself turned by ``robot.yaw`` relative to the scene. So
        the rotation between the pendant and here is yaw + pi, and with the
        yaw of pi this cell uses those cancel and the numbers are the same.
        Which is convenient and is exactly why it is worth doing properly: the
        day the arm is bolted on at a different angle, the numbers stop
        matching and nothing would have said so.

        Only x and y. Height is not taken from a measurement at all -- objects
        are placed sitting on the table.
        """
        x, y = (float(v) for v in xy)
        if frame == "world":
            return x, y
        if frame != "base":
            raise ValueError(
                f"unknown frame '{frame}' for an object position; "
                f"use 'world' (the default) or 'base' (what the pendant shows)"
            )
        angle = self.mount_yaw + math.pi
        cos, sin = math.cos(angle), math.sin(angle)
        base_x, base_y = self.mount_xy
        return base_x + cos * x - sin * y, base_y + sin * x + cos * y

    def _build(self) -> None:
        # The table and the pedestal both stand on the floor and neither
        # carries the other, so each is placed from its own xy and its own
        # height. They used to be nested -- pedestal on table, table at the
        # origin -- and the arithmetic below is what changed when the frame
        # moved to the robot's base.
        table = self.config["table"]
        tw, td, th = (float(v) for v in table["size"])
        tx, ty = (float(v) for v in table.get("xy", [0.0, 0.0]))
        self._add(
            SceneObject(
                name="table",
                type=TYPE_FURNITURE,
                position=(tx, ty, th / 2.0),
                dimensions=(tw, td, th),
                color=tuple(table.get("color", [0.7, 0.6, 0.45, 1.0])),
            )
        )

        robot = self.config["robot"]
        sw, sd, sh = (float(v) for v in robot["stand_size"])
        sx, sy = (float(v) for v in robot["stand_xy"])
        self._add(
            SceneObject(
                name="robot_stand",
                type=TYPE_FURNITURE,
                position=(sx, sy, sh / 2.0),
                dimensions=(sw, sd, sh),
                color=tuple(robot.get("stand_color", [0.35, 0.35, 0.38, 1.0])),
            )
        )

        for tray in self.config.get("trays", []):
            self._add(self._make_tray(tray, table_top=th, scene=self))

        cubes = self.config.get("cubes", {})
        # `size` is either one number, for a cube, or the three full extents.
        # The blocks in this cell are 3 x 3 x 6 cm, so they are not cubes; the
        # scalar form is kept because it reads better when they are, and
        # because it is what every earlier scene.yaml wrote.
        size = cubes.get("size", 0.1)
        if isinstance(size, (list, tuple)):
            bw, bd, bh = (float(v) for v in size)
        else:
            bw = bd = bh = float(size)
        mass = float(cubes.get("mass", 0.2))
        for cube in cubes.get("items", []):
            cx, cy = self.to_world(cube["xy"], cube.get("frame", "world"))
            self._add(
                SceneObject(
                    name=cube["name"],
                    type=TYPE_CUBE,
                    position=(cx, cy, th + bh / 2.0),
                    yaw=float(cube.get("yaw", 0.0)),
                    dimensions=(bw, bd, bh),
                    color=tuple(cube.get("color", [0.2, 0.4, 0.9, 1.0])),
                    mass=mass,
                )
            )

    @staticmethod
    def _make_tray(tray: Dict, table_top: float, scene: "Scene") -> SceneObject:
        """Build a tray out of a floor and four walls."""
        iw, idp, ih = (float(v) for v in tray["inner"])
        wall = float(tray.get("wall", 0.015))
        ow, od, oh = iw + 2 * wall, idp + 2 * wall, ih + wall
        x, y = scene.to_world(tray["xy"], tray.get("frame", "world"))

        # Offsets are from the centre of the outer bounding box, whose bottom
        # rests on the table top.
        bottom = -oh / 2.0 + wall / 2.0
        walls = ih / 2.0 - oh / 2.0 + wall
        parts = [
            ((0.0, 0.0, bottom), (ow, od, wall)),
            ((0.0, (idp + wall) / 2.0, walls), (ow, wall, ih)),
            ((0.0, -(idp + wall) / 2.0, walls), (ow, wall, ih)),
            (((iw + wall) / 2.0, 0.0, walls), (wall, idp, ih)),
            ((-(iw + wall) / 2.0, 0.0, walls), (wall, idp, ih)),
        ]
        return SceneObject(
            name=tray["name"],
            type=TYPE_TRAY,
            position=(x, y, table_top + oh / 2.0),
            dimensions=(ow, od, oh),
            color=tuple(tray.get("color", [0.5, 0.5, 0.5, 1.0])),
            parts=parts,
        )

    def _add(self, obj: SceneObject) -> None:
        if obj.name in self._objects:
            raise ValueError(f"duplicate scene object name: {obj.name}")
        self._objects[obj.name] = obj

    # -- queries ---------------------------------------------------------

    def __iter__(self):
        return iter(self._objects.values())

    def __len__(self) -> int:
        return len(self._objects)

    def get(self, name: str) -> Optional[SceneObject]:
        """Return the object called ``name``, or None."""
        return self._objects.get(name)

    def of_type(self, type_name: str) -> List[SceneObject]:
        """Return every object of the given type, in declaration order."""
        return [o for o in self._objects.values() if o.type == type_name]

    @property
    def table_height(self) -> float:
        """Height of the table's top surface above the floor."""
        return self._objects["table"].top_z

    @property
    def mount_height(self) -> float:
        """Height of the robot's ``base_link`` above the floor."""
        return float(self.config["robot"]["mount_height"])

    @property
    def mount_xy(self) -> Tuple[float, float]:
        """Where on the floor the robot's pedestal stands."""
        x, y = self.config["robot"]["stand_xy"]
        return float(x), float(y)

    @property
    def table_xy(self) -> Tuple[float, float]:
        """Centre of the table top, in the scene frame."""
        x, y = self.config["table"].get("xy", [0.0, 0.0])
        return float(x), float(y)

    @property
    def table_near_edge(self) -> float:
        """The y of the table edge nearest the robot.

        The reachable part of the table starts here, so it is worth having a
        name: everything the arm works on has to sit between this and the
        limit of its reach.
        """
        return self.table_xy[1] - self._objects["table"].dimensions[1] / 2.0

    @property
    def mount_yaw(self) -> float:
        """Yaw of the robot base about z."""
        return float(self.config["robot"].get("yaw", 0.0))

    # -- SDF -------------------------------------------------------------

    def model_sdf(self, obj: SceneObject, static: Optional[bool] = None) -> str:
        """Return a standalone ``<sdf>`` document for one object, at the origin.

        This is what ``/spawn_entity`` is given. The model is emitted at the
        origin with no rotation, and the caller says where it goes through the
        request's ``initial_pose``.

        The pose must not be written here as well. Gazebo composes
        ``initial_pose`` with the pose in the SDF rather than overriding it, so
        an absolute pose in both places is applied twice: a cube meant for
        (-0.26, 0.26, 1.05) is created at (-0.52, 0.52, 2.10), which is past
        the edge of the table, and falls on the floor. Leaving the pose out
        also lets ``scene_manager`` respawn a released object wherever the tool
        happens to be, which is the other reason this document is generated.
        """
        if static is None:
            static = obj.mass <= 0.0
        return (
            '<?xml version="1.0" ?>\n'
            '<sdf version="1.6">\n'
            f"{self._model_element(obj, static, indent='  ', at_origin=True)}"
            "</sdf>\n"
        )

    def _model_element(
        self, obj: SceneObject, static: bool, indent: str = "", at_origin: bool = False
    ) -> str:
        # The boxes are always laid out around the object's own centre; only the
        # model pose changes, so that a spawned model can be placed by the
        # caller while one embedded in the world file carries its own position.
        px, py, pz = obj.position
        mx, my, mz, myaw = (0.0, 0.0, 0.0, 0.0) if at_origin else (px, py, pz, obj.yaw)
        body = [
            f'{indent}<model name="{obj.name}">',
            f"{indent}  <static>{'true' if static else 'false'}</static>",
            f"{indent}  <pose>{mx:.6f} {my:.6f} {mz:.6f} 0 0 {myaw:.6f}</pose>",
            f'{indent}  <link name="base_link">',
        ]
        if not static:
            body.append(self._inertial(obj, indent + "    "))
        for index, (centre, size) in enumerate(obj.boxes()):
            # Boxes are placed relative to the model pose, which already sits at
            # the object's centre.
            offset = (centre[0] - px, centre[1] - py, centre[2] - pz)
            body.append(self._box(f"{obj.name}_{index}", offset, size, obj.color, indent + "    "))
        body += [f"{indent}  </link>", f"{indent}</model>", ""]
        return "\n".join(body)

    @staticmethod
    def _inertial(obj: SceneObject, indent: str) -> str:
        """Solid-box inertia, so dropped cubes fall like boxes."""
        m = obj.mass
        w, d, h = obj.dimensions
        ixx = m * (d * d + h * h) / 12.0
        iyy = m * (w * w + h * h) / 12.0
        izz = m * (w * w + d * d) / 12.0
        return (
            f"{indent}<inertial>\n"
            f"{indent}  <mass>{m:.6f}</mass>\n"
            f"{indent}  <inertia>\n"
            f"{indent}    <ixx>{ixx:.8f}</ixx><ixy>0</ixy><ixz>0</ixz>\n"
            f"{indent}    <iyy>{iyy:.8f}</iyy><iyz>0</iyz>\n"
            f"{indent}    <izz>{izz:.8f}</izz>\n"
            f"{indent}  </inertia>\n"
            f"{indent}</inertial>"
        )

    @staticmethod
    def _box(
        name: str,
        offset: Sequence[float],
        size: Sequence[float],
        color: Sequence[float],
        indent: str,
    ) -> str:
        ox, oy, oz = offset
        sx, sy, sz = size
        r, g, b, a = color
        pose = f"{ox:.6f} {oy:.6f} {oz:.6f} 0 0 0"
        geometry = f"<geometry><box><size>{sx:.6f} {sy:.6f} {sz:.6f}</size></box></geometry>"
        return (
            f'{indent}<collision name="{name}_collision">\n'
            f"{indent}  <pose>{pose}</pose>\n"
            f"{indent}  {geometry}\n"
            f"{indent}  <surface><friction><ode>\n"
            f"{indent}    <mu>1.0</mu><mu2>1.0</mu2>\n"
            f"{indent}  </ode></friction></surface>\n"
            f"{indent}</collision>\n"
            f'{indent}<visual name="{name}_visual">\n'
            f"{indent}  <pose>{pose}</pose>\n"
            f"{indent}  {geometry}\n"
            f"{indent}  <material>\n"
            f"{indent}    <ambient>{r:.3f} {g:.3f} {b:.3f} {a:.3f}</ambient>\n"
            f"{indent}    <diffuse>{r:.3f} {g:.3f} {b:.3f} {a:.3f}</diffuse>\n"
            f"{indent}  </material>\n"
            f"{indent}</visual>"
        )

    def world_sdf(self) -> str:
        """Return the whole Gazebo world: furniture and trays, but no cubes.

        The cubes are left out on purpose. They are spawned at run time by
        ``scene_spawner``, which is also what respawns them when the fake grasp
        releases one, so there is a single code path putting a cube in the
        world.

        The table and the stand are left out as well: they are part of the
        robot's URDF, which is what mounts ``base_link`` at the right height,
        and spawning them twice would have the arm collide with a copy of its
        own stand.
        """
        models = "".join(
            self._model_element(obj, static=True, indent="    ") for obj in self.of_type(TYPE_TRAY)
        )
        return _WORLD_TEMPLATE.format(models=models)


_WORLD_TEMPLATE = """<?xml version="1.0" ?>
<!--
  Generated by plantorv_sim/scripts/generate_world.py from config/scene.yaml.
  Do not edit: edit the yaml and regenerate.
-->
<sdf version="1.6">
  <world name="plantorv">

    <physics name="default_physics" default="true" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Ground-truth object poses on /gazebo/model_states, which world_model
         answers pose queries from. -->
    <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
      <ros>
        <namespace>/gazebo</namespace>
      </ros>
      <update_rate>20.0</update_rate>
    </plugin>

    <include><uri>model://sun</uri></include>
    <include><uri>model://ground_plane</uri></include>

    <scene>
      <ambient>0.6 0.6 0.6 1</ambient>
      <shadows>true</shadows>
    </scene>

    <gui>
      <camera name="user_camera">
        <pose>0.0 2.4 2.2 0 0.6 -1.5707963</pose>
      </camera>
    </gui>

{models}
  </world>
</sdf>
"""


def quaternion_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    """Return ``(x, y, z, w)`` for a rotation of ``yaw`` about z."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Return the rotation about z of a quaternion, radians."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
