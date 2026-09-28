#!/usr/bin/env python3
"""HTTP front door to the one-shot RGB-D capture and the SAM-GPT pipeline.

PLANTOR's web GUI runs in a container that has neither ROS 2 nor a GPU, and the
camera and the models live on this machine. This service is the seam between the
two: it takes the picture, shows it, and -- once somebody has looked at it --
runs segmentation, annotation and depth over it.

Run it in the environment the pipeline runs in -- ROS 2 sourced, the workspace
overlaid, the plantorv virtualenv active::

    python3 -m plantorv_ros.scene_service --port 8010

It starts no camera by default. A capture attaches to whatever is already
publishing ``rgb_topic``, because this machine is expected to be running an
experiment and a second driver would only fight the first over the device. Pass
``"start_camera": true`` in the request body for the case where nothing is up.

Endpoints
---------
``GET /health``
    What the service is configured with, and whether ``ros2`` is on PATH.

``POST /preview``
    Capture one registered RGB-D pair and stop there. Seconds, not minutes: no
    model is loaded. Answers with the two frames and the ``run_id`` they were
    written under.

``POST /process``
    Run segmentation, annotation and depth over the frames of a ``run_id`` that
    ``/preview`` already captured. No camera is touched. Answers with the scene.

``POST /capture``
    Both in one request, for a caller with nobody to show the picture to.

``POST /discard``
    Delete a run directory, for a picture that was rejected.

Every field of a request body is optional apart from ``run_id`` in ``/process``
and ``/discard``::

    {
        "run_id": "kitchen-3",
        "segmenter_config": "segmentation/conf/sam3.yaml",
        "llm_config": "LLM/conf/azure_gpt54.yaml",
        "depth_association": "mask-median",
        "start_camera": false,
        "timeout": 900,
        "extra_arguments": ["--view-masks"],
    }
"""

import argparse
import base64
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# What a capture runs with when the request does not say otherwise. The
# segmenter file carries the concept VLM (azure_gpt52) with it, so only the
# annotator model is named here.
DEFAULT_SEGMENTER_CONFIG = "segmentation/conf/sam3.yaml"
DEFAULT_LLM_CONFIG = "LLM/conf/azure_gpt54.yaml"
DEFAULT_DEPTH_ASSOCIATION = "mask-median"

# Taking the picture loads no model and is over in seconds once the camera is
# up; the pipeline is SAM 3 plus two round trips to Azure per object.
DEFAULT_PREVIEW_TIMEOUT_S = 120.0
DEFAULT_TIMEOUT_S = 900.0

# The name `one_shot_pipeline.py` gives its node, which is the key a parameter
# file has to be written under for the parameters to reach it.
NODE_NAME = "one_shot_pipeline"

# Where the node leaves the frames inside a run directory, relative to it.
RGB_FRAME = Path("input") / "rgb" / "img_0.png"
DEPTH_FRAME = Path("input") / "depth" / "img_0.png"
DEPTH_PREVIEW_FRAME = Path("input") / "depth_preview" / "img_0.png"

# How much of a command's output to hand back when it fails. The whole log is on
# disk in the run directory; this is what the GUI shows without it.
LOG_TAIL_LINES = 60

# Node parameters a request may set, and what they have to be. Everything about
# which camera is used lives here: a driver launched into a namespace publishes
# elsewhere, and a request that cannot say so can only wait for topics nobody
# writes to. Anything not in this table is not forwarded.
CAMERA_PARAMETERS: Dict[str, Callable[[Any], Any]] = {
    "rgb_topic": str,
    "depth_topic": str,
    "pointcloud_topic": str,
    "camera_launch_package": str,
    "camera_launch_file": str,
    "camera_timeout": float,
    "camera_discovery_time": float,
    "sync_slop": float,
    "pointcloud_timeout": float,
}

# The same, for the two that are lists rather than scalars. `[0, 0]` accepts a
# frame of any size, and an empty list of launch arguments has to be written as
# [""], which ROS can type.
CAMERA_LIST_PARAMETERS: Dict[str, Callable[[Any], Any]] = {
    "rgb_size": int,
    "depth_size": int,
}


def find_plantorv_root(explicit: Optional[str] = None) -> Path:
    """Return the plantorv checkout the pipeline is run from.

    Parameters
    ----------
    explicit : str or None
        ``--plantorv-root``, when it was given.

    Returns
    -------
    Path
        Directory holding ``samgpt.py``.

    Raises
    ------
    FileNotFoundError
        If no candidate holds ``samgpt.py``. The service is useless without it,
        so this is raised at start-up rather than on the first capture.
    """
    candidates: List[Path] = []

    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("PLANTORV_ROOT"):
        candidates.append(Path(os.environ["PLANTORV_ROOT"]))

    # Walking up from this file covers a source tree. It does not cover an
    # installed overlay, where this file sits under install/, which is why the
    # two explicit routes above come first.
    candidates.extend(Path(__file__).resolve().parents)

    for candidate in candidates:
        if (candidate / "samgpt.py").is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find the plantorv checkout: none of the candidates holds "
        "samgpt.py. Pass --plantorv-root, or set PLANTORV_ROOT."
    )


class CaptureError(RuntimeError):
    """A request that did not produce what it was asked for.

    Attributes
    ----------
    status : int
        The HTTP status to answer with.
    detail : dict
        What the GUI is told beyond the message: the command, the exit status
        and the tail of the log.
    """

    def __init__(self, message: str, detail: Optional[Dict[str, Any]] = None, status: int = 500):
        super().__init__(message)
        self.detail = detail or {}
        self.status = status


class SceneCapturer:
    """Takes the picture, and runs the pipeline over it when asked to.

    The two are separate because they are decided separately: a picture is worth
    looking at before minutes of segmentation are spent on it, and a rejected
    one costs only the seconds the camera took. :meth:`preview` leaves a run
    directory behind that :meth:`process` picks up, and :meth:`capture` does
    both for a caller with nobody to show the picture to.

    One camera and one GPU, so the work is serialized rather than queued: a
    second request while one is running is refused, which the GUI can say
    plainly, instead of waiting an unbounded time behind it.

    Attributes
    ----------
    root : Path
        The plantorv checkout.
    runs_dir : Path
        Where run directories are created.
    """

    def __init__(
        self,
        root: Path,
        runs_dir: Path,
        *,
        ros_distro_setup: Optional[str] = None,
        workspace_setup: Optional[str] = None,
        python_executable: Optional[str] = None,
    ):
        self.root = root
        self.runs_dir = runs_dir
        self.ros_distro_setup = ros_distro_setup
        self.workspace_setup = workspace_setup
        self.python_executable = python_executable or sys.executable

        self._lock = threading.Lock()
        self.busy_since: Optional[float] = None
        self.busy_with: Optional[str] = None

        self.runs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Run directories
    # ------------------------------------------------------------------

    def new_run_id(self) -> str:
        """Return a name for a run nobody named.

        Returns
        -------
        str
            A timestamp with a short random tail, so two captures in the same
            second do not land in the same directory.
        """
        return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    def run_directory(self, run_id: str) -> Path:
        """Return the directory of a run, refusing one that escapes ``runs_dir``.

        Parameters
        ----------
        run_id : str
            The run's name, which reaches this from the GUI.

        Returns
        -------
        Path
            The run directory, which need not exist yet.

        Raises
        ------
        CaptureError
            If the name holds nothing usable once everything that could climb
            out of ``runs_dir`` is removed from it.
        """
        safe = "".join(
            character for character in run_id if character.isalnum() or character in "-_"
        )

        if not safe:
            raise CaptureError("'run_id' holds no usable character", status=400)

        return self.runs_dir / safe

    def existing_run(self, options: Dict[str, Any]) -> Tuple[str, Path]:
        """Return the run a request names, which must already have frames.

        Parameters
        ----------
        options : dict
            The request body, holding ``run_id``.

        Returns
        -------
        tuple of (str, Path)
            The run's name and its directory.

        Raises
        ------
        CaptureError
            If no ``run_id`` was given, the directory is gone, or it holds no
            RGB frame -- which is what a run looks like after it was discarded.
        """
        run_id = str(options.get("run_id") or "").strip()
        if not run_id:
            raise CaptureError("'run_id' is required: capture a picture first", status=400)

        directory = self.run_directory(run_id)

        if not (directory / RGB_FRAME).is_file():
            raise CaptureError(
                f"No captured picture for run {run_id!r}. It was discarded, or never taken.",
                status=404,
            )

        return directory.name, directory

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def resolve_under_root(self, value: str) -> Path:
        """Return a configuration path, relative ones taken from the checkout.

        Parameters
        ----------
        value : str
            A path from the request or from the defaults.

        Returns
        -------
        Path
            The absolute path.
        """
        path = Path(value)

        return path if path.is_absolute() else (self.root / path)

    def pipeline_arguments(self, options: Dict[str, Any]) -> List[str]:
        """Build the flags the pipeline is run with.

        Parameters
        ----------
        options : dict
            The request body.

        Returns
        -------
        list of str
            The flags, with the configuration files resolved to absolute paths.

        Raises
        ------
        CaptureError
            If a configuration file named by the request does not exist. Failing
            here costs nothing; failing after the frames are captured wastes the
            capture.
        """
        segmenter = self.resolve_under_root(
            str(options.get("segmenter_config") or DEFAULT_SEGMENTER_CONFIG)
        )
        llm_config = self.resolve_under_root(str(options.get("llm_config") or DEFAULT_LLM_CONFIG))
        association = str(options.get("depth_association") or DEFAULT_DEPTH_ASSOCIATION)

        for path, what in (
            (segmenter, "Segmentation configuration"),
            (llm_config, "LLM configuration"),
        ):
            if not path.is_file():
                raise CaptureError(f"{what} does not exist: {path}", status=400)

        arguments = [
            "--segmenter-config",
            str(segmenter),
            "--annotator",
            "gpt",
            "--llm-config",
            str(llm_config),
            "--depth-source",
            "sensor",
            "--depth-association",
            association,
        ]

        extra = options.get("extra_arguments") or []
        if not isinstance(extra, list):
            raise CaptureError("'extra_arguments' must be a list of strings", status=400)
        arguments.extend(str(value) for value in extra)

        return arguments

    def camera_parameters(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Return the node parameters a request asked to override.

        Parameters
        ----------
        options : dict
            The request body.

        Returns
        -------
        dict
            Only the keys of :data:`CAMERA_PARAMETERS`,
            :data:`CAMERA_LIST_PARAMETERS` and ``camera_launch_arguments`` that
            the request carried, converted to the type the node declares them
            with.

        Raises
        ------
        CaptureError
            If a value cannot be converted. Refusing here is better than a node
            that starts and then reads a parameter of the wrong type.
        """
        overrides: Dict[str, Any] = {}

        for name, convert in CAMERA_PARAMETERS.items():
            if options.get(name) is None:
                continue
            try:
                overrides[name] = convert(options[name])
            except (TypeError, ValueError) as error:
                raise CaptureError(f"'{name}' is not a {convert.__name__}", status=400) from error

        for name, convert in CAMERA_LIST_PARAMETERS.items():
            if options.get(name) is None:
                continue
            value = options[name]
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise CaptureError(f"'{name}' must be a list of two numbers", status=400)
            try:
                overrides[name] = [convert(entry) for entry in value]
            except (TypeError, ValueError) as error:
                raise CaptureError(f"'{name}' must be a list of two numbers", status=400) from error

        launch_arguments = options.get("camera_launch_arguments")
        if launch_arguments is not None:
            if not isinstance(launch_arguments, (list, tuple)):
                raise CaptureError(
                    "'camera_launch_arguments' must be a list of strings", status=400
                )
            # Never empty: an empty array has no type ROS can infer.
            overrides["camera_launch_arguments"] = [str(entry) for entry in launch_arguments] or [
                ""
            ]

        return overrides

    def write_parameter_file(
        self,
        directory: Path,
        output_dir: Path,
        *,
        start_camera: bool,
        run_pipeline: bool,
        arguments: List[str],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Write the node's parameters, and return the file.

        A parameter file rather than ``-p name:=value`` on the command line:
        ``pipeline_arguments`` is a string array, and quoting one through a
        shell-less argv and then through the ROS command-line parser is the kind
        of thing that works until a path has a space in it.

        Parameters
        ----------
        directory : Path
            Where to write the file.
        output_dir : Path
            The run directory the node writes to.
        start_camera : bool
            Whether the node may bring a camera driver up.
        run_pipeline : bool
            Whether the node runs the pipeline itself once the frames are saved.
        arguments : list of str
            The pipeline flags, which the node ignores when it runs no pipeline.
        overrides : dict or None
            Camera and topic parameters from the request, which win over the
            node's own defaults.

        Returns
        -------
        Path
            The parameter file.
        """
        parameters = {
            NODE_NAME: {
                "ros__parameters": {
                    **(overrides or {}),
                    "start_camera": bool(start_camera),
                    "run_pipeline": bool(run_pipeline),
                    "pipeline_script": str(self.root / "samgpt.py"),
                    "output_dir": str(output_dir),
                    # Never an empty array: ROS cannot infer the type of one, so
                    # the parameter would arrive uninitialized rather than empty
                    # and reading it raises. [""] is what "no arguments" looks
                    # like, and the node drops the empty string.
                    "pipeline_arguments": arguments or [""],
                }
            }
        }

        path = directory / "one_shot_pipeline.params.yaml"

        # JSON is valid YAML, and it quotes and escapes every string for us.
        path.write_text(json.dumps(parameters, indent=2), encoding="utf-8")

        return path

    # ------------------------------------------------------------------
    # Running things
    # ------------------------------------------------------------------

    def node_command(self, parameter_file: Path) -> List[str]:
        """Return the argv that runs the one-shot node.

        Parameters
        ----------
        parameter_file : Path
            The file written by :meth:`write_parameter_file`.

        Returns
        -------
        list of str
            The command. When a setup script was configured it is a
            ``bash -c`` that sources it first, since sourcing cannot be done
            from inside an already-running process.
        """
        node = [
            "ros2",
            "run",
            "plantorv_ros",
            NODE_NAME,
            "--ros-args",
            "--params-file",
            str(parameter_file),
        ]

        setups = [path for path in (self.ros_distro_setup, self.workspace_setup) if path]
        if not setups:
            return node

        sourced = " && ".join(f". {shlex.quote(path)}" for path in setups)

        return ["bash", "-c", f"{sourced} && exec {shlex.join(node)}"]

    def pipeline_command(self, output_dir: Path, arguments: List[str]) -> List[str]:
        """Return the argv that runs the pipeline over an existing run.

        The frames are already on disk, so this needs neither ROS nor the
        camera: it is the same ``samgpt.py`` call the node would have made.

        Parameters
        ----------
        output_dir : Path
            The run directory.
        arguments : list of str
            The pipeline flags.

        Returns
        -------
        list of str
            The command.
        """
        return [
            self.python_executable,
            str(self.root / "samgpt.py"),
            "--input-dir",
            str(output_dir / RGB_FRAME.parent),
            "--depth-dir",
            str(output_dir / DEPTH_FRAME.parent),
            "--depth-source",
            "sensor",
            "--output-dir",
            str(output_dir),
            *arguments,
        ]

    def run(
        self,
        command: List[str],
        *,
        output_dir: Path,
        log_name: str,
        timeout: float,
        what: str,
    ) -> Tuple[float, Path]:
        """Run a command, log it into the run directory, and time it.

        Parameters
        ----------
        command : list of str
            The argv.
        output_dir : Path
            The run directory, which the log is written into.
        log_name : str
            File name of the log.
        timeout : float
            Seconds before the command is abandoned.
        what : str
            How the command is named in an error message.

        Returns
        -------
        tuple of (float, Path)
            Seconds it took, and the log file.

        Raises
        ------
        CaptureError
            If the command timed out, could not be started, or exited non-zero.
        """
        log_path = output_dir / log_name
        started = time.time()

        try:
            completed = subprocess.run(
                command,
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise CaptureError(
                f"{what} did not finish within {timeout:.0f}s",
                {"command": command, "log": _tail(error.stdout), "stderr": _tail(error.stderr)},
            ) from error
        except FileNotFoundError as error:
            raise CaptureError(
                f"Could not start {what}: {error.filename} is not on PATH. Start this "
                "service from a shell with ROS 2 and the workspace sourced, or pass "
                "--ros-setup/--workspace-setup.",
                {"command": command},
            ) from error

        log_path.write_text(
            f"$ {shlex.join(command)}\n\n{completed.stdout}\n{completed.stderr}",
            encoding="utf-8",
        )

        if completed.returncode != 0:
            # The status alone says nothing anybody can act on. The node's own
            # last complaint usually does -- "no frames 30 s after start. Is
            # /camera/color/image_raw being published?" is the whole diagnosis
            # -- so it goes in the message, where the GUI already shows it.
            reason = _last_complaint(completed.stdout, completed.stderr)

            raise CaptureError(
                f"{what} exited with status {completed.returncode}"
                + (f": {reason}" if reason else ""),
                {
                    "command": command,
                    "returncode": completed.returncode,
                    "log": _tail(completed.stdout),
                    "stderr": _tail(completed.stderr),
                    "log_file": str(log_path),
                },
            )

        return time.time() - started, log_path

    def exclusively(self, what: str, work: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        """Run one piece of work, refusing to start a second at the same time.

        Parameters
        ----------
        what : str
            What is being done, reported to whoever is refused.
        work : callable
            The work.

        Returns
        -------
        dict
            Whatever ``work`` returned.

        Raises
        ------
        CaptureError
            If something else is already running.
        """
        if not self._lock.acquire(blocking=False):
            raise CaptureError(
                f"The scene service is busy ({self.busy_with}). The camera and the GPU "
                "are taken until it finishes.",
                {"busy_since": self.busy_since, "busy_with": self.busy_with},
                status=409,
            )

        self.busy_since = time.time()
        self.busy_with = what
        try:
            return work()
        finally:
            self.busy_since = None
            self.busy_with = None
            self._lock.release()

    # ------------------------------------------------------------------
    # The three things a caller can ask for
    # ------------------------------------------------------------------

    def preview(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Take one picture and stop.

        Parameters
        ----------
        options : dict
            The request body.

        Returns
        -------
        dict
            The two frames and the run they were written under, for somebody to
            look at before :meth:`process` is asked for.
        """
        return self.exclusively("taking a picture", lambda: self._preview(options))

    def _preview(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Do the work of :meth:`preview`, with the lock already held."""
        run_id = str(options.get("run_id") or "").strip() or self.new_run_id()
        output_dir = self.run_directory(run_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Checked before the camera is touched rather than after: a bad model
        # path should not cost a capture, and /process would be the one to fail.
        self.pipeline_arguments(options)

        timeout = float(options.get("timeout") or DEFAULT_PREVIEW_TIMEOUT_S)

        with tempfile.TemporaryDirectory() as staging:
            parameter_file = self.write_parameter_file(
                Path(staging),
                output_dir,
                start_camera=bool(options.get("start_camera", False)),
                run_pipeline=False,
                arguments=[""],
                overrides=self.camera_parameters(options),
            )

            elapsed, log_path = self.run(
                self.node_command(parameter_file),
                output_dir=output_dir,
                log_name="capture.log",
                timeout=timeout,
                what="The capture",
            )

        if not (output_dir / RGB_FRAME).is_file():
            raise CaptureError(
                f"The capture finished but wrote no RGB frame (no {RGB_FRAME} in {output_dir})",
                {"log_file": str(log_path)},
            )

        return {
            "run_id": output_dir.name,
            "output_dir": str(output_dir),
            "stage": "preview",
            "log_file": str(log_path),
            "elapsed_s": round(elapsed, 1),
            "captured_at": datetime.now(timezone.utc).isoformat(),
            **self.frames(output_dir),
        }

    def process(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Run the pipeline over a picture that was already taken.

        Parameters
        ----------
        options : dict
            The request body, holding the ``run_id`` of a previewed capture.

        Returns
        -------
        dict
            The scene.
        """
        return self.exclusively("segmenting and annotating", lambda: self._process(options))

    def _process(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Do the work of :meth:`process`, with the lock already held."""
        run_id, output_dir = self.existing_run(options)
        arguments = self.pipeline_arguments(options)
        timeout = float(options.get("timeout") or DEFAULT_TIMEOUT_S)

        elapsed, log_path = self.run(
            self.pipeline_command(output_dir, arguments),
            output_dir=output_dir,
            log_name="pipeline.log",
            timeout=timeout,
            what="The pipeline",
        )

        return self.read_scene(
            run_id,
            output_dir,
            arguments=arguments,
            elapsed=elapsed,
            log_file=log_path,
        )

    def capture(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Take one picture and run the pipeline over it, in one request.

        Parameters
        ----------
        options : dict
            The request body.

        Returns
        -------
        dict
            The scene, with the frames the preview would have shown.
        """

        def both() -> Dict[str, Any]:
            preview = self._preview(options)
            scene = self._process({**options, "run_id": preview["run_id"]})

            # The preview's own timing and log would otherwise be lost; the
            # caller asked for one thing and gets one answer.
            scene["capture_log_file"] = preview["log_file"]
            scene["capture_elapsed_s"] = preview["elapsed_s"]
            scene["elapsed_s"] = round(scene["elapsed_s"] + preview["elapsed_s"], 1)

            return scene

        return self.exclusively("capturing and annotating", both)

    def discard(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Delete the run directory of a picture that was rejected.

        Parameters
        ----------
        options : dict
            The request body, holding ``run_id``.

        Returns
        -------
        dict
            What was deleted.
        """
        run_id, output_dir = self.existing_run(options)

        # Only ever a directory this service created under runs_dir:
        # `existing_run` refuses a run_id that points anywhere else.
        shutil.rmtree(output_dir, ignore_errors=True)

        return {"run_id": run_id, "output_dir": str(output_dir), "discarded": True}

    # ------------------------------------------------------------------
    # Reading back
    # ------------------------------------------------------------------

    def frames(self, output_dir: Path) -> Dict[str, Any]:
        """Return the frames of a run, as data URLs and as paths.

        The GUI's browser cannot read this machine's filesystem and the
        container serving it does not share one, so the previews travel inside
        the response rather than as paths.

        Parameters
        ----------
        output_dir : Path
            The run directory.

        Returns
        -------
        dict
            ``rgb_image`` and ``depth_image`` as ``data:`` URLs, and the two
            files they came from.
        """
        rgb = output_dir / RGB_FRAME
        depth = output_dir / DEPTH_FRAME

        return {
            "rgb_image": _data_url(rgb),
            "depth_image": _data_url(output_dir / DEPTH_PREVIEW_FRAME),
            "rgb_file": str(rgb) if rgb.is_file() else None,
            "depth_file": str(depth) if depth.is_file() else None,
        }

    def read_scene(
        self,
        run_id: str,
        output_dir: Path,
        *,
        arguments: List[str],
        elapsed: float,
        log_file: Path,
    ) -> Dict[str, Any]:
        """Collect what a finished run left on disk.

        Parameters
        ----------
        run_id : str
            The run's name.
        output_dir : Path
            The run directory.
        arguments : list of str
            The flags the pipeline ran with, echoed back so the GUI can show
            which models produced the scene.
        elapsed : float
            Seconds the pipeline took.
        log_file : Path
            The pipeline's log.

        Returns
        -------
        dict
            The response body.

        Raises
        ------
        CaptureError
            If the run left no ``output_img<id>.json``.
        """
        results = sorted(output_dir.glob("output_img*.json"))
        if not results:
            raise CaptureError(
                "The pipeline finished but wrote no scene file "
                f"(no output_img*.json in {output_dir})",
                {"log_file": str(log_file)},
            )

        scene_path = results[0]
        objects = json.loads(scene_path.read_text(encoding="utf-8"))

        return {
            "run_id": run_id,
            "output_dir": str(output_dir),
            "stage": "scene",
            "scene_file": str(scene_path),
            "objects": objects,
            "object_count": len(objects),
            "log_file": str(log_file),
            "elapsed_s": round(elapsed, 1),
            "pipeline_arguments": arguments,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            **self.frames(output_dir),
        }

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        """Return what the service is configured with.

        Returns
        -------
        dict
            Enough for the GUI to say why a capture would fail before one is
            attempted.
        """
        # Looked up rather than run: `ros2 --help` imports the whole command
        # extension set and takes tens of seconds cold, which would make a
        # health check that is meant to be instant time out and report a
        # working installation as missing.
        has_ros = shutil.which("ros2") is not None

        return {
            "ok": has_ros and (self.root / "samgpt.py").is_file(),
            "ros2_available": has_ros,
            "plantorv_root": str(self.root),
            "runs_dir": str(self.runs_dir),
            "busy": self.busy_since is not None,
            "busy_since": self.busy_since,
            "busy_with": self.busy_with,
            "defaults": {
                "segmenter_config": DEFAULT_SEGMENTER_CONFIG,
                "llm_config": DEFAULT_LLM_CONFIG,
                "depth_association": DEFAULT_DEPTH_ASSOCIATION,
                "start_camera": False,
            },
        }


def _tail(text: Optional[str]) -> str:
    """Return the last :data:`LOG_TAIL_LINES` lines of a log.

    Parameters
    ----------
    text : str or None
        The captured output.

    Returns
    -------
    str
        The tail, empty when there was no output.
    """
    if not text:
        return ""

    return "\n".join(text.strip().splitlines()[-LOG_TAIL_LINES:])


def _last_complaint(*outputs: Optional[str]) -> str:
    """Return the line of a failed command that best explains it.

    Parameters
    ----------
    *outputs : str or None
        The captured stdout and stderr.

    Returns
    -------
    str
        The last line logged at ERROR or FATAL, falling back to the last
        non-empty line, with the ROS log prefix stripped. Empty when the
        command said nothing.
    """
    lines = [
        line.strip() for output in outputs if output for line in output.splitlines() if line.strip()
    ]

    if not lines:
        return ""

    chosen = next(
        (line for line in reversed(lines) if "[ERROR]" in line or "[FATAL]" in line),
        lines[-1],
    )

    # "[ERROR] [1789459877.293] [one_shot_pipeline]: no frames ..." is three
    # brackets of timestamp and node name before the part worth reading.
    tail = chosen.rsplit("]: ", 1)

    return tail[-1] if len(tail) > 1 else chosen


def _data_url(path: Path) -> Optional[str]:
    """Return a PNG as a ``data:`` URL, or None when it is not there.

    Parameters
    ----------
    path : Path
        The PNG.

    Returns
    -------
    str or None
        The data URL.
    """
    if not path.is_file():
        return None

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")

    return f"data:image/png;base64,{encoded}"


class SceneRequestHandler(BaseHTTPRequestHandler):
    """The endpoints, over the standard library's HTTP server."""

    server_version = "PlantorvSceneService/1.1"

    capturer: SceneCapturer

    def _send(self, status: int, payload: Dict[str, Any]) -> None:
        """Write a JSON response.

        Parameters
        ----------
        status : int
            HTTP status code.
        payload : dict
            The body.
        """
        body = json.dumps(payload).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802 - the name is BaseHTTPRequestHandler's
        """Answer the browser's preflight."""
        self._send(204, {})

    def do_GET(self) -> None:  # noqa: N802 - the name is BaseHTTPRequestHandler's
        """Serve ``/health``."""
        if self.path.rstrip("/") in ("/health", ""):
            self._send(200, self.capturer.health())
            return

        self._send(404, {"error": f"No such endpoint: {self.path}"})

    def do_POST(self) -> None:  # noqa: N802 - the name is BaseHTTPRequestHandler's
        """Serve the capture endpoints."""
        handlers = {
            "/preview": self.capturer.preview,
            "/process": self.capturer.process,
            "/capture": self.capturer.capture,
            "/discard": self.capturer.discard,
        }

        handler = handlers.get(self.path.rstrip("/"))
        if handler is None:
            self._send(404, {"error": f"No such endpoint: {self.path}"})
            return

        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"

        try:
            options = json.loads(raw or b"{}")
        except json.JSONDecodeError as error:
            self._send(400, {"error": f"Body is not JSON: {error}"})
            return

        if not isinstance(options, dict):
            self._send(400, {"error": "Body must be a JSON object"})
            return

        try:
            self._send(200, handler(options))
        except CaptureError as error:
            self._send(error.status, {"error": str(error), **error.detail})
        except Exception as error:  # noqa: BLE001 - the service must not die on one request
            self._send(500, {"error": f"{type(error).__name__}: {error}"})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - the name is the base class's
        """Log to stderr with a readable prefix."""
        sys.stderr.write(f"[scene_service] {self.address_string()} {format % args}\n")


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the service's command line.

    Parameters
    ----------
    argv : list of str or None
        Arguments, defaulting to ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Address to listen on. 0.0.0.0 is what a container reaches (default: %(default)s)",
    )
    parser.add_argument(
        "--port", type=int, default=8010, help="Port to listen on (default: %(default)s)"
    )
    parser.add_argument(
        "--plantorv-root",
        help="The plantorv checkout holding samgpt.py. Found from this file when unset",
    )
    parser.add_argument(
        "--runs-dir",
        help="Where run directories are created (default: <plantorv-root>/output/gui_captures)",
    )
    parser.add_argument(
        "--python",
        dest="python_executable",
        help=(
            "Interpreter the pipeline is run with. Defaults to the one running this "
            "service, which is the one its dependencies are installed for"
        ),
    )
    parser.add_argument(
        "--ros-setup",
        help=(
            "ROS 2 setup.bash to source before running the node. Only needed when this "
            "service is started from a shell that has not sourced it"
        ),
    )
    parser.add_argument(
        "--workspace-setup",
        help="The workspace's install/setup.bash, sourced after --ros-setup",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Start the service.

    Parameters
    ----------
    argv : list of str or None
        Arguments, defaulting to ``sys.argv``.
    """
    args = parse_arguments(argv)

    try:
        root = find_plantorv_root(args.plantorv_root)
    except FileNotFoundError as error:
        sys.stderr.write(f"{error}\n")
        raise SystemExit(1) from error

    runs_dir = Path(args.runs_dir) if args.runs_dir else root / "output" / "gui_captures"

    handler = type(
        "BoundSceneRequestHandler",
        (SceneRequestHandler,),
        {
            "capturer": SceneCapturer(
                root,
                runs_dir,
                ros_distro_setup=args.ros_setup,
                workspace_setup=args.workspace_setup,
                python_executable=args.python_executable,
            )
        },
    )

    server = ThreadingHTTPServer((args.host, args.port), handler)

    sys.stderr.write(
        "[scene_service] listening\n"
        f"  address:       http://{args.host}:{args.port}\n"
        f"  plantorv root: {root}\n"
        f"  runs dir:      {runs_dir}\n"
        "  the camera is never started or stopped unless a request asks for it\n"
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
