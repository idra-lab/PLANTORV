"""Build a depth backend from its name.

The single place that knows which depth sources exist. Adding a backend means
adding its module to this package, a branch to :func:`create_depth_provider` and its
name to :data:`DEPTH_SOURCES`; the pipeline and its ``--depth-source`` flag pick it up
from here.
"""

from __future__ import annotations

from typing import Optional

from depth.depth_anything import (
    DEFAULT_FOCAL_LENGTH_PX,
    DEFAULT_MODEL_ID,
    DEFAULT_V3_MODEL_ID,
    DepthAnythingV2Provider,
    DepthAnythingV3Provider,
)
from depth.depth_provider import DepthProvider

# `sensor` reads the depth images recorded with the frames; the others infer depth
# from the RGB frame alone.
DEPTH_SOURCES = ("sensor", "depth-anything-v2", "monocular")


def create_depth_provider(
    source: str,
    *,
    model_id: Optional[str] = None,
    device: Optional[str] = None,
    focal_length_px: float = DEFAULT_FOCAL_LENGTH_PX,
    process_res: int = 504,
) -> Optional[DepthProvider]:
    """Build the depth backend named by ``source``.

    Parameters
    ----------
    source : str
        One of :data:`DEPTH_SOURCES`.
    model_id : str or None
        Checkpoint of an estimator; each one has its own default.
    device : str or None
        Device of an estimator; it chooses on its own when None.
    focal_length_px : float
        Mean RGB focal length, used by Depth Anything 3 for metric scaling.
    process_res : int
        Depth Anything 3 processing resolution.

    Returns
    -------
    DepthProvider or None
        The estimator, or None for ``sensor``: the sensor provider needs each frame's
        depth image, so the caller builds it per frame with
        :meth:`SensorDepthProvider.from_frames` (or :func:`main_coords`).

    Raises
    ------
    ValueError
        If ``source`` is not one of :data:`DEPTH_SOURCES`.
    """
    if source == "sensor":
        return None
    if source == "depth-anything-v2":
        return DepthAnythingV2Provider(model_id=model_id or DEFAULT_MODEL_ID, device=device)
    if source == "monocular":
        return DepthAnythingV3Provider(
            model_id=model_id or DEFAULT_V3_MODEL_ID,
            focal_length_px=focal_length_px,
            process_res=process_res,
            device=device,
        )
    raise ValueError(f"Unknown depth source {source!r}; expected one of {DEPTH_SOURCES}")
