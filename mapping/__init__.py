"""Mapping utilities.

The point-cloud helpers need Open3D, which the main environment does not install: its
precompiled PyTorch ops pin torch to 2.2.*, and the rest of the project has moved past
that (see requirements-o3dml.txt). They are therefore resolved on first use rather than
at import, so that importing anything from this package -- the depth providers, the
RGB-D mapper -- costs nothing and works without Open3D.

Reaching one of them in an environment that has no Open3D raises ImportError naming the
environment to use, rather than failing at the import of an unrelated module.
"""

import importlib
from typing import TYPE_CHECKING, Any

from mapping.depth_anything import DepthAnythingV2Provider, DepthAnythingV3Provider
from mapping.depth_provider import DepthProvider, DepthResult, SensorDepthProvider
from mapping.rgbd_mapper import attach_object_depths

if TYPE_CHECKING:
    # Imported for the type checkers only: at runtime `__getattr__` below resolves these,
    # so that Open3D is never imported by a program that does not ask for them.
    from mapping.geometric_clustering import (
        GeometricSegmentation,
        compare_label_images,
        segment_planes_and_clusters,
    )
    from mapping.rgbd_pointcloud import (
        RGBDPointCloud,
        RGBDPointCloudGenerator,
        create_aligned_point_cloud,
        create_point_cloud_from_aligned_depth,
    )

# Name -> the module it lives in, for the exports that need Open3D.
_OPEN3D_EXPORTS = {
    "GeometricSegmentation": "mapping.geometric_clustering",
    "compare_label_images": "mapping.geometric_clustering",
    "segment_planes_and_clusters": "mapping.geometric_clustering",
    "RGBDPointCloud": "mapping.rgbd_pointcloud",
    "RGBDPointCloudGenerator": "mapping.rgbd_pointcloud",
    "create_aligned_point_cloud": "mapping.rgbd_pointcloud",
    "create_point_cloud_from_aligned_depth": "mapping.rgbd_pointcloud",
}

__all__ = [
    "DepthAnythingV2Provider",
    "DepthAnythingV3Provider",
    "DepthProvider",
    "DepthResult",
    "GeometricSegmentation",
    "RGBDPointCloud",
    "RGBDPointCloudGenerator",
    "SensorDepthProvider",
    "attach_object_depths",
    "compare_label_images",
    "create_aligned_point_cloud",
    "create_point_cloud_from_aligned_depth",
    "segment_planes_and_clusters",
]


def __getattr__(name: str) -> Any:
    """Resolve the Open3D-backed exports on first use.

    Parameters
    ----------
    name : str
        Attribute being read from the package.

    Returns
    -------
    Any
        The requested object.

    Raises
    ------
    AttributeError
        If the package does not export ``name``.
    ImportError
        If ``name`` needs Open3D and Open3D is not installed.
    """
    module_name = _OPEN3D_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ImportError(
            f"{name} needs Open3D, which the main environment does not install because "
            "Open3D 0.19 pins torch to 2.2.*. Use the environment of "
            "requirements-o3dml.txt (`make install-o3dml`) for the point-cloud work."
        ) from error

    return getattr(module, name)
