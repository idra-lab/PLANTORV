"""Interchangeable sources of RGB-aligned metric depth.

Every backend implements :class:`depth.depth_provider.DepthProvider`;
:func:`depth.depth_factory.create_depth_provider` builds one by name. The camera
geometry they share (calibration, depth-to-RGB alignment, back-projection) lives in
``mapping``.
"""
