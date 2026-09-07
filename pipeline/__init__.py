"""Shared plumbing of the three pipeline stages.

The stages are separate programs -- ``segmentation.py``, ``annotation.py`` and
``depth_estimation.py`` -- so that changing the annotation model does not mean
segmenting the dataset again. This package holds what they have in common: the
on-disk contract between them in :mod:`pipeline.artifacts`, and the command-line
options they share in :mod:`pipeline.cli`.
"""
