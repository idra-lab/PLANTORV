samgpt module
=============

A thin wrapper around :mod:`pipeline.run`, kept at the repository root because the
ROS nodes and the cluster jobs call it by path. It runs the three stages --
:mod:`pipeline.segment`, :mod:`pipeline.annotate` and :mod:`pipeline.estimate_depth`
-- in a row; they and what they exchange are documented in :doc:`api/pipeline`.

.. automodule:: samgpt
   :members:
   :undoc-members:
   :show-inheritance:
