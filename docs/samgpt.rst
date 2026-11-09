samgpt module
=============

Runs the three stage scripts -- ``segmentation.py``, ``annotation.py`` and
``depth_estimation.py`` -- in a row. Hand-written rather than generated: ``sphinx-apidoc``
only walks packages, and this is a top-level module, so it lives here instead of in the
generated ``docs/api/``. The stage scripts themselves are not documented here: one of them
shares a name with the ``segmentation`` package, which an ``automodule`` directive would
resolve to instead. What they exchange is in :doc:`api/pipeline`.

.. automodule:: samgpt
   :members:
   :undoc-members:
   :show-inheritance:
