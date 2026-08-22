"""Sphinx configuration for the PLANTORV API documentation.

The docs are API-only: every page is generated from the numpydoc docstrings already in the
source, so there is nothing to keep in sync by hand. ``README.md`` stays the project's
landing page and is deliberately not pulled in here.

Build with ``make docs``, or ``make docs-serve`` for a live-reloading preview.
"""

import sys
from pathlib import Path

# `pip install -e .` puts the packages on the path, so this is only a safety net for a
# checkout where that has not been run.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

## PROJECT #############################################################################################################

project = "PLANTORV"
author = "Enrico Saccon"
release = "0.1.0"

## EXTENSIONS ##########################################################################################################

extensions = [
    "sphinx.ext.autodoc",
    # Renders the numpydoc sections the codebase already uses, so no docstring has to be
    # rewritten into reStructuredText.
    "sphinx.ext.napoleon",
    # Turns type names in signatures into links to their own documentation.
    "sphinx.ext.intersphinx",
    # Adds a "[source]" link next to every documented object.
    "sphinx.ext.viewcode",
    # Lets pages be written in Markdown rather than reST.
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

## AUTODOC #############################################################################################################

# autodoc imports every module it documents. These are heavy, GPU-adjacent, or need
# credentials, and nothing in a signature depends on them, so they are faked at build time.
# This is what lets the docs build on a machine with no GPU and no model checkpoints.
#
# numpy is deliberately *not* mocked: several defaults are `np.asarray([])`, which would
# render as a Mock repr instead of the value.
autodoc_mock_imports = [
    "torch",
    "torchvision",
    "ultralytics",
    "vllm",
    "cv2",
    "skimage",
    "openai",
    "anthropic",
    "google",
    "transformers",
    "tiktoken",
    "matplotlib",
    "pandas",
    "loguru",
]

# Render a default as it is written in the source rather than as its value. Without this,
# `prompt: str = SCENE_INVENTORY_PROMPT` inlines the whole prompt and `SAM3Model.__init__`
# comes out as a 3000-character signature.
autodoc_preserve_defaults = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

# Keep the class docstring on the class and the constructor docstring on `__init__`, which
# is how they are written here.
autoclass_content = "both"

napoleon_numpy_docstring = True
# Off deliberately. Every docstring here is numpydoc, which ruff enforces via
# `convention = "numpy"`, so leaving the Google parser on would only let a stray
# `Args:` block render without anyone noticing the codebase had drifted.
napoleon_google_docstring = False

# Render an "Attributes" section as :ivar: fields on the class rather than as separate
# attribute directives, which autodoc would then document a second time as members.
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pillow": ("https://pillow.readthedocs.io/en/stable", None),
}

## HTML ################################################################################################################

html_theme = "furo"
html_title = f"{project} {release}"
html_static_path = ["_static"]
