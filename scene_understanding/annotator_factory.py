"""Build an annotator from its name, and describe it for the run record.

The single place that knows which annotators exist. Adding one means adding its
module to this package, a branch to :func:`create_annotator` and its name to
:data:`ANNOTATORS`; the pipeline and its ``--annotator`` flag pick it up from here.
"""

from typing import Any, Dict

from scene_understanding.annotator import Annotator

# `gpt` sends the crops to the remote model of an LLM configuration file; `dam` runs
# Describe Anything locally over the masks.
ANNOTATORS = ("gpt", "dam")


def create_annotator(name: str, **params: Any) -> Annotator:
    """Build the annotator named by ``name``.

    Parameters
    ----------
    name : str
        One of :data:`ANNOTATORS`.
    **params : Any
        Forwarded to the backend: ``GPTAnnotator.from_config`` for ``gpt`` (e.g.
        ``llm_config_file``), ``DAMAnnotator`` for ``dam`` (e.g. ``model_path``,
        ``query``, ``device``).

    Returns
    -------
    Annotator
        The configured annotator.

    Raises
    ------
    ValueError
        If ``name`` is not one of :data:`ANNOTATORS`, or the LLM configuration does
        not describe a usable backend.
    FileNotFoundError
        If the checkpoint or the configuration file is missing.
    ImportError
        If DAM is selected but the ``dam`` package is not installed.
    """
    # Imported per branch rather than at module scope so that a run does not pay for a
    # dependency tree it will not use.
    if name == "dam":
        from scene_understanding.dam_annotator import DAMAnnotator

        return DAMAnnotator(**params)

    if name == "gpt":
        from scene_understanding.gpt_annotator import GPTAnnotator

        # The YAML file selects the model, the endpoint, the credentials and the
        # request parameters, so switching model means pointing at another file.
        return GPTAnnotator.from_config(**params)

    raise ValueError(f"Unknown annotator {name!r}; expected one of {ANNOTATORS}")


def describe(annotator: Annotator, name: str) -> Dict[str, Any]:
    """Describe a built annotator, for the record written next to a run.

    Reads the annotator rather than the arguments, so a value its constructor filled
    in is part of the answer.

    Parameters
    ----------
    annotator : Annotator
        The annotator, as returned by :func:`create_annotator`.
    name : str
        The name it was created with.

    Returns
    -------
    Dict[str, Any]
        The effective configuration.
    """
    described: Dict[str, Any] = {"ANNOTATOR": name}

    llm = getattr(annotator, "llm", None)
    if llm is not None:
        described["LLM_CONFIG_FILE"] = llm.config_file
        described["LLM_VERSION"] = llm.model
        described["LLM_CONFIG"] = llm.request_params()
        # The prompt decides the vocabulary of the tags as much as the model does.
        described["PROMPT"] = getattr(annotator, "prompt", None)
        return described

    described.update(
        {
            "DAM_MODEL": getattr(annotator, "model_path", None),
            "DAM_CONFIG": {
                "temperature": getattr(annotator, "temperature", None),
                "top_p": getattr(annotator, "top_p", None),
                "num_beams": getattr(annotator, "num_beams", None),
                "max_new_tokens": getattr(annotator, "max_new_tokens", None),
            },
            "QUERY": getattr(annotator, "query", None),
        }
    )

    return described
