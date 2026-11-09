"""Record what a stage was actually configured with, next to what it produced.

A run is only worth comparing against another if what separates the two is known.
Each stage therefore writes a YAML file beside its output, holding the values it
really used -- the defaults its models filled in included, not merely the ones the
configuration file happened to set.

The file is a record, never an input: nothing reads it back. It is written where
the stage writes its output, so it travels with the artefacts it describes.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

import yaml

from utility.json_serialization import to_json_compatible as convert
from utility.utility import logger


def plain(value: Any) -> Any:
    """Turn a value into something ``yaml.safe_dump`` accepts.

    Parameters
    ----------
    value : Any
        Any value, possibly holding ``Path`` objects or NumPy scalars.

    Returns
    -------
    Any
        The same value, with the types the project already knows how to serialise
        converted to their plain equivalents.
    """
    return json.loads(json.dumps(value, default=convert))


def write_run_config(
    output_dir: Union[str, Path],
    stage: str,
    config: Dict[str, Any],
    *,
    device: str = "",
) -> Path:
    """Write the effective configuration of one stage.

    Parameters
    ----------
    output_dir : Union[str, Path]
        Directory the stage writes its output to.
    stage : str
        The stage name, used for the file name and recorded in it.
    config : Dict[str, Any]
        The values the stage ran with.
    device : str
        The device it ran on, when that applies.

    Returns
    -------
    Path
        ``<output_dir>/<stage>_config.yaml``, the file that was written.
    """
    recorded: Dict[str, Any] = {
        "STAGE": stage,
        "RUN_AT": datetime.now().isoformat(timespec="seconds"),
    }
    if device:
        recorded["DEVICE"] = device
    recorded.update(config)

    path = Path(output_dir) / f"{stage}_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        yaml.safe_dump(plain(recorded), handle, sort_keys=False, default_flow_style=False)

    logger.info(f"Wrote the {stage} configuration to {path}")

    return path
