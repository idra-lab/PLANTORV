import json
from pathlib import Path

import numpy as np

from utility.json_serialization import to_json_compatible


def test_pipeline_json_converter_handles_paths_and_numpy_values() -> None:
    result = json.dumps(
        {
            "mask": Path("ppt_outputs/image1/mask_0.png"),
            "bbox": np.asarray([1, 2, 3, 4]),
            "depth": np.float32(1250.5),
        },
        default=to_json_compatible,
    )

    assert json.loads(result) == {
        "mask": "ppt_outputs/image1/mask_0.png",
        "bbox": [1, 2, 3, 4],
        "depth": 1250.5,
    }


def test_pipeline_json_converter_rejects_unknown_types_clearly() -> None:
    try:
        json.dumps({"value": object()}, default=to_json_compatible)
    except TypeError as exc:
        assert "not JSON serializable" in str(exc)
    else:
        raise AssertionError("Unknown object unexpectedly serialized")
