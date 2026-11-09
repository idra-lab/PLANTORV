import argparse
import json
import os
import sys
from pathlib import Path

import cv2

from utility.utility import logger

DEFAULT_OUTPUT_DICTIONARY = "aruco_markers.json"


def load_output_dictionary(path: Path) -> dict:
    """
    Load the output dictionary from a JSON file.

    Parameters
    ----------
    path : Path
        Path to the JSON file.

    Returns
    -------
    dict
        The loaded output dictionary.
    """
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        output_dictionary = json.load(f)

    if not isinstance(output_dictionary, dict):
        raise ValueError(f"{path} must contain a JSON object mapping file paths to marker IDs.")

    return output_dictionary


def save_output_dictionary(path: Path, output_dictionary: dict) -> None:
    """
    Save the output dictionary to a JSON file.

    Parameters
    ----------
    path : Path
        Path to the JSON file.
    output_dictionary : dict
        The output dictionary to save.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(output_dictionary, f, indent=2, sort_keys=True)
        f.write("\n")


def relative_to_output_dictionary(file_path: Path, output_dictionary_path: Path) -> Path:
    """
    Get the relative path of a file with respect to the output dictionary's parent directory.

    Parameters
    ----------
    file_path : Path
        The absolute path of the file.
    output_dictionary_path : Path
        The path to the output dictionary JSON file.

    Returns
    -------
    Path
        The relative path of the file with respect to the output dictionary's parent directory.
    """
    relative_path = os.path.relpath(file_path.resolve(), output_dictionary_path.parent.resolve())
    # return Path(relative_path).as_posix()
    return Path(
        relative_path
    )  # TODO Check if this works. Probably returning Path breaks update_output_dictionary when saving the dictionary


def update_output_dictionary(
    output_dictionary_path: Path, generated_file: Path, marker_id: int, description: str
) -> None:
    output_dictionary = load_output_dictionary(output_dictionary_path)
    relative_file = relative_to_output_dictionary(generated_file, output_dictionary_path)
    output_dictionary[relative_file] = {
        "id": marker_id,
        "description": description,
    }
    save_output_dictionary(output_dictionary_path, output_dictionary)


def validate_output_dictionary(output_dictionary_path: Path) -> tuple[list, list]:
    """
    Validate the output dictionary by checking for missing files and invalid entries.

    Parameters
    ----------
    output_dictionary_path : Path
        Path to the output dictionary JSON file.

    Returns
    -------
    tuple[list, list]
        A tuple containing two lists:
        - The first list contains invalid entries in the output dictionary.
        - The second list contains missing files that are referenced in the output dictionary but do not exist on disk.
    """
    output_dictionary = load_output_dictionary(output_dictionary_path)
    missing_files = []
    invalid_entries = []

    for relative_file, marker_data in output_dictionary.items():
        if (
            not isinstance(relative_file, str)
            or not relative_file
            or Path(relative_file).is_absolute()
        ):
            invalid_entries.append(relative_file)
            continue

        if type(marker_data) is int:
            pass
        elif (
            not isinstance(marker_data, dict)
            or type(marker_data.get("id")) is not int
            or not isinstance(marker_data.get("description"), str)
        ):
            invalid_entries.append(relative_file)
            continue

        file_path = output_dictionary_path.parent / relative_file
        if not file_path.is_file():
            missing_files.append(relative_file)

    return invalid_entries, missing_files


def generate(
    marker_id: int,
    marker_size_px: int = 800,
    border_bits: int = 1,
    dictionary_name: str = "DICT_6X6_250",
) -> Path:
    """
    Generate an ArUco marker image and save it to a file.

    Parameters
    ----------
    marker_id : int
        The ID of the ArUco marker to generate.
    marker_size_px : int, optional
        The size of the generated marker image in pixels (default is 800).
    border_bits : int, optional
        The number of bits in the marker border (default is 1).
    dictionary_name : str, optional
        The name of the predefined ArUco dictionary to use (default is "DICT_6X6_250").

    Returns
    -------
    Path
        The path to the generated marker image file.
    """
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dictionary_name))

    img = aruco.generateImageMarker(dictionary, marker_id, marker_size_px, borderBits=border_bits)

    output_file = Path(f"aruco_{marker_id}.png")
    if not cv2.imwrite(str(output_file), img):
        raise OSError(f"Failed to write marker image to {output_file}")

    return output_file


if __name__ == "__main__":
    argsparser = argparse.ArgumentParser(description="Generate ArUco marker images.")
    argsparser.add_argument(
        "marker_id", type=int, nargs="?", help="ID of the ArUco marker to generate."
    )
    argsparser.add_argument(
        "--size",
        type=int,
        default=800,
        help="Size of the generated marker image in pixels (default: 800).",
    )
    argsparser.add_argument(
        "--border", type=int, default=1, help="Number of bits in the marker border (default: 1)."
    )
    argsparser.add_argument(
        "--dictionary",
        type=str,
        default="DICT_6X6_250",
        help="Predefined ArUco dictionary to use (default: DICT_6X6_250).",
    )
    argsparser.add_argument(
        "--output-dictionary",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DICTIONARY),
        help=f"JSON file mapping generated marker paths to marker metadata (default: {DEFAULT_OUTPUT_DICTIONARY}).",
    )
    argsparser.add_argument(
        "--description",
        type=str,
        default="",
        help="Description to store in the output dictionary for the generated marker.",
    )
    argsparser.add_argument(
        "--validate",
        action="store_true",
        help="Validate that all files listed in the output dictionary still exist.",
    )
    args = argsparser.parse_args()

    if args.validate:
        if not args.output_dictionary.exists():
            logger.error(f"{args.output_dictionary} does not exist.")
            sys.exit(1)

        invalid_entries, missing_files = validate_output_dictionary(args.output_dictionary)
        if invalid_entries or missing_files:
            logger.error(f"{args.output_dictionary} is invalid.")
            for entry in invalid_entries:
                logger.error(f"Invalid entry: {entry}")
            for missing_file in missing_files:
                logger.error(f"Missing file: {missing_file}")
            sys.exit(1)

        logger.info(f"{args.output_dictionary} is valid.")
        sys.exit(0)

    if args.marker_id is None:
        argsparser.error("marker_id is required unless --validate is used.")

    output_file = generate(args.marker_id, args.size, args.border, args.dictionary)
    update_output_dictionary(args.output_dictionary, output_file, args.marker_id, args.description)
