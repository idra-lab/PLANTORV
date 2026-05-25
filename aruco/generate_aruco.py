import argparse
import json
import os
import sys
from pathlib import Path


DEFAULT_OUTPUT_DICTIONARY = "aruco_markers.json"


def load_output_dictionary(path):
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        output_dictionary = json.load(f)

    if not isinstance(output_dictionary, dict):
        raise ValueError(f"{path} must contain a JSON object mapping file paths to marker IDs.")

    return output_dictionary


def save_output_dictionary(path, output_dictionary):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(output_dictionary, f, indent=2, sort_keys=True)
        f.write("\n")


def relative_to_output_dictionary(file_path, output_dictionary_path):
    relative_path = os.path.relpath(file_path.resolve(), output_dictionary_path.parent.resolve())
    return Path(relative_path).as_posix()


def update_output_dictionary(output_dictionary_path, generated_file, marker_id, description):
    output_dictionary = load_output_dictionary(output_dictionary_path)
    relative_file = relative_to_output_dictionary(generated_file, output_dictionary_path)
    output_dictionary[relative_file] = {
        "id": marker_id,
        "description": description,
    }
    save_output_dictionary(output_dictionary_path, output_dictionary)


def validate_output_dictionary(output_dictionary_path):
    output_dictionary = load_output_dictionary(output_dictionary_path)
    missing_files = []
    invalid_entries = []

    for relative_file, marker_data in output_dictionary.items():
        if not isinstance(relative_file, str) or not relative_file or Path(relative_file).is_absolute():
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


def generate(marker_id, marker_size_px=800, border_bits=1, dictionary_name="DICT_6X6_250"):
    import cv2

    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dictionary_name))

    img = aruco.generateImageMarker(
        dictionary,
        marker_id,
        marker_size_px,
        borderBits=border_bits
    )

    output_file = Path(f"aruco_{marker_id}.png")
    if not cv2.imwrite(str(output_file), img):
        raise OSError(f"Failed to write marker image to {output_file}")

    return output_file


if __name__ == "__main__":
    argsparser = argparse.ArgumentParser(description="Generate ArUco marker images.")
    argsparser.add_argument("marker_id", type=int, nargs="?", help="ID of the ArUco marker to generate.")
    argsparser.add_argument(
        "--size",
        type=int,
        default=800,
        help="Size of the generated marker image in pixels (default: 800)."
    )
    argsparser.add_argument("--border", type=int, default=1, help="Number of bits in the marker border (default: 1).")
    argsparser.add_argument(
        "--dictionary",
        type=str,
        default="DICT_6X6_250",
        help="Predefined ArUco dictionary to use (default: DICT_6X6_250)."
    )
    argsparser.add_argument(
        "--output-dictionary",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DICTIONARY),
        help=f"JSON file mapping generated marker paths to marker metadata (default: {DEFAULT_OUTPUT_DICTIONARY})."
    )
    argsparser.add_argument(
        "--description",
        type=str,
        default="",
        help="Description to store in the output dictionary for the generated marker."
    )
    argsparser.add_argument(
        "--validate",
        action="store_true",
        help="Validate that all files listed in the output dictionary still exist."
    )
    args = argsparser.parse_args()

    if args.validate:
        if not args.output_dictionary.exists():
            print(f"{args.output_dictionary} does not exist.")
            sys.exit(1)

        invalid_entries, missing_files = validate_output_dictionary(args.output_dictionary)
        if invalid_entries or missing_files:
            print(f"{args.output_dictionary} is invalid.")
            for entry in invalid_entries:
                print(f"Invalid entry: {entry}")
            for missing_file in missing_files:
                print(f"Missing file: {missing_file}")
            sys.exit(1)

        print(f"{args.output_dictionary} is valid.")
        sys.exit(0)

    if args.marker_id is None:
        argsparser.error("marker_id is required unless --validate is used.")

    output_file = generate(args.marker_id, args.size, args.border, args.dictionary)
    update_output_dictionary(args.output_dictionary, output_file, args.marker_id, args.description)
