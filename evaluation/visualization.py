import os

import cv2


def create_overlay(image_path: str, matches: list[dict], output_path: str) -> None:
    """
    Create an overlay image showing the segmentation and ArUco marker matches.

    Parameters
    ----------
    image_path : str
        The path to the original RGB image.
    matches : list[dict]
        A list of dictionaries containing the segmentation and ArUco marker information.
    output_path : str
        The path where the overlay image will be saved.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    for match in matches:
        seg_center = (int(match["seg_center_x"]), int(match["seg_center_y"]))
        aruco_center = (int(match["aruco_center_x"]), int(match["aruco_center_y"]))

        cv2.circle(image, seg_center, 10, (0, 255, 0), -1)
        cv2.circle(image, aruco_center, 10, (0, 0, 255), -1)
        cv2.line(image, seg_center, aruco_center, (0, 255, 255), 3)

        cv2.putText(
            image,
            f"{match['error_px']:.1f}px",
            seg_center,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cv2.imwrite(output_path, image)
