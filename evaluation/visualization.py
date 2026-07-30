import cv2
import os


def create_overlay(image_path, matches, output_path):

    image = cv2.imread(image_path)

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
