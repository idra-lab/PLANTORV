"""Tests for the Femto camera model and depth-to-RGB alignment in ``mapping``.

The depth code is pure NumPy/OpenCV math, so it can be tested fully headless
(no GPU, model checkpoint, or network required).
"""

import os
import tempfile
import unittest

import cv2
import numpy as np

from depth import depth_provider as dp
from mapping import camera_model as cm
from mapping import rgbd_mapper as rm


class TestFromHardcoded(unittest.TestCase):
    def test_matches_calibration_640x360(self) -> None:
        m = rm.RGBDMapper.from_hardcoded(color_size=(640, 360), depth_size=(1024, 1024))
        self.assertIs(m.calibration, cm._HARDCODED_CALIBRATIONS[0])

    def test_matches_calibration_640x480(self) -> None:
        m = rm.RGBDMapper.from_hardcoded(color_size=(640, 480), depth_size=(1024, 1024))
        self.assertIs(m.calibration, cm._HARDCODED_CALIBRATIONS[1])

    def test_no_match_raises(self) -> None:
        with self.assertRaises(ValueError):
            rm.RGBDMapper.from_hardcoded(color_size=(123, 456), depth_size=(7, 8))


class TestDistortionRoundtrip(unittest.TestCase):
    def setUp(self) -> None:
        cal = cm._HARDCODED_CALIBRATIONS[0]
        self.intr = cal.rgb_intrinsic
        self.dist = cal.rgb_distortion

    def test_project_then_undistort_recovers_normalized(self) -> None:
        x = np.array([0.0, 0.1, -0.15, 0.05])
        y = np.array([0.0, -0.05, 0.12, -0.2])
        u, v = cm._project_to_pixels(x, y, self.intr, self.dist)
        xu, yu = cm._undistort_pixels_to_normalized(u, v, self.intr, self.dist)
        np.testing.assert_allclose(xu, x, atol=1e-6)
        np.testing.assert_allclose(yu, y, atol=1e-6)

    def test_zero_distortion_is_identity_center(self) -> None:
        # At the optical center the projection returns (cx, cy).
        u, v = cm._project_to_pixels(np.array([0.0]), np.array([0.0]), self.intr, self.dist)
        self.assertAlmostEqual(float(u[0]), self.intr.cx, places=6)
        self.assertAlmostEqual(float(v[0]), self.intr.cy, places=6)


class TestAlignDepthToColor(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = rm.RGBDMapper.from_hardcoded(color_size=(640, 360), depth_size=(1024, 1024))
        self.rgb_h = self.mapper.calibration.rgb_intrinsic.height
        self.rgb_w = self.mapper.calibration.rgb_intrinsic.width

    def test_non_2d_depth_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.mapper.align_depth_to_color_with_correspondence(np.zeros((10, 10, 3)))

    def test_wrong_depth_shape_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.mapper.align_depth_to_color_with_correspondence(np.zeros((100, 100), np.uint16))

    def test_all_zero_depth_returns_empty_maps(self) -> None:
        depth = np.zeros((1024, 1024), np.uint16)
        aligned, su, sv = self.mapper.align_depth_to_color_with_correspondence(depth)
        self.assertEqual(aligned.shape, (self.rgb_h, self.rgb_w))
        self.assertEqual(aligned.dtype, np.float32)
        self.assertTrue(np.all(aligned == 0.0))
        self.assertTrue(np.all(su == -1))
        self.assertTrue(np.all(sv == -1))

    def test_uniform_depth_projects(self) -> None:
        depth = np.full((1024, 1024), 1000, np.uint16)
        aligned, su, sv = self.mapper.align_depth_to_color_with_correspondence(
            depth, depth_unit_scale=1.0
        )
        self.assertEqual(aligned.shape, (self.rgb_h, self.rgb_w))
        self.assertEqual(su.dtype, np.int32)
        self.assertEqual(sv.dtype, np.int32)
        # A uniform depth field should project onto most of the RGB plane.
        self.assertGreater(np.mean(aligned > 0), 0.5)
        # Wherever we have a valid source pixel, aligned depth must be positive.
        valid = su >= 0
        self.assertTrue(np.all(aligned[valid] > 0))

    def test_align_depth_to_color_returns_2d(self) -> None:
        depth = np.full((1024, 1024), 800, np.uint16)
        aligned = self.mapper.align_depth_to_color(depth)
        self.assertEqual(aligned.ndim, 2)
        self.assertEqual(aligned.shape, (self.rgb_h, self.rgb_w))

    def test_get_depth_at_rgb_out_of_bounds(self) -> None:
        depth = np.full((1024, 1024), 800, np.uint16)
        self.assertIsNone(self.mapper.get_depth_at_rgb(depth, -1, 0))
        self.assertIsNone(self.mapper.get_depth_at_rgb(depth, 0, 10_000))


class TestFindDepthAtRgb(unittest.TestCase):
    def setUp(self) -> None:
        self.aligned = np.zeros((5, 5), np.float32)

    def test_exact_hit(self) -> None:
        self.aligned[2, 3] = 500.0
        self.assertEqual(rm._find_depth_at_rgb(self.aligned, 3, 2, 0), 500.0)

    def test_neighborhood_search(self) -> None:
        # Nothing at (3,2) but a valid neighbor at (4,2).
        self.aligned[2, 4] = 300.0
        self.assertEqual(rm._find_depth_at_rgb(self.aligned, 3, 2, 1), 300.0)

    def test_out_of_bounds_returns_none(self) -> None:
        self.assertIsNone(rm._find_depth_at_rgb(self.aligned, -1, 0, 1))

    def test_no_valid_returns_none(self) -> None:
        self.assertIsNone(rm._find_depth_at_rgb(self.aligned, 2, 2, 0))


class TestMainCoords(unittest.TestCase):
    def test_populates_center_and_depth(self) -> None:
        tmp = tempfile.mkdtemp()
        rgb_path = os.path.join(tmp, "rgb.png")
        depth_path = os.path.join(tmp, "depth.png")
        cv2.imwrite(rgb_path, np.zeros((360, 640, 3), np.uint8))
        cv2.imwrite(depth_path, np.full((1024, 1024), 1000, np.uint16))

        dict_objects = {"mask_0": {"bbox": [300, 160, 40, 40]}}
        out = dp.main_coords(rgb_path, depth_path, dict_objects)

        self.assertIn("coord_center&depth", out["mask_0"])
        cx, cy, depth_mm = out["mask_0"]["coord_center&depth"]
        # Center of bbox [300,160,40,40] -> (320, 180).
        self.assertEqual(cx, 320)
        self.assertEqual(cy, 180)
        # Uniform depth -> a positive reading at the center.
        self.assertIsNotNone(depth_mm)
        self.assertGreater(depth_mm, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
