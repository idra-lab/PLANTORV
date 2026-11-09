"""Tests for the depth module (depth.femto_camera).

The depth code is pure NumPy/OpenCV math, so it can be tested fully headless
(no GPU, model checkpoint, or network required).

Run:
    ./venv/bin/python -m unittest tests.test_depth
"""

import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

# Allow running this file directly (python tests/test_depth.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depth import femto_camera as fc


class TestFromHardcoded(unittest.TestCase):
    def test_matches_calibration_640x360(self):
        m = fc.DepthRgbMapper.from_hardcoded(color_size=(640, 360), depth_size=(1024, 1024))
        self.assertIs(m.calibration, fc._HARDCODED_CALIBRATIONS[0])

    def test_matches_calibration_640x480(self):
        m = fc.DepthRgbMapper.from_hardcoded(color_size=(640, 480), depth_size=(1024, 1024))
        self.assertIs(m.calibration, fc._HARDCODED_CALIBRATIONS[1])

    def test_no_match_raises(self):
        with self.assertRaises(ValueError):
            fc.DepthRgbMapper.from_hardcoded(color_size=(123, 456), depth_size=(7, 8))


class TestDistortionRoundtrip(unittest.TestCase):
    def setUp(self):
        cal = fc._HARDCODED_CALIBRATIONS[0]
        self.intr = cal.rgb_intrinsic
        self.dist = cal.rgb_distortion

    def test_project_then_undistort_recovers_normalized(self):
        x = np.array([0.0, 0.1, -0.15, 0.05])
        y = np.array([0.0, -0.05, 0.12, -0.2])
        u, v = fc._project_to_pixels(x, y, self.intr, self.dist)
        xu, yu = fc._undistort_pixels_to_normalized(u, v, self.intr, self.dist)
        np.testing.assert_allclose(xu, x, atol=1e-6)
        np.testing.assert_allclose(yu, y, atol=1e-6)

    def test_zero_distortion_is_identity_center(self):
        # At the optical center the projection returns (cx, cy).
        u, v = fc._project_to_pixels(np.array([0.0]), np.array([0.0]), self.intr, self.dist)
        self.assertAlmostEqual(float(u[0]), self.intr.cx, places=6)
        self.assertAlmostEqual(float(v[0]), self.intr.cy, places=6)


class TestAlignDepthToColor(unittest.TestCase):
    def setUp(self):
        self.mapper = fc.DepthRgbMapper.from_hardcoded(
            color_size=(640, 360), depth_size=(1024, 1024)
        )
        self.rgb_h = self.mapper.calibration.rgb_intrinsic.height
        self.rgb_w = self.mapper.calibration.rgb_intrinsic.width

    def test_non_2d_depth_raises(self):
        with self.assertRaises(ValueError):
            self.mapper.align_depth_to_color_with_correspondence(np.zeros((10, 10, 3)))

    def test_wrong_depth_shape_raises(self):
        with self.assertRaises(ValueError):
            self.mapper.align_depth_to_color_with_correspondence(np.zeros((100, 100), np.uint16))

    def test_all_zero_depth_returns_empty_maps(self):
        depth = np.zeros((1024, 1024), np.uint16)
        aligned, su, sv = self.mapper.align_depth_to_color_with_correspondence(depth)
        self.assertEqual(aligned.shape, (self.rgb_h, self.rgb_w))
        self.assertEqual(aligned.dtype, np.float32)
        self.assertTrue(np.all(aligned == 0.0))
        self.assertTrue(np.all(su == -1))
        self.assertTrue(np.all(sv == -1))

    def test_uniform_depth_projects(self):
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

    def test_align_depth_to_color_returns_2d(self):
        depth = np.full((1024, 1024), 800, np.uint16)
        aligned = self.mapper.align_depth_to_color(depth)
        self.assertEqual(aligned.ndim, 2)
        self.assertEqual(aligned.shape, (self.rgb_h, self.rgb_w))

    def test_get_depth_at_rgb_out_of_bounds(self):
        depth = np.full((1024, 1024), 800, np.uint16)
        self.assertIsNone(self.mapper.get_depth_at_rgb(depth, -1, 0))
        self.assertIsNone(self.mapper.get_depth_at_rgb(depth, 0, 10_000))


class TestFindDepthAndSource(unittest.TestCase):
    def setUp(self):
        self.aligned = np.zeros((5, 5), np.float32)
        self.su = np.full((5, 5), -1, np.int32)
        self.sv = np.full((5, 5), -1, np.int32)

    def test_exact_hit(self):
        self.aligned[2, 3] = 500.0
        self.su[2, 3] = 11
        self.sv[2, 3] = 22
        d, uv = fc._find_depth_and_source(self.aligned, self.su, self.sv, 3, 2, 0)
        self.assertEqual(d, 500.0)
        self.assertEqual(uv, (11, 22))

    def test_neighborhood_search(self):
        # Nothing at (3,2) but a valid neighbor at (4,2).
        self.aligned[2, 4] = 300.0
        self.su[2, 4] = 7
        self.sv[2, 4] = 8
        d, uv = fc._find_depth_and_source(self.aligned, self.su, self.sv, 3, 2, 1)
        self.assertEqual(d, 300.0)
        self.assertEqual(uv, (7, 8))

    def test_out_of_bounds_returns_none(self):
        d, uv = fc._find_depth_and_source(self.aligned, self.su, self.sv, -1, 0, 1)
        self.assertIsNone(d)
        self.assertIsNone(uv)

    def test_no_valid_returns_none(self):
        d, uv = fc._find_depth_and_source(self.aligned, self.su, self.sv, 2, 2, 0)
        self.assertIsNone(d)
        self.assertIsNone(uv)


class TestDepthToColormap(unittest.TestCase):
    def test_valid_depth_produces_bgr(self):
        depth = np.zeros((20, 20), np.float32)
        depth[5:15, 5:15] = np.linspace(100, 900, 100).reshape(10, 10)
        vis = fc._depth_to_colormap(depth)
        self.assertEqual(vis.shape, (20, 20, 3))
        self.assertEqual(vis.dtype, np.uint8)

    def test_non_2d_raises(self):
        with self.assertRaises(ValueError):
            fc._depth_to_colormap(np.zeros((5, 5, 3), np.float32))


class TestMainCoords(unittest.TestCase):
    def test_populates_center_and_depth(self):
        tmp = tempfile.mkdtemp()
        rgb_path = os.path.join(tmp, "rgb.png")
        depth_path = os.path.join(tmp, "depth.png")
        cv2.imwrite(rgb_path, np.zeros((360, 640, 3), np.uint8))
        cv2.imwrite(depth_path, np.full((1024, 1024), 1000, np.uint16))

        dict_objects = {"mask_0": {"bbox": [300, 160, 40, 40]}}
        out = fc.main_coords(rgb_path, depth_path, dict_objects)

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
