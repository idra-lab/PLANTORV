"""Tests for the vlm module (vlm.gpt.GPTModel).

These tests avoid any real network calls: the Azure client's completion call is
mocked. encode_image_data_url is tested against a real temp file. GPTModel is
instantiated with object.__new__ where we don't need a client, and with a real
constructor (dummy credentials, no request issued) to check wiring.

Run:
    ./venv/bin/python -m unittest tests.test_vlm
"""

import base64
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vlm.gpt as gpt_mod
from vlm.gpt import GPTModel


def _bare_model(deployment="gpt-test"):
    """Create a GPT model instance without running ``__init__``.

    Parameters
    ----------
    deployment : str, optional
        Deployment name to attach to the bare model.

    Returns
    -------
    GPTModel
        Uninitialized model instance with ``deployment`` set.
    """
    m = object.__new__(GPTModel)
    m.deployment = deployment
    return m


def _fake_response(content):
    """Build a fake OpenAI SDK chat completion response.

    Parameters
    ----------
    content : str
        Message content returned by the fake completion.

    Returns
    -------
    unittest.mock.Mock
        Mock response with a ``choices[0].message.content`` attribute.
    """
    msg = mock.Mock()
    msg.content = content
    choice = mock.Mock()
    choice.message = msg
    resp = mock.Mock()
    resp.choices = [choice]
    return resp


class TestModuleImports(unittest.TestCase):
    """Regression guard for the missing-imports bug found after the split."""

    def test_required_names_available(self):
        for name in ("AzureOpenAI", "json", "base64", "mimetypes", "Path"):
            self.assertTrue(hasattr(gpt_mod, name), f"vlm.gpt is missing '{name}'")


class TestEncodeImageDataUrl(unittest.TestCase):
    def test_roundtrip_png(self):
        model = _bare_model()
        tmp = tempfile.mkdtemp()
        img_path = Path(tmp) / "crop.png"
        cv2.imwrite(str(img_path), np.full((4, 4, 3), 128, np.uint8))

        url = model.encode_image_data_url(img_path)
        self.assertTrue(url.startswith("data:image/png;base64,"))
        payload = url.split(",", 1)[1]
        decoded = base64.b64decode(payload)
        self.assertEqual(decoded, img_path.read_bytes())

    def test_missing_file_raises(self):
        model = _bare_model()
        with self.assertRaises(FileNotFoundError):
            model.encode_image_data_url(Path("/no/such/image_xyz.png"))


class TestConstructor(unittest.TestCase):
    def test_init_sets_deployment_without_network(self):
        # Constructing AzureOpenAI does not issue a request; dummy creds are fine.
        model = GPTModel(
            endpoint="https://example.openai.azure.com/",
            model_name="gpt-x",
            deployment="dep-x",
            subscription_key="dummy-key",
            api_version="2024-12-01-preview",
        )
        self.assertEqual(model.deployment, "dep-x")
        self.assertIsNotNone(model.client)


class TestMainGpt(unittest.TestCase):
    def _run_with_response(self, raw_content, n_masks=1):
        model = _bare_model()
        model.client = mock.Mock()
        model.client.chat.completions.create.return_value = _fake_response(raw_content)

        tmp = tempfile.mkdtemp()
        full = Path(tmp) / "scene.png"
        cv2.imwrite(str(full), np.zeros((8, 8, 3), np.uint8))
        mask_paths = []
        for i in range(n_masks):
            p = Path(tmp) / f"mask_{i}.png"
            cv2.imwrite(str(p), np.full((4, 4, 3), 200, np.uint8))
            mask_paths.append(str(p))

        bboxes = [[10 * i, 20 * i, 5, 5] for i in range(n_masks)]
        return model.main_gpt(str(full), mask_paths, None, bboxes)

    def test_valid_json_response(self):
        raw = json.dumps({"tag": "red block", "description": "on the left", "full_object": True})
        out = self._run_with_response(raw, n_masks=1)
        self.assertIn("mask_0", out)
        self.assertEqual(out["mask_0"]["tag"], "red block")
        self.assertEqual(out["mask_0"]["description"], "on the left")
        self.assertTrue(out["mask_0"]["full_object"])
        # main_gpt attaches the mask path and bbox.
        self.assertTrue(out["mask_0"]["mask"].endswith("mask_0.png"))
        self.assertEqual(out["mask_0"]["bbox"], [0, 0, 5, 5])

    def test_invalid_json_falls_back(self):
        out = self._run_with_response("this is not json", n_masks=1)
        self.assertEqual(out["mask_0"]["tag"], "unknown")
        self.assertEqual(out["mask_0"]["description"], "unknown")
        self.assertFalse(out["mask_0"]["full_object"])
        self.assertIn("mask", out["mask_0"])

    def test_multiple_masks(self):
        raw = json.dumps({"tag": "t", "description": "d", "full_object": False})
        out = self._run_with_response(raw, n_masks=3)
        self.assertEqual(set(out.keys()), {"mask_0", "mask_1", "mask_2"})
        self.assertEqual(out["mask_2"]["bbox"], [20, 40, 5, 5])


if __name__ == "__main__":
    unittest.main(verbosity=2)
