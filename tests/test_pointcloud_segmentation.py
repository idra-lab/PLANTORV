"""Tests for the architecture-neutral point-cloud segmentation interface."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import open3d as o3d
import torch

from segment_pcd import (
    Open3DSemanticSegmenter,
    _enable_pointtransformer_device_batching,
    _enable_pointtransformer_small_cloud_knn,
    prepare_semantic_input,
)


def _cloud() -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    cloud.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.5], [0.0, 1.0, 0.25]])
    return cloud


class PointCloudSegmentationTest(unittest.TestCase):
    def test_pointtransformer_knn_pads_when_fewer_than_16_points_remain(self) -> None:
        _enable_pointtransformer_small_cloud_knn()
        from open3d._ml3d.torch.models import point_transformer

        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        row_splits = torch.tensor([0, len(points)], dtype=torch.int64)

        indices = point_transformer.knn_batch(
            points,
            points,
            k=16,
            points_row_splits=row_splits,
            queries_row_splits=row_splits,
            return_distances=False,
        )

        self.assertEqual(indices.shape, (5, 16))
        self.assertTrue(torch.all(indices >= 0))
        self.assertTrue(torch.all(indices < len(points)))

    def test_pointtransformer_batch_is_moved_to_pipeline_device(self) -> None:
        _enable_pointtransformer_device_batching()
        from open3d._ml3d.torch.dataloaders import concat_batcher

        batcher = concat_batcher.ConcatBatcher(
            torch.device("cpu"),
            model="PointTransformer",
        )
        sample = {
            "data": {
                "point": torch.zeros((2, 3)),
                "feat": torch.zeros((2, 3)),
                "label": torch.zeros(2, dtype=torch.int64),
            }
        }

        with patch.object(concat_batcher.PointTransformerBatch, "to") as move:
            batcher.collate_fn([sample])

        move.assert_called_once_with(torch.device("cpu"))

    def test_segmenter_uses_the_same_contract_for_both_models(self) -> None:
        model = SimpleNamespace(cfg=SimpleNamespace(in_channels=6))
        pipeline = SimpleNamespace(
            run_inference=lambda data: {
                "predict_labels": np.array([1, 0]),
                "predict_scores": np.array([[0.1, 0.9], [0.8, 0.2]]),
            }
        )

        for model_name in ("RandLANet", "PointTransformer"):
            with self.subTest(model_name=model_name):
                cfg = SimpleNamespace(model=SimpleNamespace(name=model_name))
                result = Open3DSemanticSegmenter(pipeline, model, cfg).segment(
                    _cloud(),
                    use_z_up=False,
                )

                np.testing.assert_array_equal(result.labels, [1, 0])
                np.testing.assert_allclose(result.confidence, [0.9, 0.8])

    def test_prepare_input_converts_coordinates_and_rgb_range(self) -> None:
        model = SimpleNamespace(cfg=SimpleNamespace(in_channels=6))

        data = prepare_semantic_input(_cloud(), model=model, use_z_up=True)

        np.testing.assert_array_equal(
            data["point"],
            [[3.0, -1.0, -2.0], [6.0, -4.0, -5.0]],
        )
        np.testing.assert_allclose(
            data["feat"],
            [[255.0, 0.0, 127.5], [0.0, 255.0, 63.75]],
        )
        np.testing.assert_array_equal(data["label"], [0, 0])

    def test_segmenter_rejects_malformed_model_scores(self) -> None:
        model = SimpleNamespace(cfg=SimpleNamespace(in_channels=6))
        pipeline = SimpleNamespace(
            run_inference=lambda data: {
                "predict_labels": np.array([1, 0]),
                "predict_scores": np.array([0.9, 0.8]),
            }
        )

        with self.assertRaisesRegex(RuntimeError, "invalid prediction scores"):
            Open3DSemanticSegmenter(pipeline, model, cfg=None).segment(
                _cloud(),
                use_z_up=False,
            )


if __name__ == "__main__":
    unittest.main()
