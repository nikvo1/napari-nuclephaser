from unittest.mock import Mock, patch

import numpy as np
import pytest

from napari_nuclephaser.predict_on_single import make_points


@pytest.fixture
def mock_prediction():
    return [
        {"bbox": [10, 10, 20, 20], "score": 0.8},
        {"bbox": [30, 30, 40, 40], "score": 0.9},
    ]


def test_make_points_basic(make_napari_viewer, mock_prediction):
    viewer = make_napari_viewer()
    image_layer = viewer.add_image(
        np.random.randint(0, 256, (100, 100), dtype=np.uint8),
        name="test_image",
    )

    with (
        patch(
            "napari_nuclephaser.predict_on_single.initialize_model"
        ) as mock_init,
        patch(
            "napari_nuclephaser.predict_on_single.get_sliced_prediction"
        ) as mock_pred,
    ):
        mock_init.return_value = (Mock(), "mock_model")
        mock_pred.return_value = Mock(
            to_coco_predictions=lambda: mock_prediction
        )

        widget = make_points()
        widget(
            Select_image=image_layer,
            viewer=viewer,
            Output_format="Points",
            Detection_mode="Regular detection",
        )

        assert len(viewer.layers) == 2
        points_layer = viewer.layers["2 points test_image"]
        assert len(points_layer.data) == 2


def test_make_points_bbox_generation(make_napari_viewer, mock_prediction):
    viewer = make_napari_viewer()
    image_layer = viewer.add_image(
        np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    )

    with (
        patch(
            "napari_nuclephaser.predict_on_single.initialize_model"
        ) as mock_init,
        patch(
            "napari_nuclephaser.predict_on_single.get_sliced_prediction"
        ) as mock_pred,
    ):
        mock_init.return_value = (Mock(), "mock_model")
        mock_pred.return_value = Mock(
            to_coco_predictions=lambda: mock_prediction
        )

        widget = make_points()
        widget(
            Select_image=image_layer,
            viewer=viewer,
            Output_format="Bounding boxes",
            Detection_mode="Regular detection",
        )

        shapes_layer = viewer.layers[-1]
        assert len(shapes_layer.data) == 2
        assert shapes_layer.edge_width[0] == 5


def test_make_points_with_confidence(make_napari_viewer, mock_prediction):
    viewer = make_napari_viewer()
    image_layer = viewer.add_image(
        np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    )

    with (
        patch(
            "napari_nuclephaser.predict_on_single.initialize_model"
        ) as mock_init,
        patch(
            "napari_nuclephaser.predict_on_single.get_sliced_prediction"
        ) as mock_pred,
    ):
        mock_init.return_value = (Mock(), "mock_model")
        mock_pred.return_value = Mock(
            to_coco_predictions=lambda: mock_prediction
        )

        widget = make_points()
        widget(
            Select_image=image_layer,
            viewer=viewer,
            Output_format="Bounding boxes with confidence scores",
            Detection_mode="Regular detection",
        )

        shapes_layer = viewer.layers[-1]
        assert shapes_layer.text is not None


def test_make_points_default_generation(make_napari_viewer, mock_prediction):
    viewer = make_napari_viewer()
    image_layer = viewer.add_image(
        np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    )

    with (
        patch(
            "napari_nuclephaser.predict_on_single.initialize_model"
        ) as mock_init,
        patch(
            "napari_nuclephaser.predict_on_single.get_sliced_prediction"
        ) as mock_pred,
    ):
        mock_init.return_value = (Mock(), "mock_model")
        mock_pred.return_value = Mock(
            to_coco_predictions=lambda: mock_prediction
        )

        widget = make_points()
        widget(
            Select_image=image_layer,
            viewer=viewer,
            Detection_mode="Regular detection",
        )

        assert "points" in viewer.layers[-1].name.lower()


def test_make_points_error_handling(make_napari_viewer):
    viewer = make_napari_viewer()
    image_layer = viewer.add_image(np.random.rand(5, 100, 100))

    widget = make_points()
    with patch(
        "napari_nuclephaser.predict_on_single.show_modal_error"
    ) as mock_error:
        result = widget(
            Select_image=image_layer,
            viewer=viewer,
            Detection_mode="Regular detection",
        )

    assert result is None
    assert len(viewer.layers) == 1
    mock_error.assert_called_once()
    assert "Image is not a single frame" in mock_error.call_args[0][0]


def test_make_points_parameter_effects(make_napari_viewer, mock_prediction):
    viewer = make_napari_viewer()
    image_layer = viewer.add_image(
        np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    )

    with (
        patch(
            "napari_nuclephaser.predict_on_single.initialize_model"
        ) as mock_init,
        patch(
            "napari_nuclephaser.predict_on_single.get_sliced_prediction"
        ) as mock_pred,
    ):
        mock_init.return_value = (Mock(), "mock_model")
        mock_pred.return_value = Mock(
            to_coco_predictions=lambda: mock_prediction
        )

        widget = make_points()
        widget(
            Select_image=image_layer,
            viewer=viewer,
            Output_format="Bounding boxes with confidence scores",
            Points_size=15,
            Bbox_thickness=2,
            Score_text_size=5,
            Detection_mode="Regular detection",
        )

        shapes_layer = viewer.layers[-1]
        assert shapes_layer.edge_width[0] == 2
        assert shapes_layer.text.size == 5
