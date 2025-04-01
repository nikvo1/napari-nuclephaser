from unittest.mock import Mock, patch

import numpy as np
import pytest

from napari_nuclephaser._widget import make_points


# Tests for make_points function
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
        patch("napari_nuclephaser._widget.initialize_model") as mock_init,
        patch("napari_nuclephaser._widget.get_sliced_prediction") as mock_pred,
    ):
        mock_init.return_value = (Mock(), "mock_model")
        mock_pred.return_value = Mock(
            to_coco_predictions=lambda: mock_prediction
        )

        widget = make_points()
        widget(image_layer, viewer=viewer)

        assert len(viewer.layers) == 2, "Should add points layer"
        points_layer = viewer.layers["2 points test_image"]
        assert len(points_layer.data) == 2, "Should create 2 points"


def test_make_points_bbox_generation(make_napari_viewer, mock_prediction):
    viewer = make_napari_viewer()
    image_layer = viewer.add_image(
        np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    )

    with (
        patch("napari_nuclephaser._widget.initialize_model") as mock_init,
        patch("napari_nuclephaser._widget.get_sliced_prediction") as mock_pred,
    ):
        mock_init.return_value = (Mock(), "mock_model")
        mock_pred.return_value = Mock(
            to_coco_predictions=lambda: mock_prediction
        )

        widget = make_points()
        widget(image_layer, viewer=viewer, Generate_bbox=True)

        shapes_layer = viewer.layers[-1]
        assert len(shapes_layer.data) == 2, "Should create 2 bounding boxes"
        assert shapes_layer.edge_width[0] == 5, "Should use default thickness"


def test_make_points_both_outputs(make_napari_viewer, mock_prediction):
    viewer = make_napari_viewer()
    image_layer = viewer.add_image(
        np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    )

    with (
        patch("napari_nuclephaser._widget.initialize_model") as mock_init,
        patch("napari_nuclephaser._widget.get_sliced_prediction") as mock_pred,
    ):
        mock_init.return_value = (Mock(), "mock_model")
        mock_pred.return_value = Mock(
            to_coco_predictions=lambda: mock_prediction
        )

        widget = make_points()
        widget(
            image_layer,
            viewer=viewer,
            Generate_points=True,
            Generate_bbox=True,
        )

        assert len(viewer.layers) == 3, "Should have image + points + shapes"
        assert "2 points" in viewer.layers[-2].name
        assert "2 bounding boxes" in viewer.layers[-1].name


def test_make_points_default_generation(make_napari_viewer, mock_prediction):
    viewer = make_napari_viewer()
    image_layer = viewer.add_image(
        np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    )

    with (
        patch("napari_nuclephaser._widget.initialize_model") as mock_init,
        patch("napari_nuclephaser._widget.get_sliced_prediction") as mock_pred,
    ):
        mock_init.return_value = (Mock(), "mock_model")
        mock_pred.return_value = Mock(
            to_coco_predictions=lambda: mock_prediction
        )

        widget = make_points()
        widget(
            image_layer,
            viewer=viewer,
            Generate_points=False,
            Generate_bbox=False,
        )

        assert "points" in viewer.layers[-1].name, "Should default to points"


def test_make_points_output_messages(
    make_napari_viewer, mock_prediction, capsys
):
    viewer = make_napari_viewer()
    image_layer = viewer.add_image(
        np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    )

    with (
        patch("napari_nuclephaser._widget.initialize_model") as mock_init,
        patch("napari_nuclephaser._widget.get_sliced_prediction") as mock_pred,
    ):
        mock_init.return_value = (Mock(), "mock_model")
        mock_pred.return_value = Mock(
            to_coco_predictions=lambda: mock_prediction
        )

        widget = make_points()
        widget(image_layer, viewer=viewer, Generate_bbox=True)

        captured = capsys.readouterr()
        assert "Initializing model..." in captured.out
        assert "Performing sliced prediction..." in captured.out
        assert "Generating boxes..." in captured.out


def test_make_points_error_handling(make_napari_viewer):
    viewer = make_napari_viewer()
    # Create invalid 3D image
    image_layer = viewer.add_image(np.random.rand(5, 100, 100))  # 5 frames

    widget = make_points()
    result = widget(image_layer, viewer=viewer)

    assert result is None, "Should return None on error"
    assert len(viewer.layers) == 1, "Shouldn't add layers on error"


def test_make_points_parameter_effects(make_napari_viewer, mock_prediction):
    viewer = make_napari_viewer()
    image_layer = viewer.add_image(
        np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    )

    with (
        patch("napari_nuclephaser._widget.initialize_model") as mock_init,
        patch("napari_nuclephaser._widget.get_sliced_prediction") as mock_pred,
    ):
        mock_init.return_value = (Mock(), "mock_model")
        mock_pred.return_value = Mock(
            to_coco_predictions=lambda: mock_prediction
        )

        widget = make_points()
        widget(
            image_layer,
            viewer=viewer,
            Generate_bbox=True,
            Points_size=15,
            Bbox_thickness=2,
            Score_text_size=5,
            Show_confidence=True,
        )

        shapes_layer = viewer.layers[-1]

        assert shapes_layer.edge_width[0] == 2, "Should respect bbox thickness"
        assert shapes_layer.text.size == 5, "Should respect score text size"
