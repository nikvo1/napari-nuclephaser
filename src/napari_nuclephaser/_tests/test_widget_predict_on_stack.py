from unittest.mock import MagicMock

import numpy as np
from napari.layers import Points

from napari_nuclephaser._widget import predict_on_stack


def test_predict_on_stack_valid(make_napari_viewer, mocker, tmp_path):
    # Setup viewer with a valid 3D stack (10 frames of 100x100 grayscale)
    viewer = make_napari_viewer()
    valid_stack = np.random.randint(0, 255, (10, 100, 100), dtype=np.uint8)
    image_layer = viewer.add_image(valid_stack, name="test_stack")

    # Mock model initialization to return a mock model and model type
    mock_initialize = mocker.patch(
        "napari_nuclephaser._widget.initialize_model"
    )
    mock_detection_model = MagicMock()
    mock_initialize.return_value = (mock_detection_model, "mock_model")

    # Mock get_sliced_prediction to return a result with two detections per frame
    mock_sliced_pred = mocker.patch(
        "napari_nuclephaser._widget.get_sliced_prediction"
    )
    mock_result = MagicMock()
    mock_result.to_coco_predictions.return_value = [
        {"bbox": [10, 10, 20, 20]},  # x, y, width, height
        {"bbox": [30, 30, 20, 20]},
    ]
    mock_sliced_pred.return_value = mock_result

    # Mock show functions to track calls
    mock_show_info = mocker.patch("napari_nuclephaser._widget.show_info")
    mock_show_error = mocker.patch("napari_nuclephaser._widget.show_error")

    # Instantiate the widget and call it with test parameters
    widget = predict_on_stack()
    widget(
        Select_stack=image_layer,
        Select_model="dummy_model_path",
        viewer=viewer,
        Postprocess="GREEDYNMM",
        Match_metric="IOS",
        Confidence_threshold=0.5,
        Sahi_size=640,
        Sahi_overlap=0.2,
        Intersection_threshold=0.3,
        Points_size=30,
        Save_result=False,
        Save_folder=tmp_path,
        Experiment_name="test_exp",
        Save_csv=False,
        Save_xlsx=False,
    )

    # Verify a Points layer was added to the viewer
    assert len(viewer.layers) == 2, "Points layer should be added"
    points_layer = viewer.layers[-1]
    assert isinstance(points_layer, Points), "Added layer should be Points"

    # Check points data: 10 frames * 2 detections per frame = 20 points
    assert points_layer.data.shape == (20, 3), "Unexpected points shape"

    # Ensure success message is shown and no errors
    mock_show_info.assert_called_once_with(
        "Made predictions for stack successfully!"
    )
    mock_show_error.assert_not_called()


def test_predict_on_stack_invalid_input(make_napari_viewer, mocker):
    # Setup viewer with an invalid 2D image
    viewer = make_napari_viewer()
    invalid_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    image_layer = viewer.add_image(invalid_image, name="invalid_image")

    # Mock show_error to verify it's called
    mock_show_error = mocker.patch("napari_nuclephaser._widget.show_error")

    # Instantiate and call the widget
    widget = predict_on_stack()
    widget(Select_stack=image_layer, viewer=viewer)

    # Check error message and no layers added
    mock_show_error.assert_called_once_with(
        "Chosen image is a single frame, not a stack!"
    )
    assert len(viewer.layers) == 1, "No additional layers should be added"


def test_predict_on_stack_saving_files(make_napari_viewer, mocker, tmp_path):
    # Setup viewer with a small valid stack
    viewer = make_napari_viewer()
    valid_stack = np.random.randint(0, 255, (2, 50, 50), dtype=np.uint8)
    image_layer = viewer.add_image(valid_stack, name="save_test")

    # Mock model and empty predictions to simplify output
    mock_initialize = mocker.patch(
        "napari_nuclephaser._widget.initialize_model"
    )
    mock_initialize.return_value = (MagicMock(), "mock_model")
    mock_sliced_pred = mocker.patch(
        "napari_nuclephaser._widget.get_sliced_prediction"
    )
    mock_result = MagicMock()
    mock_result.to_coco_predictions.return_value = []
    mock_sliced_pred.return_value = mock_result

    # Mock show functions
    mocker.patch("napari_nuclephaser._widget.show_info")
    mocker.patch("napari_nuclephaser._widget.show_error")

    # Call widget with saving enabled
    widget = predict_on_stack()
    widget(
        Select_stack=image_layer,
        viewer=viewer,
        Save_folder=tmp_path,
        Save_result=True,
        Save_csv=True,
        Save_xlsx=False,
        Experiment_name="test_save",
    )

    # Check that the subfolder and files were created
    subfolder = tmp_path / "test_save"
    assert subfolder.exists(), "Subfolder should be created"

    csv_file = subfolder / "save_test count results.csv"
    metadata_file = subfolder / "save_test count metadata.txt"

    assert csv_file.exists(), "CSV file should be saved"
    assert metadata_file.exists(), "Metadata file should be saved"
