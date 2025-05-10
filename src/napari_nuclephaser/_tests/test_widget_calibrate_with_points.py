import matplotlib

matplotlib.use("Agg")  # Set backend before other imports

import pathlib
from unittest.mock import MagicMock

import numpy as np

from napari_nuclephaser._widget import calibrate_with_points


def test_calibrate_with_points_happy_path(make_napari_viewer, mocker):
    # Setup viewer with valid 2D image and points
    viewer = make_napari_viewer()
    phase_data = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
    points_data = np.array([[50, 50], [150, 150]])

    # Add layers and get points layer reference
    viewer.add_image(phase_data, name="Phase")
    points_layer = viewer.add_points(points_data, name="Nuclei")

    # Mock the points layer save method directly
    mock_points_save = mocker.patch.object(points_layer, "save")

    # Mock model initialization and predictions
    mocker.patch(
        "napari_nuclephaser._widget.initialize_model",
        return_value=(MagicMock(), "Phase_Model"),
    )
    mock_result = MagicMock()
    mock_result.object_prediction_list = [
        MagicMock(score=MagicMock(value=0.7))
    ]
    mocker.patch(
        "napari_nuclephaser._widget.get_sliced_prediction",
        return_value=mock_result,
    )

    # Mock file operations
    mock_create_folder = mocker.patch(
        "napari_nuclephaser._widget.create_unique_subfolder",
        return_value="/mock/path",
    )
    mocker.patch("matplotlib.pyplot.savefig")
    mocker.patch("builtins.open", mocker.mock_open())

    # Execute widget
    widget = calibrate_with_points()
    result = widget(
        viewer.layers["Phase"],
        points_layer,
        viewer=viewer,
        Division_size=100,
        Calibration_proportion=0.3,
        Save_folder=pathlib.Path("/tmp"),
        Experiment_name="Test",
    )

    # Verify core functionality
    assert "Best threshold for" in result
    assert "MAPE" in result

    # Verify points layer saving
    expected_path = pathlib.Path("/mock/path/reference points.csv")
    args, _ = mock_points_save.call_args
    assert pathlib.Path(args[0]) == expected_path

    # Verify metadata file creation
    mock_create_folder.assert_called()


def test_calibrate_with_points_error_image_shape(make_napari_viewer, mocker):
    viewer = make_napari_viewer()
    mock_error = mocker.patch("napari_nuclephaser._widget.show_error")

    # Test 3D phase image
    viewer.add_image(
        np.random.randint(0, 256, (5, 100, 100), dtype=np.uint8),
        name="3D_Phase",
    )
    points_layer = viewer.add_points(
        np.array([[50, 50], [150, 150]]), name="Nuclei"
    )
    widget = calibrate_with_points()
    result = widget(viewer.layers["3D_Phase"], points_layer, viewer=viewer)
    mock_error.assert_called_with(
        "Phase image is not a single frame! Can't calibrate on a stack of images"
    )
    assert result is None


def test_calibrate_with_points_error_empty_points(make_napari_viewer, mocker):
    viewer = make_napari_viewer()
    mock_error = mocker.patch("napari_nuclephaser._widget.show_error")

    # Test empty points layer
    viewer.add_image(
        np.random.randint(0, 256, (100, 100), dtype=np.uint8), name="Phase"
    )
    viewer.add_points(np.array([]), name="Empty")
    widget = calibrate_with_points()
    result = widget(
        viewer.layers["Phase"], viewer.layers["Empty"], viewer=viewer
    )
    mock_error.assert_called_with(
        "Points layer is empty! Can't proceed further"
    )
    assert result is None


def test_calibrate_with_points_file_creation(
    make_napari_viewer, mocker, tmp_path
):
    viewer = make_napari_viewer()

    # Setup valid data
    phase_layer = viewer.add_image(
        np.random.randint(0, 256, (2560, 2560), dtype=np.uint8)
    )

    points = []
    for i in range(0, 2560, 30):
        for j in range(0, 2560, 30):
            points.append([i, j])
    points_layer = viewer.add_points(np.array(points))

    # Mock dependencies
    mocker.patch(
        "napari_nuclephaser._widget.initialize_model",
        return_value=(MagicMock(), "mock_model"),
    )
    mock_pred = MagicMock()
    mock_pred.object_prediction_list = [
        MagicMock(score=MagicMock(value=x)) for x in np.arange(0.01, 1, 0.01)
    ]
    mocker.patch(
        "napari_nuclephaser._widget.get_sliced_prediction",
        return_value=mock_pred,
    )

    # Setup test directory
    test_save_dir = tmp_path / "points_test"
    test_save_dir.mkdir()
    mocker.patch(
        "napari_nuclephaser._widget.create_unique_subfolder",
        return_value=str(test_save_dir),
    )

    # Execute
    widget = calibrate_with_points()
    widget(
        viewer=viewer,
        Select_Phase_image=phase_layer,
        Select_points_layer=points_layer,
        Division_size=256,
        Calibration_proportion=0.2,
        Save_folder=tmp_path,
    )

    # Verify files
    assert (test_save_dir / "Calibration error plot.png").exists()
    assert (test_save_dir / "metadata.txt").exists()
    assert (test_save_dir / "reference points.csv").exists()
