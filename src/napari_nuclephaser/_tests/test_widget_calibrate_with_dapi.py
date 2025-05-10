import matplotlib

matplotlib.use("Agg")  # Set backend before other imports

import pathlib
from unittest.mock import MagicMock

import numpy as np

from napari_nuclephaser._widget import calibrate_with_dapi_image


def test_calibrate_happy_path(make_napari_viewer, mocker):
    # Setup viewer with valid 2D images
    viewer = make_napari_viewer()
    phase_data = np.random.randint(0, 256, (2560, 2560), dtype=np.uint8)
    dapi_data = np.random.randint(0, 256, (2560, 2560), dtype=np.uint8)

    viewer.add_image(phase_data, name="Phase")
    viewer.add_image(dapi_data, name="DAPI")

    # Mock external dependencies
    mock_initialize = mocker.patch(
        "napari_nuclephaser._widget.initialize_model"
    )
    mock_initialize.side_effect = [
        (MagicMock(), "DAPI_Model"),
        (MagicMock(), "Phase_Model"),
        (MagicMock(), "Calibrated_Model"),
    ]

    mock_sahi = mocker.patch(
        "napari_nuclephaser._widget.get_sliced_prediction"
    )
    mock_result = MagicMock()
    mock_result.object_prediction_list = [
        MagicMock(score=MagicMock(value=x)) for x in np.arange(0.01, 1, 0.01)
    ]
    mock_sahi.return_value = mock_result

    # Mock file operations
    mock_create_folder = mocker.patch(
        "napari_nuclephaser._widget.create_unique_subfolder"
    )
    mock_create_folder.return_value = "/mock/path"

    # Mock filesystem interactions
    mocker.patch("os.makedirs")
    mocker.patch("matplotlib.pyplot.savefig")
    mocker.patch("PIL.Image.Image.save")
    mock_file = mocker.patch("builtins.open", mocker.mock_open())

    # Instantiate and call widget
    widget = calibrate_with_dapi_image()
    result = widget(
        viewer.layers["Phase"],
        viewer.layers["DAPI"],
        viewer=viewer,
        Division_size=256,
        Calibration_proportion=0.2,
        DAPI_confidence_threshold=0.5,
        Save_folder=pathlib.Path("/tmp"),
        Experiment_name="Test",
    )

    # Verify core functionality
    assert "Best threshold for" in result
    assert "MAPE" in result
    mock_sahi.assert_called()
    mock_create_folder.assert_called()
    expected_path = pathlib.Path("/mock/path/metadata.txt")
    args, _ = mock_file.call_args
    assert pathlib.Path(args[0]) == expected_path
    mock_file().write.assert_called()


def test_image_dimension_errors(make_napari_viewer, mocker):
    viewer = make_napari_viewer()
    mock_error = mocker.patch("napari_nuclephaser._widget.show_error")

    # Test 3D phase image
    viewer.add_image(np.random.randint(0, 256, (5, 100, 100)), name="3D_Phase")
    viewer.add_image(np.random.randint(0, 256, (100, 100)), name="DAPI")
    widget = calibrate_with_dapi_image()
    result = widget(
        viewer.layers["3D_Phase"], viewer.layers["DAPI"], viewer=viewer
    )
    mock_error.assert_called_with(
        "Phase image is not a single frame! Can't calibrate on a stack of images"
    )
    assert result is None

    # Test shape mismatch
    viewer.add_image(np.random.randint(0, 256, (200, 200)), name="Phase")
    viewer.add_image(np.random.randint(0, 256, (100, 100)), name="DAPI")
    widget = calibrate_with_dapi_image()
    result = widget(
        viewer.layers["Phase"], viewer.layers["DAPI"], viewer=viewer
    )
    mock_error.assert_called_with(
        "Phase and DAPI images have different dimensions! Phase image is (200, 200) and DAPI is (100, 100)"
    )
    assert result is None


def test_file_creation_basic(make_napari_viewer, mocker, tmp_path):
    """Verify calibration creates plot and metadata files in target directory."""
    viewer = make_napari_viewer()

    # Create valid test images
    image_size = 2560
    phase_layer = viewer.add_image(
        np.random.randint(0, 256, (image_size, image_size)), name="Phase"
    )
    dapi_layer = viewer.add_image(
        np.random.randint(0, 256, (image_size, image_size)), name="DAPI"
    )

    # Mock model/prediction components
    mock_model = MagicMock()
    mocker.patch(
        "napari_nuclephaser._widget.initialize_model",
        return_value=(mock_model, "mock_model"),
    )

    # Mock predictions with valid results
    mock_pred = MagicMock()
    mock_pred.object_prediction_list = [
        MagicMock(score=MagicMock(value=x)) for x in np.arange(0.01, 1, 0.01)
    ]
    mocker.patch(
        "napari_nuclephaser._widget.get_sliced_prediction",
        return_value=mock_pred,
    )

    # Create and prepare test directory
    test_save_dir = tmp_path / "test_experiment"
    test_save_dir.mkdir()  # ← KEY FIX: Create directory before saving

    # Mock path handling
    mocker.patch(
        "napari_nuclephaser._widget.create_unique_subfolder",
        return_value=str(test_save_dir),
    )

    # Execute widget
    widget = calibrate_with_dapi_image()
    widget(
        viewer=viewer,
        Select_Phase_image=phase_layer,
        Select_DAPI_image=dapi_layer,
        Division_size=256,
        Calibration_proportion=0.2,
        Save_folder=tmp_path,
        Experiment_name="test_experiment",
    )

    # Verify file existence
    assert (test_save_dir / "Calibration error plot.png").exists()
    assert (test_save_dir / "metadata.txt").exists()
