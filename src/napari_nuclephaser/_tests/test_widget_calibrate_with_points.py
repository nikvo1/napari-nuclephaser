import matplotlib

matplotlib.use("Agg")  # Set backend before other imports

from unittest.mock import MagicMock, patch

import dask.array as da
import numpy as np
import pandas as pd
import pytest

from napari_nuclephaser.calibrate_points import (
    _ensure_numpy,
    _prepare_frame,
    _split_image_and_points,
    calibrate_with_points,
)
from napari_nuclephaser.utils import initialize_model


# ----------------------------------------------------------------------
# Fixtures for real model initialization (once per test session)
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_yolov5_model():
    """Load real YOLOv5 model from package models folder."""
    from napari_nuclephaser.calibrate_points import (
        cuda_available,
        models_folder,
    )

    model_path = models_folder / "ND_v5n.pt"
    if not model_path.exists():
        pytest.skip(f"Model file not found: {model_path}")
    model, model_type = initialize_model(str(model_path), 0.5, cuda_available)
    return model, model_type


@pytest.fixture(scope="module")
def real_yolov11_model():
    """Load real YOLOv11 model from package models folder."""
    from napari_nuclephaser.calibrate_points import (
        cuda_available,
        models_folder,
    )

    model_path = models_folder / "ND_v11n.pt"
    if not model_path.exists():
        pytest.skip(f"Model file not found: {model_path}")
    model, model_type = initialize_model(str(model_path), 0.5, cuda_available)
    return model, model_type


# ----------------------------------------------------------------------
# Helper: mock get_sliced_prediction to return controlled detection scores
# ----------------------------------------------------------------------
def mock_prediction_with_confidences(confidences):
    """Return a MagicMock that behaves like SAHI result with given confidence scores."""
    mock_result = MagicMock()
    mock_result.object_prediction_list = []
    for conf in confidences:
        box = MagicMock()
        box.score.value = conf
        mock_result.object_prediction_list.append(box)
    return mock_result


# ----------------------------------------------------------------------
# Fixture to mock subfolder creation (avoids timestamp issues)
# ----------------------------------------------------------------------
@pytest.fixture
def mock_subfolder(tmp_path):
    """Mock create_unique_subfolder to return a fixed path inside tmp_path."""
    fake_subfolder = tmp_path / "test_output"
    fake_subfolder.mkdir()
    with patch(
        "napari_nuclephaser.calibrate_points.create_unique_subfolder",
        return_value=str(fake_subfolder),
    ) as mock:
        yield fake_subfolder, mock


# ----------------------------------------------------------------------
# Tests for _ensure_numpy (Dask conversion)
# ----------------------------------------------------------------------
def test_ensure_numpy_with_dask():
    dask_arr = da.from_array(np.random.rand(10, 10), chunks=(5, 5))
    result = _ensure_numpy(dask_arr)
    assert isinstance(result, np.ndarray)
    assert result.shape == (10, 10)


def test_ensure_numpy_with_numpy():
    numpy_arr = np.random.rand(10, 10)
    result = _ensure_numpy(numpy_arr)
    assert result is numpy_arr


# ----------------------------------------------------------------------
# Tests for tile splitting
# ----------------------------------------------------------------------
def test_split_image_and_points():
    image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    points = [(50, 50), (150, 150), (250, 250), (350, 350)]
    window_size = 100
    tiles, counts = _split_image_and_points(image, points, window_size)
    assert len(tiles) == 16
    assert len(counts) == 16
    assert counts[0] == 1
    assert counts[5] == 1
    assert counts[10] == 1
    assert counts[15] == 1


def test_split_image_and_points_empty():
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    tiles, counts = _split_image_and_points(image, [], 50)
    assert len(tiles) == 4
    assert all(c == 0 for c in counts)


# ----------------------------------------------------------------------
# Tests for _prepare_frame
# ----------------------------------------------------------------------
def test_prepare_frame():
    image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    points = [(10, 10), (110, 110)]
    division_size = 100
    calib_prop = 0.5
    calib_tiles, calib_gt, test_tiles, test_gt = _prepare_frame(
        image, points, division_size, calib_prop, random_seed=42
    )
    total_tiles = 4
    n_calib = int(total_tiles * calib_prop)
    assert len(calib_tiles) == n_calib
    assert len(calib_gt) == n_calib
    assert len(test_tiles) == total_tiles - n_calib
    assert len(test_gt) == total_tiles - n_calib
    assert all(gt in (0, 1) for gt in calib_gt + test_gt)


def test_prepare_frame_grayscale_conversion():
    image = np.random.randint(0, 65535, (200, 200), dtype=np.uint16)
    points = []
    calib_tiles, _, _, _ = _prepare_frame(image, points, 100, 0.5, 42)
    for tile in calib_tiles:
        assert tile.shape[-1] == 3
        assert tile.dtype == np.uint8


# ----------------------------------------------------------------------
# Error handling tests
# ----------------------------------------------------------------------
def test_calibrate_error_3d_image_with_2d_points(make_napari_viewer):
    viewer = make_napari_viewer()
    image_3d = np.random.randint(0, 255, (5, 100, 100), dtype=np.uint8)
    points_2d = np.array([[50, 50], [60, 60]])
    viewer.add_image(image_3d, name="stack")
    viewer.add_points(points_2d, name="points")
    widget = calibrate_with_points()
    with patch("napari_nuclephaser.calibrate_points.show_error") as mock_error:
        result = widget(
            viewer.layers["stack"], viewer.layers["points"], viewer=viewer
        )
        mock_error.assert_called_once()
        assert (
            "Points layer has 2 columns (y,x) but image stack has multiple frames"
            in mock_error.call_args[0][0]
        )
        assert result is None


def test_calibrate_error_points_wrong_columns(make_napari_viewer):
    viewer = make_napari_viewer()
    image_2d = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    points_bad = np.array([[1, 2, 3, 4]])
    viewer.add_image(image_2d, name="phase")
    viewer.add_points(points_bad, name="points")
    widget = calibrate_with_points()
    with patch("napari_nuclephaser.calibrate_points.show_error") as mock_error:
        result = widget(
            viewer.layers["phase"], viewer.layers["points"], viewer=viewer
        )
        mock_error.assert_called_once()
        assert "has 4 columns" in mock_error.call_args[0][0]
        assert result is None


def test_calibrate_error_empty_points(make_napari_viewer):
    """Empty points layer should trigger early error."""
    viewer = make_napari_viewer()
    image_2d = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    points_empty = np.array([])
    viewer.add_image(image_2d, name="phase")
    viewer.add_points(points_empty, name="points")
    widget = calibrate_with_points()
    with patch("napari_nuclephaser.calibrate_points.show_error") as mock_error:
        result = widget(
            viewer.layers["phase"], viewer.layers["points"], viewer=viewer
        )
        mock_error.assert_called_once_with(
            "Points layer is empty! Can't proceed further"
        )
        assert result is None


def test_calibrate_error_unsupported_image_ndim(make_napari_viewer):
    viewer = make_napari_viewer()
    image_4d = np.random.randint(0, 255, (2, 2, 100, 100), dtype=np.uint8)
    points = np.array([[0, 50, 50], [0, 60, 60]])
    viewer.add_image(image_4d, name="bad")
    viewer.add_points(points, name="points")
    widget = calibrate_with_points()
    with patch("napari_nuclephaser.calibrate_points.show_error") as mock_error:
        result = widget(
            viewer.layers["bad"], viewer.layers["points"], viewer=viewer
        )
        mock_error.assert_called_once()
        assert "Unsupported image dimensions" in mock_error.call_args[0][0]
        assert result is None


# ----------------------------------------------------------------------
# Successful calibration tests (real model, mocked predictions)
# ----------------------------------------------------------------------
def test_calibrate_success_with_real_model_and_mocked_predictions(
    make_napari_viewer, real_yolov11_model, tmp_path, mock_subfolder
):
    """Test full calibration pipeline with real model but controlled predictions."""
    fake_subfolder, _ = mock_subfolder
    viewer = make_napari_viewer()

    image_data = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    points_data = np.array([[50, 50], [150, 150]])
    viewer.add_image(image_data, name="Phase")
    points_layer = viewer.add_points(points_data, name="Nuclei")

    from napari_nuclephaser import calibrate_points as cp_module

    def mock_prediction_side_effect(*args, **kwargs):
        return mock_prediction_with_confidences(
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        )

    with patch.object(
        cp_module,
        "get_sliced_prediction",
        side_effect=mock_prediction_side_effect,
    ):
        widget = calibrate_with_points()
        result = widget(
            Select_Phase_stack=viewer.layers["Phase"],
            Select_Points_layer=points_layer,
            viewer=viewer,
            Division_size=100,
            Calibration_proportion=0.5,  # ensures at least one calibration tile
            Save_folder=tmp_path,
            Experiment_name="test_calib",
            Random_seed=42,
        )

    assert result is not None
    assert "Best threshold" in result
    assert "MAPE" in result

    # Check saved files in the mocked subfolder
    plot_file = fake_subfolder / "Calibration_error_plot.png"
    assert plot_file.exists()
    meta_file = fake_subfolder / "metadata.txt"
    assert meta_file.exists()
    points_file = fake_subfolder / "reference_points.csv"
    assert points_file.exists()

    df = pd.read_csv(points_file)
    assert list(df.columns) == ["frame", "y", "x"]
    assert len(df) == 2
    assert df["frame"].iloc[0] == 0

    meta_text = meta_file.read_text()
    assert "Overall threshold" in meta_text
    assert "Overall MAPE" in meta_text


def test_calibrate_success_with_real_yolov5_model(
    make_napari_viewer, real_yolov5_model, tmp_path, mock_subfolder
):
    fake_subfolder, _ = mock_subfolder
    viewer = make_napari_viewer()
    image_data = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    points_data = np.array([[50, 50], [150, 150]])
    viewer.add_image(image_data, name="Phase")
    points_layer = viewer.add_points(points_data, name="Nuclei")

    from napari_nuclephaser import calibrate_points as cp_module

    def mock_prediction_side_effect(*args, **kwargs):
        return mock_prediction_with_confidences(
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        )

    with patch.object(
        cp_module,
        "get_sliced_prediction",
        side_effect=mock_prediction_side_effect,
    ):
        widget = calibrate_with_points()
        result = widget(
            Select_Phase_stack=viewer.layers["Phase"],
            Select_Points_layer=points_layer,
            viewer=viewer,
            Division_size=100,
            Calibration_proportion=0.5,
            Save_folder=tmp_path,
        )
    assert result is not None
    assert "Best threshold" in result


def test_calibrate_with_dask_array(
    make_napari_viewer, real_yolov11_model, tmp_path, mock_subfolder
):
    """Image data as Dask array should be converted to numpy."""
    fake_subfolder, _ = mock_subfolder
    viewer = make_napari_viewer()
    numpy_arr = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    dask_arr = da.from_array(numpy_arr, chunks=(100, 100))
    points_data = np.array([[50, 50], [150, 150]])
    viewer.add_image(dask_arr, name="Phase")
    points_layer = viewer.add_points(points_data, name="Nuclei")

    from napari_nuclephaser import calibrate_points as cp_module

    def mock_prediction_side_effect(*args, **kwargs):
        return mock_prediction_with_confidences([0.9, 0.8, 0.7])

    with patch.object(
        cp_module,
        "get_sliced_prediction",
        side_effect=mock_prediction_side_effect,
    ):
        widget = calibrate_with_points()
        result = widget(
            Select_Phase_stack=viewer.layers["Phase"],
            Select_Points_layer=points_layer,
            viewer=viewer,
            Division_size=100,
            Calibration_proportion=0.5,  # ensures calibration tiles exist
            Save_folder=tmp_path,
        )
    assert result is not None
    assert (fake_subfolder / "reference_points.csv").exists()


def test_calibrate_multiframe(
    make_napari_viewer, real_yolov11_model, tmp_path, mock_subfolder
):
    fake_subfolder, _ = mock_subfolder
    viewer = make_napari_viewer()
    stack = np.random.randint(0, 255, (3, 100, 100), dtype=np.uint8)
    points = np.array([[0, 20, 20], [0, 80, 80], [1, 30, 30], [2, 50, 50]])
    viewer.add_image(stack, name="Phase_stack")
    points_layer = viewer.add_points(points, name="Points")

    from napari_nuclephaser import calibrate_points as cp_module

    def mock_prediction_side_effect(*args, **kwargs):
        return mock_prediction_with_confidences([0.9, 0.8, 0.7])

    with patch.object(
        cp_module,
        "get_sliced_prediction",
        side_effect=mock_prediction_side_effect,
    ):
        widget = calibrate_with_points()
        result = widget(
            Select_Phase_stack=viewer.layers["Phase_stack"],
            Select_Points_layer=points_layer,
            viewer=viewer,
            Division_size=50,
            Calibration_proportion=0.5,
            Save_folder=tmp_path,
        )
    assert result is not None
    points_file = fake_subfolder / "reference_points.csv"
    df = pd.read_csv(points_file)
    assert list(df.columns) == ["frame", "y", "x"]
    assert set(df["frame"]) == {0, 1, 2}


def test_calibrate_no_detections(
    make_napari_viewer, real_yolov11_model, tmp_path
):
    """If model makes no detections on calibration tiles, return None."""
    viewer = make_napari_viewer()
    image_data = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    points_data = np.array([[50, 50], [150, 150]])
    viewer.add_image(image_data, name="Phase")
    points_layer = viewer.add_points(points_data, name="Nuclei")

    from napari_nuclephaser import calibrate_points as cp_module

    def empty_prediction(*args, **kwargs):
        mock_res = MagicMock()
        mock_res.object_prediction_list = []
        return mock_res

    with patch.object(
        cp_module, "get_sliced_prediction", side_effect=empty_prediction
    ):
        widget = calibrate_with_points()
        with patch(
            "napari_nuclephaser.calibrate_points.show_error"
        ) as mock_error:
            result = widget(
                Select_Phase_stack=viewer.layers["Phase"],
                Select_Points_layer=points_layer,
                viewer=viewer,
                Division_size=100,
                Calibration_proportion=0.5,
                Save_folder=tmp_path,
            )
            mock_error.assert_called_once()
            assert "No valid calibration data" in mock_error.call_args[0][0]
            assert result is None


def test_calibrate_all_tiles_calibration(
    make_napari_viewer, real_yolov11_model, tmp_path, mock_subfolder
):
    """Calibration_proportion = 1.0 -> no test tiles, but calibration still runs."""
    fake_subfolder, _ = mock_subfolder
    viewer = make_napari_viewer()
    image_data = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    points_data = np.array([[50, 50], [150, 150]])
    viewer.add_image(image_data, name="Phase")
    points_layer = viewer.add_points(points_data, name="Nuclei")

    from napari_nuclephaser import calibrate_points as cp_module

    def mock_prediction_side_effect(*args, **kwargs):
        return mock_prediction_with_confidences([0.9, 0.8, 0.7])

    with patch.object(
        cp_module,
        "get_sliced_prediction",
        side_effect=mock_prediction_side_effect,
    ):
        widget = calibrate_with_points()
        result = widget(
            Select_Phase_stack=viewer.layers["Phase"],
            Select_Points_layer=points_layer,
            viewer=viewer,
            Division_size=100,
            Calibration_proportion=1.0,
            Save_folder=tmp_path,
        )
    # Returns string even without test tiles (no MAPE)
    assert result is not None
    assert (fake_subfolder / "reference_points.csv").exists()
    assert (fake_subfolder / "metadata.txt").exists()
