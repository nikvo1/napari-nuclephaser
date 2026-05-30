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
# Tests for _ensure_numpy (Dask conversion)
# ----------------------------------------------------------------------
def test_ensure_numpy_with_dask():
    """Dask array should be converted to numpy."""
    dask_arr = da.from_array(np.random.rand(10, 10), chunks=(5, 5))
    result = _ensure_numpy(dask_arr)
    assert isinstance(result, np.ndarray)
    assert result.shape == (10, 10)


def test_ensure_numpy_with_numpy():
    """Numpy array should be returned unchanged."""
    numpy_arr = np.random.rand(10, 10)
    result = _ensure_numpy(numpy_arr)
    assert result is numpy_arr


# ----------------------------------------------------------------------
# Tests for tile splitting (_split_image_and_points)
# ----------------------------------------------------------------------
def test_split_image_and_points():
    # Create a 3-channel image 400x400
    image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    points = [(50, 50), (150, 150), (250, 250), (350, 350)]  # (y, x)
    window_size = 100

    tiles, counts = _split_image_and_points(image, points, window_size)
    # Expect 4x4 = 16 tiles
    assert len(tiles) == 16
    assert len(counts) == 16
    # Points should be in tiles (0,0), (1,1), (2,2), (3,3) -> each count 1
    # Tile indices: row-major, tile (row, col) at index row*4+col
    assert counts[0] == 1  # (0,0)
    assert counts[5] == 1  # (1,1) -> row1 col1 = 1*4+1=5
    assert counts[10] == 1  # (2,2) -> 2*4+2=10
    assert counts[15] == 1  # (3,3) -> 3*4+3=15


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
    points = [(10, 10), (110, 110)]  # one in first tile, one in second tile
    division_size = 100
    calib_prop = 0.5
    calib_tiles, calib_gt, test_tiles, test_gt = _prepare_frame(
        image, points, division_size, calib_prop, random_seed=42
    )
    total_tiles = 4  # 2x2
    n_calib = int(total_tiles * calib_prop)  # 2
    assert len(calib_tiles) == n_calib
    assert len(calib_gt) == n_calib
    assert len(test_tiles) == total_tiles - n_calib
    assert len(test_gt) == total_tiles - n_calib
    # Check that ground truth counts are 0 or 1
    assert all(gt in (0, 1) for gt in calib_gt + test_gt)


def test_prepare_frame_grayscale_conversion():
    # 2D grayscale uint16 -> should convert to RGB uint8
    image = np.random.randint(0, 65535, (200, 200), dtype=np.uint16)
    points = []
    calib_tiles, _, _, _ = _prepare_frame(image, points, 100, 0.5, 42)
    # Each tile should be RGB (3 channels)
    for tile in calib_tiles:
        assert tile.shape[-1] == 3
        assert tile.dtype == np.uint8


# ----------------------------------------------------------------------
# Tests for error handling (image/points shape mismatches)
# ----------------------------------------------------------------------
def test_calibrate_error_3d_image_with_2d_points(make_napari_viewer):
    """3D image stack (T, H, W) with points having only (y,x) -> error."""
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
    """Points with 4 columns -> error."""
    viewer = make_napari_viewer()
    image_2d = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    points_bad = np.array([[1, 2, 3, 4]])  # 4 columns
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
    """Empty points layer -> error."""
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
    """4D image without trailing color channel -> error."""
    viewer = make_napari_viewer()
    # shape (2, 2, 100, 100) -> 4D but last dim not 1,3,4
    image_4d = np.random.randint(0, 255, (2, 2, 100, 100), dtype=np.uint8)
    points = np.array([[0, 50, 50], [0, 60, 60]])  # (frame, y, x)
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
# Successful calibration test using real model (YOLOv11) but mocked predictions
# ----------------------------------------------------------------------
def test_calibrate_success_with_real_model_and_mocked_predictions(
    make_napari_viewer, real_yolov11_model, tmp_path
):
    """Test full calibration pipeline with real model but controlled detection confidences."""
    model, model_type = real_yolov11_model
    viewer = make_napari_viewer()

    # Small 2D image (200x200)
    image_data = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    # Create points that will fall into tiles
    points_data = np.array([[50, 50], [150, 150]])

    viewer.add_image(image_data, name="Phase")
    points_layer = viewer.add_points(points_data, name="Nuclei")

    # Patch get_sliced_prediction to return known confidences
    # For calibration: we need enough detections to find a threshold.
    # We'll create two phases: during _find_best_threshold_for_frame we return varying confidences.
    # But simpler: we can mock at a higher level to just test the pipeline.
    # However, we must keep the real model initialization. We'll mock get_sliced_prediction globally.
    from napari_nuclephaser import calibrate_points as cp_module

    # We'll create a side effect that returns different confidences based on call arguments
    def mock_prediction_side_effect(*args, **kwargs):
        # Return a result with a fixed list of confidences for calibration step
        # We need to differentiate calibration vs test? The test uses the same function.
        # Just return confidences that lead to a threshold around 0.5.
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
            Experiment_name="test_calib",
            Random_seed=42,
        )

    # Check return string
    assert result is not None
    assert "Best threshold" in result
    assert "MAPE" in result

    # Check saved files
    subfolders = list(tmp_path.glob("test_calib_*"))
    assert len(subfolders) == 1
    subfolder = subfolders[0]

    plot_file = subfolder / "Calibration error plot.png"
    assert plot_file.exists()
    meta_file = subfolder / "metadata.txt"
    assert meta_file.exists()
    points_file = subfolder / "reference points.csv"
    assert points_file.exists()

    # Check CSV content
    df = pd.read_csv(points_file)
    assert list(df.columns) == ["frame", "y", "x"]
    assert len(df) == 2
    assert df["frame"].iloc[0] == 0

    # Check metadata contains expected keys
    meta_text = meta_file.read_text()
    assert "Overall threshold" in meta_text
    assert "Overall MAPE" in meta_text


# ----------------------------------------------------------------------
# Test with real YOLOv5 model (similar)
# ----------------------------------------------------------------------
def test_calibrate_success_with_real_yolov5_model(
    make_napari_viewer, real_yolov5_model, tmp_path
):
    model, model_type = real_yolov5_model
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
            Experiment_name="test_yolov5",
        )
    assert result is not None
    assert "Best threshold" in result


# ----------------------------------------------------------------------
# Test Dask array input (conversion inside calibrate)
# ----------------------------------------------------------------------
def test_calibrate_with_dask_array(
    make_napari_viewer, real_yolov11_model, tmp_path
):
    """Image data as Dask array should be converted to numpy."""
    model, _ = real_yolov11_model
    viewer = make_napari_viewer()
    # Create dask array from numpy
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
            Save_folder=tmp_path,
        )
    assert result is not None
    # Ensure no errors about dask


# ----------------------------------------------------------------------
# Test multi-frame (stack) with 3D points
# ----------------------------------------------------------------------
def test_calibrate_multiframe(
    make_napari_viewer, real_yolov11_model, tmp_path
):
    """Test with 3D image stack (T, H, W) and points (frame, y, x)."""
    model, _ = real_yolov11_model
    viewer = make_napari_viewer()
    # 3 frames, each 100x100
    stack = np.random.randint(0, 255, (3, 100, 100), dtype=np.uint8)
    points = np.array(
        [
            [0, 20, 20],
            [0, 80, 80],
            [1, 30, 30],
            [2, 50, 50],
        ]
    )
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
    # Check saved CSV has frame column
    subfolders = list(tmp_path.glob("Experiment_*"))
    assert len(subfolders) == 1
    points_file = subfolders[0] / "reference_points.csv"
    df = pd.read_csv(points_file)
    assert list(df.columns) == ["frame", "y", "x"]
    assert set(df["frame"]) == {0, 1, 2}


# ----------------------------------------------------------------------
# Edge case: no detections during calibration -> should abort
# ----------------------------------------------------------------------
def test_calibrate_no_detections(
    make_napari_viewer, real_yolov11_model, tmp_path
):
    """If model makes no detections on calibration tiles, return None."""
    model, _ = real_yolov11_model
    viewer = make_napari_viewer()
    image_data = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    points_data = np.array([[50, 50], [150, 150]])
    viewer.add_image(image_data, name="Phase")
    points_layer = viewer.add_points(points_data, name="Nuclei")

    from napari_nuclephaser import calibrate_points as cp_module

    # Return empty detection list
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
                Save_folder=tmp_path,
            )
            mock_error.assert_called_once()
            assert "No valid calibration data" in mock_error.call_args[0][0]
            assert result is None


# ----------------------------------------------------------------------
# Test that saving works even when no test tiles (all tiles used for calibration)
# ----------------------------------------------------------------------
def test_calibrate_all_tiles_calibration(
    make_napari_viewer, real_yolov11_model, tmp_path
):
    """Calibration_proportion = 1.0 -> no test tiles, should still save."""
    model, _ = real_yolov11_model
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
            Calibration_proportion=1.0,
            Save_folder=tmp_path,
        )
    # Result should still contain threshold but no MAPE or MAPE=NaN?
    assert result is not None
    # Check files still created
    subfolders = list(tmp_path.glob("Experiment_*"))
    assert len(subfolders) == 1
    assert (subfolders[0] / "reference_points.csv").exists()
    assert (subfolders[0] / "metadata.txt").exists()
