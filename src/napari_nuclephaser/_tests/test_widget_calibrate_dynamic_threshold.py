import json
import pathlib
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import napari_nuclephaser.calibrate_with_dynamic_threshold as cwdt


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(cwdt, "CONFIG_PATH", tmp_path / "config.json")


@pytest.fixture(autouse=True)
def silence_modals(monkeypatch):
    monkeypatch.setattr(cwdt, "show_modal_error", MagicMock(), raising=False)
    monkeypatch.setattr(cwdt, "show_info", MagicMock(), raising=False)


@pytest.fixture
def viewer(make_napari_viewer):
    return make_napari_viewer()


class TestConfig:
    def test_default_when_missing(self):
        assert cwdt._load_last_folder() == pathlib.Path(".")

    def test_roundtrip(self, tmp_path):
        target = tmp_path / "my_experiment"
        cwdt._save_last_folder(target)
        assert cwdt._load_last_folder() == target

    def test_malformed_json(self, tmp_path):
        (tmp_path / "config.json").write_text("not json at all")
        assert cwdt._load_last_folder() == pathlib.Path(".")

    def test_non_dict_json(self, tmp_path):
        (tmp_path / "config.json").write_text("[1, 2, 3]")
        assert cwdt._load_last_folder() == pathlib.Path(".")

    def test_empty_string(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({"last_folder": ""}))
        assert cwdt._load_last_folder() == pathlib.Path(".")


class TestEnsureNumpy:
    def test_numpy_passthrough(self):
        arr = np.zeros((5, 5))
        assert cwdt._ensure_numpy(arr) is arr

    def test_dask_conversion(self):
        import dask.array as da

        dask_arr = da.from_array(np.ones((10, 10)), chunks=(5, 5))
        result = cwdt._ensure_numpy(dask_arr)
        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 10)


class TestSavePointsCsv:
    def test_2d_points_format(self, tmp_path):
        pts = np.array([[10.0, 20.0], [30.0, 40.0]])
        out = tmp_path / "ref.csv"
        cwdt._save_points_csv(pts, str(out))
        df = pd.read_csv(out)
        assert list(df.columns) == ["index", "axis-0", "axis-1"]
        assert df["axis-0"].tolist() == [10.0, 30.0]
        assert df["axis-1"].tolist() == [20.0, 40.0]

    def test_3d_points_format(self, tmp_path):
        pts = np.array([[0.0, 10.0, 20.0], [1.0, 30.0, 40.0]])
        out = tmp_path / "ref.csv"
        cwdt._save_points_csv(pts, str(out))
        df = pd.read_csv(out)
        assert list(df.columns) == ["index", "axis-0", "axis-1", "axis-2"]

    def test_empty_points(self, tmp_path):
        out = tmp_path / "ref.csv"
        cwdt._save_points_csv(np.zeros((0, 2)), str(out))
        assert out.exists()


class TestSplitImageAndPoints:
    def test_tiles_and_counts(self):
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        # one point at the centre of each 20x20 tile
        points = [(10.0, 10.0), (10.0, 30.0), (30.0, 10.0), (30.0, 30.0)]
        tiles, counts = cwdt._split_image_and_points(image, points, 20)
        assert len(tiles) == 4
        assert counts == [1, 1, 1, 1]

    def test_empty_points(self):
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        tiles, counts = cwdt._split_image_and_points(image, [], 20)
        assert len(tiles) == 4
        assert counts == [0, 0, 0, 0]

    def test_point_outside_bounds(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        tiles, counts = cwdt._split_image_and_points(
            image, [(100.0, 100.0)], 20
        )
        assert counts == [0]

    def test_multiple_points_same_tile(self):
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        points = [(5.0, 5.0), (6.0, 6.0), (7.0, 7.0)]
        tiles, counts = cwdt._split_image_and_points(image, points, 20)
        assert counts[0] == 3
        assert sum(counts[1:]) == 0


class TestBlurImage:
    def test_sigma_zero_is_identity(self):
        img = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        result = cwdt.blur_image(img, sigma=0)
        assert np.array_equal(result, img)

    def test_sigma_positive_changes_image(self):
        img = np.random.randint(0, 255, (20, 20), dtype=np.uint8)
        result = cwdt.blur_image(img, sigma=2)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_rgb_image(self):
        img = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
        result = cwdt.blur_image(img, sigma=2)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_rejects_non_array(self):
        with pytest.raises(TypeError):
            cwdt.blur_image([[1, 2], [3, 4]])

    def test_rejects_non_uint8(self):
        img = np.zeros((10, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="uint8"):
            cwdt.blur_image(img)

    def test_rejects_1d(self):
        img = np.zeros((10,), dtype=np.uint8)
        with pytest.raises(ValueError, match="2D or 3D"):
            cwdt.blur_image(img)


class TestExtractFeaturesGrayscale:
    def test_returns_all_expected_keys(self):
        region = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        features = cwdt.extract_features_grayscale(region)
        expected = {
            "height/width",
            "mean",
            "std",
            "median",
            "p25",
            "p75",
            "iqr",
            "min",
            "max",
            "range",
            "rms_contrast",
            "lap_var",
            "entropy",
            "energy",
        }
        assert set(features.keys()) == expected

    def test_constant_region(self):
        region = np.full((10, 10), 100, dtype=np.uint8)
        features = cwdt.extract_features_grayscale(region)
        assert features["mean"] == 100.0
        assert features["std"] == 0.0
        assert features["range"] == 0.0

    def test_aspect_ratio(self):
        region = np.zeros((10, 20), dtype=np.uint8)
        features = cwdt.extract_features_grayscale(region)
        assert features["height/width"] == 0.5


class TestComputeTileFeatures:
    def test_without_detections(self):
        tile = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        features = cwdt.compute_tile_features(tile, [])
        assert features["density_0.01"] == 0.0
        assert features["top10_area"] == 0.0

    def test_with_detections(self):
        tile = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        detections = [(1, 3, 1, 3, 0.9), (5, 8, 5, 8, 0.5)]
        features = cwdt.compute_tile_features(tile, detections)
        assert features["density_0.01"] > 0
        assert features["density_0.30"] > 0
        assert features["top10_area"] > 0

    def test_density_ordering(self):
        tile = np.zeros((10, 10), dtype=np.uint8)
        detections = [(1, 2, 1, 2, 0.9), (3, 4, 3, 4, 0.5)]
        features = cwdt.compute_tile_features(tile, detections)
        # higher threshold -> fewer or equal detections
        assert features["density_0.01"] >= features["density_0.10"]
        assert features["density_0.10"] >= features["density_0.20"]

    def test_top10_area_single_detection(self):
        tile = np.zeros((10, 10), dtype=np.uint8)
        # bbox from (1,1) to (3,3): area = 4
        detections = [(1, 3, 1, 3, 0.9)]
        features = cwdt.compute_tile_features(tile, detections)
        assert features["top10_area"] == 4.0


class TestFindOptimalThreshold:
    def test_empty_returns_none(self):
        assert cwdt.find_optimal_threshold([], 5) is None

    def test_single_detection_matches_gt(self):
        # conf=0.5, gt=1: best thr is any <= 0.5, algorithm picks 0.01
        detections = [(0, 1, 0, 1, 0.5)]
        assert cwdt.find_optimal_threshold(detections, 1) == 0.01

    def test_two_detections_matches_two(self):
        detections = [(0, 1, 0, 1, 0.9), (2, 3, 2, 3, 0.5)]
        result = cwdt.find_optimal_threshold(detections, 2)
        # first thr that keeps both: anything <= 0.5 -> picks 0.01
        assert result == 0.01

    def test_threshold_above_lower_conf(self):
        # gt=1 with detections at 0.9 and 0.5 -> need thr in (0.5, 0.9]
        detections = [(0, 1, 0, 1, 0.9), (2, 3, 2, 3, 0.5)]
        result = cwdt.find_optimal_threshold(detections, 1)
        # algorithm picks the first thr with minimum error
        assert 0.5 < result <= 0.9

    def test_gt_zero_with_detections(self):
        detections = [(0, 1, 0, 1, 0.5)]
        result = cwdt.find_optimal_threshold(detections, 0)
        # need to reject all detections: thr > 0.5
        assert result > 0.5

    def test_all_detections_above_thr(self):
        detections = [(0, 1, 0, 1, 0.95)] * 3
        assert cwdt.find_optimal_threshold(detections, 3) == 0.01


class TestFindStaticThreshold:
    def test_empty_returns_zero(self):
        thr, err = cwdt.find_static_threshold([], [])
        assert thr == 0.0
        assert err == float("inf")

    def test_single_tile_perfect_match(self):
        confs = [[0.5]]
        gts = [1]
        thr, err = cwdt.find_static_threshold(confs, gts)
        assert err == 0
        assert thr <= 0.5

    def test_two_tiles(self):
        confs = [[0.9, 0.5], [0.8]]
        gts = [2, 1]
        thr, err = cwdt.find_static_threshold(confs, gts)
        assert err == 0

    def test_picks_best_threshold(self):
        # tile A: 2 detections, gt=1; tile B: 1 detection, gt=1
        confs = [[0.9, 0.5], [0.7]]
        gts = [1, 1]
        thr, _ = cwdt.find_static_threshold(confs, gts)
        # need thr > 0.5 to reject the second detection in tile A
        assert thr > 0.5


class TestBuildThresholdMap:
    def test_returns_window_centres(self):
        tile = np.random.randint(0, 255, (20, 20), dtype=np.uint8)
        detections = [(2, 4, 2, 4, 0.9), (10, 12, 10, 12, 0.6)]
        regressor = MagicMock()
        regressor.predict.return_value = np.array([0.5] * 9)
        scaler = MagicMock()
        scaler.transform.side_effect = lambda x: x
        feature_names = sorted(
            cwdt.compute_tile_features(
                np.zeros((10, 10), dtype=np.uint8), []
            ).keys()
        )
        result = cwdt.build_threshold_map(
            tile, detections, regressor, scaler, feature_names, 10, 5
        )
        # 20x20 tile, win_size=10, stride=5 -> 3x3 windows = 9
        assert len(result) == 9
        for cx, cy, thr in result:
            assert isinstance(cx, float)
            assert isinstance(cy, float)
            assert isinstance(thr, float)

    def test_no_detections_still_returns_windows(self):
        tile = np.zeros((20, 20), dtype=np.uint8)
        regressor = MagicMock()
        regressor.predict.return_value = np.array([0.5] * 9)
        scaler = MagicMock()
        scaler.transform.side_effect = lambda x: x
        feature_names = sorted(
            cwdt.compute_tile_features(
                np.zeros((10, 10), dtype=np.uint8), []
            ).keys()
        )
        result = cwdt.build_threshold_map(
            tile, [], regressor, scaler, feature_names, 10, 5
        )
        assert len(result) == 9

    def test_win_size_larger_than_tile(self):
        # win_size > tile -> guard clamps
        tile = np.zeros((5, 5), dtype=np.uint8)
        regressor = MagicMock()
        regressor.predict.return_value = np.array([0.5])
        scaler = MagicMock()
        scaler.transform.side_effect = lambda x: x
        feature_names = sorted(
            cwdt.compute_tile_features(
                np.zeros((10, 10), dtype=np.uint8), []
            ).keys()
        )
        result = cwdt.build_threshold_map(
            tile, [], regressor, scaler, feature_names, 100, 50
        )
        assert len(result) == 1

    def test_calls_scaler_and_regressor(self):
        tile = np.zeros((10, 10), dtype=np.uint8)
        regressor = MagicMock()
        regressor.predict.return_value = np.array([0.5])
        scaler = MagicMock()
        scaler.transform.side_effect = lambda x: x
        feature_names = sorted(
            cwdt.compute_tile_features(
                np.zeros((10, 10), dtype=np.uint8), []
            ).keys()
        )
        cwdt.build_threshold_map(
            tile, [], regressor, scaler, feature_names, 10, 10
        )
        scaler.transform.assert_called_once()
        regressor.predict.assert_called_once()


def _make_widget_call(viewer, image, points, tmp_path, **kwargs):
    widget = cwdt.calibrate_with_dynamic_threshold()
    return widget(
        Select_Phase_stack=image,
        Select_Points_layer=points,
        viewer=viewer,
        Save_folder=tmp_path,
        **kwargs,
    )


class TestWidgetValidation:
    def test_empty_points(self, viewer, tmp_path):
        img = viewer.add_image(np.zeros((50, 50), dtype=np.uint8), name="img")
        pts = viewer.add_points(np.empty((0, 2)), name="pts")
        widget = cwdt.calibrate_with_dynamic_threshold()
        with patch.object(cwdt, "show_modal_error") as err:
            result = widget(
                Select_Phase_stack=img,
                Select_Points_layer=pts,
                viewer=viewer,
                Save_folder=tmp_path,
            )
        err.assert_called_once()
        assert "Points layer is empty" in err.call_args[0][0]
        assert result is None

    def test_reject_2stack(self, viewer, tmp_path):
        img = viewer.add_image(
            np.zeros((2, 3, 20, 20), dtype=np.uint8), name="img"
        )
        pts = viewer.add_points(np.array([[5.0, 5.0]]), name="pts")
        widget = cwdt.calibrate_with_dynamic_threshold()
        with patch.object(cwdt, "show_modal_error") as err:
            result = widget(
                Select_Phase_stack=img,
                Select_Points_layer=pts,
                viewer=viewer,
                Save_folder=tmp_path,
            )
        err.assert_called_once()
        assert "2-dimensional stacks are not supported" in err.call_args[0][0]
        assert result is None

    def test_reject_2stack_with_channel(self, viewer, tmp_path):
        img = viewer.add_image(
            np.zeros((2, 3, 20, 20, 3), dtype=np.uint8), name="img"
        )
        pts = viewer.add_points(np.array([[5.0, 5.0]]), name="pts")
        widget = cwdt.calibrate_with_dynamic_threshold()
        with patch.object(cwdt, "show_modal_error") as err:
            result = widget(
                Select_Phase_stack=img,
                Select_Points_layer=pts,
                viewer=viewer,
                Save_folder=tmp_path,
            )
        err.assert_called_once()
        assert "2-dimensional stacks are not supported" in err.call_args[0][0]
        assert result is None

    def test_2d_points_with_1stack(self, viewer, tmp_path):
        img = viewer.add_image(
            np.zeros((3, 50, 50), dtype=np.uint8), name="img"
        )
        pts = viewer.add_points(np.array([[5.0, 5.0]]), name="pts")
        widget = cwdt.calibrate_with_dynamic_threshold()
        with patch.object(cwdt, "show_modal_error") as err:
            result = widget(
                Select_Phase_stack=img,
                Select_Points_layer=pts,
                viewer=viewer,
                Save_folder=tmp_path,
            )
        err.assert_called_once()
        assert "Points have 2 columns" in err.call_args[0][0]
        assert result is None

    def test_wrong_column_count(self, viewer, tmp_path):
        img = viewer.add_image(np.zeros((50, 50), dtype=np.uint8), name="img")
        pts = viewer.add_points(np.array([[1.0, 2.0, 3.0, 4.0]]), name="pts")
        widget = cwdt.calibrate_with_dynamic_threshold()
        with patch.object(cwdt, "show_modal_error") as err:
            result = widget(
                Select_Phase_stack=img,
                Select_Points_layer=pts,
                viewer=viewer,
                Save_folder=tmp_path,
            )
        err.assert_called_once()
        assert "columns" in err.call_args[0][0]
        assert result is None

    def test_missing_save_folder(self, viewer):
        img = viewer.add_image(np.zeros((50, 50), dtype=np.uint8), name="img")
        pts = viewer.add_points(np.array([[5.0, 5.0]]), name="pts")
        widget = cwdt.calibrate_with_dynamic_threshold()
        with patch.object(cwdt, "show_modal_error") as err:
            result = widget(
                Select_Phase_stack=img,
                Select_Points_layer=pts,
                viewer=viewer,
                Save_folder="",
            )
        err.assert_called_once()
        assert "save folder" in err.call_args[0][0].lower()
        assert result is None

    def test_no_calibration_tiles(self, viewer, tmp_path):
        # points in a frame that has no image
        img = viewer.add_image(np.zeros((50, 50), dtype=np.uint8), name="img")
        pts = viewer.add_points(np.array([[5.0, 5.0]]), name="pts")
        widget = cwdt.calibrate_with_dynamic_threshold()
        with (
            patch.object(cwdt, "initialize_model") as init,
            patch.object(cwdt, "show_modal_error") as err,
        ):
            init.return_value = (MagicMock(), "mock")
            result = widget(
                Select_Phase_stack=img,
                Select_Points_layer=pts,
                viewer=viewer,
                Save_folder=tmp_path,
                Division_size=1000,  # tile larger than image
            )
        err.assert_called_once()
        assert result is None


def _make_detection(x1, y1, x2, y2, conf):
    det = MagicMock()
    det.bbox.minx = x1
    det.bbox.maxx = x2
    det.bbox.miny = y1
    det.bbox.maxy = y2
    det.score.value = conf
    return det


def _make_prediction_result(n_detections=1):
    result = MagicMock()
    result.object_prediction_list = [
        _make_detection(1, 1, 3, 3, 0.5) for _ in range(n_detections)
    ]
    return result


class TestWidgetSuccess:
    def test_minimal_pipeline(self, viewer, tmp_path, mocker):
        """End-to-end smoke test with mocked model, predictions,
        grid search, and scaler."""
        # 20x20 image, Division_size=10 -> 4 tiles
        # points at the centre of each tile -> gt_count=1 in all tiles
        img = viewer.add_image(
            np.random.randint(0, 255, (20, 20), dtype=np.uint8),
            name="img",
        )
        pts = viewer.add_points(
            np.array(
                [
                    [5.0, 5.0],
                    [5.0, 15.0],
                    [15.0, 5.0],
                    [15.0, 15.0],
                ]
            ),
            name="pts",
        )

        mocker.patch.object(
            cwdt,
            "initialize_model",
            return_value=(MagicMock(), "mock_model"),
        )
        mocker.patch.object(
            cwdt,
            "get_sliced_prediction",
            return_value=_make_prediction_result(1),
        )

        # Mock GridSearchCV so we skip actual fitting
        mock_regressor = MagicMock()
        mock_regressor.predict.return_value = np.array([0.5])
        mock_grid_search = MagicMock()
        mock_grid_search.best_params_ = {"n_neighbors": 3}
        mock_grid_search.best_estimator_ = mock_regressor
        mocker.patch.object(
            cwdt, "GridSearchCV", return_value=mock_grid_search
        )

        # Mock StandardScaler as identity
        mock_scaler = MagicMock()
        mock_scaler.transform.side_effect = lambda x: x
        mock_scaler.fit_transform.side_effect = lambda x: x
        mocker.patch.object(cwdt, "StandardScaler", return_value=mock_scaler)
        mocker.patch.object(cwdt.pickle, "dump")

        fake_subfolder = tmp_path / "fake_run"
        fake_subfolder.mkdir()
        mocker.patch.object(
            cwdt,
            "create_unique_subfolder",
            return_value=str(fake_subfolder),
        )

        widget = cwdt.calibrate_with_dynamic_threshold()
        result = widget(
            Select_Phase_stack=img,
            Select_Points_layer=pts,
            viewer=viewer,
            Save_folder=tmp_path,
            Experiment_name="test_run",
            Division_size=10,
            Calibration_proportion=0.5,
            Max_blur_strength=0,
            Sahi_size=640,
            Random_seed=42,
        )

        assert result is not None
        assert "Calibration completed" in result
        assert "Best k" in result

        dynamic_dir = fake_subfolder / "Dynamic Confidence Threshold"
        assert dynamic_dir.is_dir()
        assert (dynamic_dir / "metadata.txt").exists()
        assert (dynamic_dir / "dynamic_threshold.pkl").exists()
        assert (dynamic_dir / "reference_points.csv").exists()
        assert (dynamic_dir / "Error_plot_for_blur_sigma_0.png").exists()

    def test_reference_points_csv_format(self, viewer, tmp_path, mocker):
        img = viewer.add_image(
            np.random.randint(0, 255, (20, 20), dtype=np.uint8),
            name="img",
        )
        pts = viewer.add_points(
            np.array([[5.0, 5.0], [15.0, 15.0]]), name="pts"
        )

        mocker.patch.object(
            cwdt,
            "initialize_model",
            return_value=(MagicMock(), "mock_model"),
        )
        mocker.patch.object(
            cwdt,
            "get_sliced_prediction",
            return_value=_make_prediction_result(1),
        )

        mock_regressor = MagicMock()
        mock_regressor.predict.return_value = np.array([0.5])
        mock_grid_search = MagicMock()
        mock_grid_search.best_params_ = {"n_neighbors": 3}
        mock_grid_search.best_estimator_ = mock_regressor
        mocker.patch.object(
            cwdt, "GridSearchCV", return_value=mock_grid_search
        )

        mock_scaler = MagicMock()
        mock_scaler.transform.side_effect = lambda x: x
        mock_scaler.fit_transform.side_effect = lambda x: x
        mocker.patch.object(cwdt, "StandardScaler", return_value=mock_scaler)
        mocker.patch.object(cwdt.pickle, "dump")

        fake_subfolder = tmp_path / "fake_run"
        fake_subfolder.mkdir()
        mocker.patch.object(
            cwdt,
            "create_unique_subfolder",
            return_value=str(fake_subfolder),
        )

        widget = cwdt.calibrate_with_dynamic_threshold()
        widget(
            Select_Phase_stack=img,
            Select_Points_layer=pts,
            viewer=viewer,
            Save_folder=tmp_path,
            Experiment_name="test_run",
            Division_size=10,
            Calibration_proportion=0.5,
            Max_blur_strength=0,
            Sahi_size=640,
        )

        csv_path = (
            fake_subfolder
            / "Dynamic Confidence Threshold"
            / "reference_points.csv"
        )
        df = pd.read_csv(csv_path)
        assert list(df.columns) == ["index", "axis-0", "axis-1"]
        assert len(df) == 2

    def test_config_persisted_on_success(self, viewer, tmp_path, mocker):
        img = viewer.add_image(
            np.random.randint(0, 255, (20, 20), dtype=np.uint8),
            name="img",
        )
        pts = viewer.add_points(
            np.array([[5.0, 5.0], [15.0, 15.0]]), name="pts"
        )

        mocker.patch.object(
            cwdt,
            "initialize_model",
            return_value=(MagicMock(), "mock_model"),
        )
        mocker.patch.object(
            cwdt,
            "get_sliced_prediction",
            return_value=_make_prediction_result(1),
        )

        mock_regressor = MagicMock()
        mock_regressor.predict.return_value = np.array([0.5])
        mock_grid_search = MagicMock()
        mock_grid_search.best_params_ = {"n_neighbors": 3}
        mock_grid_search.best_estimator_ = mock_regressor
        mocker.patch.object(
            cwdt, "GridSearchCV", return_value=mock_grid_search
        )

        mock_scaler = MagicMock()
        mock_scaler.transform.side_effect = lambda x: x
        mock_scaler.fit_transform.side_effect = lambda x: x
        mocker.patch.object(cwdt, "StandardScaler", return_value=mock_scaler)
        mocker.patch.object(cwdt.pickle, "dump")

        fake_subfolder = tmp_path / "fake_run"
        fake_subfolder.mkdir()
        mocker.patch.object(
            cwdt,
            "create_unique_subfolder",
            return_value=str(fake_subfolder),
        )

        widget = cwdt.calibrate_with_dynamic_threshold()
        widget(
            Select_Phase_stack=img,
            Select_Points_layer=pts,
            viewer=viewer,
            Save_folder=tmp_path,
            Experiment_name="test_run",
            Division_size=10,
            Calibration_proportion=0.5,
            Max_blur_strength=0,
        )

        assert cwdt._load_last_folder() == tmp_path

    def test_no_training_samples(self, viewer, tmp_path, mocker):
        """All detections empty -> samples_X empty -> error raised."""
        img = viewer.add_image(
            np.random.randint(0, 255, (20, 20), dtype=np.uint8),
            name="img",
        )
        pts = viewer.add_points(
            np.array([[5.0, 5.0], [15.0, 15.0]]), name="pts"
        )

        mocker.patch.object(
            cwdt,
            "initialize_model",
            return_value=(MagicMock(), "mock_model"),
        )
        empty_result = MagicMock()
        empty_result.object_prediction_list = []
        mocker.patch.object(
            cwdt, "get_sliced_prediction", return_value=empty_result
        )

        widget = cwdt.calibrate_with_dynamic_threshold()
        with patch.object(cwdt, "show_modal_error") as err:
            result = widget(
                Select_Phase_stack=img,
                Select_Points_layer=pts,
                viewer=viewer,
                Save_folder=tmp_path,
                Division_size=10,
                Calibration_proportion=0.5,
                Max_blur_strength=0,
            )
        err.assert_called_once()
        assert "No training samples" in err.call_args[0][0]
        assert result is None
