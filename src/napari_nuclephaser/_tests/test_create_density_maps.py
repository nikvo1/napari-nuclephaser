import json
import pathlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import napari_nuclephaser.create_density_maps as cdm


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(cdm, "CONFIG_PATH", tmp_path / "config.json")


@pytest.fixture(autouse=True)
def silence_modals(monkeypatch):
    """Prevent modal dialogs from blocking the test runner.

    Tests that want to assert on the warning can still layer
    `patch.object(cdm, "show_modal_warning")` on top.
    """
    monkeypatch.setattr(cdm, "show_modal_error", MagicMock())
    monkeypatch.setattr(cdm, "show_modal_warning", MagicMock())


@pytest.fixture
def viewer(make_napari_viewer):
    return make_napari_viewer()


class TestExtractFrameShape:
    def test_single_2d(self):
        assert cdm._extract_frame_and_spatial_shape((100, 200)) == (
            (),
            (100, 200),
        )

    def test_colour_1ch(self):
        assert cdm._extract_frame_and_spatial_shape((100, 200, 1)) == (
            (),
            (100, 200),
        )

    def test_colour_3ch(self):
        assert cdm._extract_frame_and_spatial_shape((100, 200, 3)) == (
            (),
            (100, 200),
        )

    def test_colour_4ch(self):
        assert cdm._extract_frame_and_spatial_shape((100, 200, 4)) == (
            (),
            (100, 200),
        )

    def test_1_stack_grayscale(self):
        assert cdm._extract_frame_and_spatial_shape((10, 100, 200)) == (
            (10,),
            (100, 200),
        )

    def test_1_stack_with_channel(self):
        assert cdm._extract_frame_and_spatial_shape((10, 100, 200, 3)) == (
            (10,),
            (100, 200),
        )

    def test_2_stack(self):
        assert cdm._extract_frame_and_spatial_shape((3, 4, 100, 200)) == (
            (3, 4),
            (100, 200),
        )

    def test_2_stack_with_channel(self):
        assert cdm._extract_frame_and_spatial_shape((3, 4, 100, 200, 3)) == (
            (3, 4),
            (100, 200),
        )

    def test_reject_1d(self):
        with pytest.raises(ValueError):
            cdm._extract_frame_and_spatial_shape((100,))

    def test_reject_ambiguous_5(self):
        with pytest.raises(ValueError, match="Ambiguous"):
            cdm._extract_frame_and_spatial_shape((100, 100, 5))

    def test_reject_ambiguous_2(self):
        with pytest.raises(ValueError, match="Ambiguous"):
            cdm._extract_frame_and_spatial_shape((100, 100, 2))

    def test_reject_ambiguous_7(self):
        with pytest.raises(ValueError, match="Ambiguous"):
            cdm._extract_frame_and_spatial_shape((100, 100, 7))

    def test_accept_width_8_as_1_stack(self):
        assert cdm._extract_frame_and_spatial_shape((100, 100, 8)) == (
            (100,),
            (100, 8),
        )

    def test_reject_5d_non_channel(self):
        with pytest.raises(ValueError):
            cdm._extract_frame_and_spatial_shape((2, 3, 4, 5, 6))


class TestComputeMapShape:
    def test_square(self):
        assert cdm._compute_map_shape(100, 100, 50) == (50, 50)

    def test_wide(self):
        assert cdm._compute_map_shape(50, 100, 50) == (25, 50)

    def test_tall(self):
        assert cdm._compute_map_shape(100, 50, 50) == (50, 25)

    def test_extreme_aspect(self):
        assert cdm._compute_map_shape(1000, 10, 50) == (50, 1)

    def test_density_size_one(self):
        assert cdm._compute_map_shape(100, 100, 1) == (1, 1)

    def test_round_not_truncate(self):
        assert cdm._compute_map_shape(100, 70, 50)[1] == round(70 * 0.5)


class TestIterPoints:
    def test_2d_points(self):
        pts = np.array([[10.0, 20.0], [30.0, 40.0]])
        assert list(cdm._iter_points(pts)) == [
            ((), 10.0, 20.0),
            ((), 30.0, 40.0),
        ]

    def test_3d_points(self):
        pts = np.array([[1.0, 10.0, 20.0]])
        assert list(cdm._iter_points(pts)) == [((1,), 10.0, 20.0)]

    def test_4d_points(self):
        pts = np.array([[1.0, 2.0, 10.0, 20.0]])
        assert list(cdm._iter_points(pts)) == [((1, 2), 10.0, 20.0)]

    def test_wrong_columns(self):
        pts = np.zeros((3, 5))
        with pytest.raises(ValueError, match="Unsupported Points"):
            list(cdm._iter_points(pts))

    def test_empty(self):
        assert list(cdm._iter_points(np.zeros((0, 2)))) == []


class TestIterShapes:
    def test_2d_rectangle(self):
        # rectangle spanning y in [10, 30], x in [20, 40]
        rect = np.array(
            [[10.0, 20.0], [10.0, 40.0], [30.0, 40.0], [30.0, 20.0]]
        )
        result = list(cdm._iter_shapes([rect]))
        assert result == [((), 20.0, 30.0)]

    def test_3d_rectangle(self):
        rect = np.array(
            [
                [5.0, 10.0, 20.0],
                [5.0, 10.0, 40.0],
                [5.0, 30.0, 40.0],
                [5.0, 30.0, 20.0],
            ]
        )
        result = list(cdm._iter_shapes([rect]))
        assert result == [((5,), 20.0, 30.0)]

    def test_4d_rectangle(self):
        rect = np.array(
            [
                [1.0, 2.0, 10.0, 20.0],
                [1.0, 2.0, 10.0, 40.0],
                [1.0, 2.0, 30.0, 40.0],
                [1.0, 2.0, 30.0, 20.0],
            ]
        )
        result = list(cdm._iter_shapes([rect]))
        assert result == [((1, 2), 20.0, 30.0)]

    def test_empty_list(self):
        assert list(cdm._iter_shapes([])) == []


class TestFindNextIndex:
    def test_empty_folder(self, tmp_path):
        assert cdm._find_next_index(str(tmp_path), 1) == 1

    def test_nonexistent_folder(self, tmp_path):
        assert cdm._find_next_index(str(tmp_path / "nope"), 1) == 1

    def test_next_free(self, tmp_path):
        (tmp_path / "01.npy").touch()
        assert cdm._find_next_index(str(tmp_path), 1) == 2

    def test_multiple_used(self, tmp_path):
        for i in (1, 2, 3):
            (tmp_path / f"{i:02d}.npy").touch()
        assert cdm._find_next_index(str(tmp_path), 1) == 4

    def test_gap_filling(self, tmp_path):
        (tmp_path / "01.npy").touch()
        (tmp_path / "03.npy").touch()
        assert cdm._find_next_index(str(tmp_path), 1) == 2

    def test_non_npy_ignored(self, tmp_path):
        (tmp_path / "01.txt").touch()
        (tmp_path / "01.tif").touch()
        assert cdm._find_next_index(str(tmp_path), 1) == 1

    def test_non_numeric_ignored(self, tmp_path):
        (tmp_path / "abc.npy").touch()
        assert cdm._find_next_index(str(tmp_path), 1) == 1

    def test_single_digit_prefix_matches(self, tmp_path):
        (tmp_path / "1.npy").touch()
        assert cdm._find_next_index(str(tmp_path), 1) == 2

    def test_single_digit_with_frame_suffix(self, tmp_path):
        (tmp_path / "1_05.npy").touch()
        assert cdm._find_next_index(str(tmp_path), 1) == 2

    def test_three_digit_prefix_ignored(self, tmp_path):
        (tmp_path / "100.npy").touch()
        assert cdm._find_next_index(str(tmp_path), 1) == 1

    def test_2stack_style_prefix(self, tmp_path):
        (tmp_path / "02_03.npy").touch()
        # leading two digits count as index 2
        assert cdm._find_next_index(str(tmp_path), 1) == 1
        assert cdm._find_next_index(str(tmp_path), 2) == 3

    def test_all_used_raises(self, tmp_path):
        for i in range(1, 100):
            (tmp_path / f"{i:02d}.npy").touch()
        with pytest.raises(RuntimeError, match="All indices 1-99"):
            cdm._find_next_index(str(tmp_path), 1)


class TestFolderHasDensityMaps:
    def test_missing(self, tmp_path):
        assert cdm._folder_has_density_maps(str(tmp_path / "nope")) is False

    def test_empty(self, tmp_path):
        assert cdm._folder_has_density_maps(str(tmp_path)) is False

    def test_non_npy_only(self, tmp_path):
        (tmp_path / "a.txt").touch()
        assert cdm._folder_has_density_maps(str(tmp_path)) is False

    def test_with_npy(self, tmp_path):
        (tmp_path / "01.npy").touch()
        assert cdm._folder_has_density_maps(str(tmp_path)) is True


class TestDensityFilename:
    def test_single_image(self):
        assert cdm._density_filename(1, ()) == "01.npy"

    def test_1stack(self):
        assert cdm._density_filename(1, (7,)) == "01_07.npy"

    def test_2stack_no_prefix(self):
        assert cdm._density_filename(1, (7, 12)) == "07_12.npy"

    def test_padding(self):
        assert cdm._density_filename(5, (3, 100)) == "03_100.npy"


class TestCalibrationFile:
    def test_read_missing(self, tmp_path):
        assert cdm._read_calibration(str(tmp_path)) == (None, None, None)

    def test_read_full(self, tmp_path):
        (tmp_path / "calibration.txt").write_text(
            "pixel_size_um=0.5\nimage_size=1024x2048\n"
        )
        assert cdm._read_calibration(str(tmp_path)) == (0.5, 1024, 2048)

    def test_read_missing_image_size(self, tmp_path):
        (tmp_path / "calibration.txt").write_text("pixel_size_um=0.5\n")
        assert cdm._read_calibration(str(tmp_path)) == (0.5, None, None)

    def test_read_missing_pixel(self, tmp_path):
        (tmp_path / "calibration.txt").write_text("image_size=100x200\n")
        assert cdm._read_calibration(str(tmp_path)) == (None, 100, 200)

    def test_read_garbage(self, tmp_path):
        (tmp_path / "calibration.txt").write_text("not a valid file\n")
        assert cdm._read_calibration(str(tmp_path)) == (None, None, None)

    def test_read_negative_pixel_treated_as_none(self, tmp_path):
        # parser uses a regex that only matches non-negative numbers,
        # so a negative value simply doesn't match
        (tmp_path / "calibration.txt").write_text(
            "pixel_size_um=-0.5\nimage_size=100x200\n"
        )
        pixel, h, w = cdm._read_calibration(str(tmp_path))
        assert pixel is None

    def test_roundtrip(self, tmp_path):
        cdm._write_calibration(str(tmp_path), 0.32, 1000, 2000)
        assert cdm._read_calibration(str(tmp_path)) == (0.32, 1000, 2000)


class TestConfig:
    def test_load_defaults(self, tmp_path):
        assert cdm._load_last_calibration() == 0.0
        assert cdm._load_last_folder() == pathlib.Path(".")

    def test_malformed_json(self, tmp_path, monkeypatch):
        cfg = tmp_path / "config.json"
        cfg.write_text("not json")
        monkeypatch.setattr(cdm, "CONFIG_PATH", cfg)
        assert cdm._load_last_calibration() == 0.0
        assert cdm._load_last_folder() == pathlib.Path(".")

    def test_non_dict_json(self, tmp_path, monkeypatch):
        cfg = tmp_path / "config.json"
        cfg.write_text("[1, 2, 3]")
        monkeypatch.setattr(cdm, "CONFIG_PATH", cfg)
        assert cdm._load_last_calibration() == 0.0
        assert cdm._load_last_folder() == pathlib.Path(".")

    def test_calibration_roundtrip(self, tmp_path):
        cdm._save_last_calibration(0.42)
        assert cdm._load_last_calibration() == 0.42

    def test_folder_roundtrip(self, tmp_path):
        p = tmp_path / "my_experiment"
        cdm._save_last_folder(p)
        # compare Path objects to avoid separator issues
        assert cdm._load_last_folder() == p

    def test_two_keys_coexist(self, tmp_path):
        cdm._save_last_calibration(0.25)
        cdm._save_last_folder(tmp_path)
        assert cdm._load_last_calibration() == 0.25
        assert cdm._load_last_folder() == tmp_path

    def test_negative_calibration_loads_as_zero(self, tmp_path):
        cdm._save_last_calibration(-1.0)
        assert cdm._load_last_calibration() == 0.0

    def test_json_is_human_readable(self, tmp_path):
        cdm._save_last_calibration(0.5)
        data = json.loads((tmp_path / "config.json").read_text())
        assert data["last_pixel_calibration"] == 0.5


def _make_reference(viewer, shape, name="Ref"):
    return viewer.add_image(np.zeros(shape, dtype=np.uint8), name=name)


def _make_points(viewer, data, name="Pts"):
    return viewer.add_points(np.asarray(data, dtype=float), name=name)


class TestWidgetValidation:
    def test_image_as_input_layer(self, viewer, tmp_path):
        img = viewer.add_image(np.zeros((50, 50), dtype=np.uint8))
        ref = viewer.add_image(np.zeros((50, 50), dtype=np.uint8))
        widget = cdm.generate_density_maps()
        with patch.object(cdm, "show_modal_error") as err:
            result = widget(
                Input_layer=img,
                Reference_image=ref,
                Save_folder=tmp_path,
            )
        err.assert_called_once()
        assert result == "Invalid input layer."

    def test_points_as_reference(self, viewer, tmp_path):
        ref = viewer.add_points(np.zeros((1, 2)))
        pts = viewer.add_points(np.array([[1.0, 1.0]]))
        widget = cdm.generate_density_maps()
        with patch.object(cdm, "show_modal_error") as err:
            result = widget(
                Input_layer=pts,
                Reference_image=ref,
                Save_folder=tmp_path,
            )
        err.assert_called_once()
        assert result == "Invalid reference image."

    def test_missing_save_folder(self, viewer):
        pts = viewer.add_points(np.array([[1.0, 1.0]]))
        ref = viewer.add_image(np.zeros((50, 50), dtype=np.uint8))
        widget = cdm.generate_density_maps()
        with patch.object(cdm, "show_modal_error") as err:
            result = widget(
                Input_layer=pts,
                Reference_image=ref,
                Save_folder="",
            )
        err.assert_called_once()
        assert "parent folder" in err.call_args[0][0]
        assert result == "No parent folder selected."

    def test_empty_subfolder_name(self, viewer, tmp_path):
        pts = viewer.add_points(np.array([[1.0, 1.0]]))
        ref = viewer.add_image(np.zeros((50, 50), dtype=np.uint8))
        widget = cdm.generate_density_maps()
        with patch.object(cdm, "show_modal_error") as err:
            result = widget(
                Input_layer=pts,
                Reference_image=ref,
                Save_folder=tmp_path,
                Subfolder_name="   ",
            )
        err.assert_called_once()
        assert result == "No subfolder name provided."

    def test_density_size_zero(self, viewer, tmp_path):
        pts = viewer.add_points(np.array([[1.0, 1.0]]))
        ref = viewer.add_image(np.zeros((50, 50), dtype=np.uint8))
        widget = cdm.generate_density_maps()
        with patch.object(cdm, "show_modal_error") as err:
            result = widget(
                Input_layer=pts,
                Reference_image=ref,
                Density_size=0,
                Save_folder=tmp_path,
            )
        err.assert_called_once()
        assert result == "Invalid Density_size."

    def test_ambiguous_reference_shape(self, viewer, tmp_path):
        pts = viewer.add_points(np.array([[1.0, 1.0]]))
        # (50, 50, 5) is rejected by _extract_frame_and_spatial_shape
        ref = viewer.add_image(np.zeros((50, 50, 5), dtype=np.uint8))
        widget = cdm.generate_density_maps()
        with patch.object(cdm, "show_modal_error") as err:
            result = widget(
                Input_layer=pts,
                Reference_image=ref,
                Save_folder=tmp_path,
            )
        err.assert_called_once()
        assert result == "Invalid reference image shape."

    def test_dimensionality_mismatch(self, viewer, tmp_path):
        # 2D points, 1-stack reference
        pts = viewer.add_points(np.array([[1.0, 1.0]]))
        ref = viewer.add_image(np.zeros((3, 50, 50), dtype=np.uint8))
        widget = cdm.generate_density_maps()
        with patch.object(cdm, "show_modal_error") as err:
            result = widget(
                Input_layer=pts,
                Reference_image=ref,
                Save_folder=tmp_path,
            )
        err.assert_called_once()
        assert result == "Mismatched dimensionality."


class TestCalibrationBranches:
    def _run(self, viewer, tmp_path, ref_shape, pts_data, calib):
        pts = viewer.add_points(np.asarray(pts_data, dtype=float), name="Pts")
        ref = viewer.add_image(np.zeros(ref_shape, dtype=np.uint8))
        widget = cdm.generate_density_maps()
        with (
            patch.object(cdm, "show_modal_error"),
            patch.object(cdm, "show_modal_warning") as warn,
            patch.object(cdm, "show_info"),
        ):
            result = widget(
                Input_layer=pts,
                Reference_image=ref,
                Pixel_calibration=calib,
                Save_folder=tmp_path,
                Subfolder_name="Run",
            )
        return result, warn

    def test_create_new_calibration(self, viewer, tmp_path):
        result, warn = self._run(
            viewer, tmp_path, (100, 100), [[30.0, 30.0]], 0.5
        )
        calib_file = tmp_path / "Run" / "calibration.txt"
        assert calib_file.exists()
        assert "created" in result
        warn.assert_called_once()

    def test_missing_calibration_rejected(self, viewer, tmp_path):
        pts = viewer.add_points(np.array([[30.0, 30.0]]))
        ref = viewer.add_image(np.zeros((100, 100), dtype=np.uint8))
        widget = cdm.generate_density_maps()
        with patch.object(cdm, "show_modal_error") as err:
            result = widget(
                Input_layer=pts,
                Reference_image=ref,
                Pixel_calibration=0.0,
                Save_folder=tmp_path,
                Subfolder_name="Run",
            )
        err.assert_called_once()
        assert result == "No pixel calibration provided."

    def test_keep_existing_when_field_zero(self, viewer, tmp_path):
        run_dir = tmp_path / "Run"
        run_dir.mkdir()
        (run_dir / "calibration.txt").write_text(
            "pixel_size_um=0.32\nimage_size=100x100\n"
        )
        result, warn = self._run(
            viewer, tmp_path, (100, 100), [[30.0, 30.0]], 0.0
        )
        assert "kept" in result
        assert "0.32" in (run_dir / "calibration.txt").read_text()
        warn.assert_not_called()

    def test_unchanged(self, viewer, tmp_path):
        run_dir = tmp_path / "Run"
        run_dir.mkdir()
        (run_dir / "calibration.txt").write_text(
            "pixel_size_um=0.5\nimage_size=100x100\n"
        )
        result, warn = self._run(
            viewer, tmp_path, (100, 100), [[30.0, 30.0]], 0.5
        )
        assert "unchanged" in result
        warn.assert_not_called()

    def test_updated_fires_warning(self, viewer, tmp_path):
        run_dir = tmp_path / "Run"
        run_dir.mkdir()
        (run_dir / "calibration.txt").write_text(
            "pixel_size_um=0.5\nimage_size=100x100\n"
        )
        result, warn = self._run(
            viewer, tmp_path, (100, 100), [[30.0, 30.0]], 0.7
        )
        assert "updated" in result
        warn.assert_called_once()
        assert "0.5" in warn.call_args[0][0]
        assert "0.7" in warn.call_args[0][0]

    def test_image_size_change_warning(self, viewer, tmp_path):
        run_dir = tmp_path / "Run"
        run_dir.mkdir()
        (run_dir / "calibration.txt").write_text(
            "pixel_size_um=0.5\nimage_size=64x64\n"
        )
        _, warn = self._run(viewer, tmp_path, (100, 100), [[30.0, 30.0]], 0.0)
        warn.assert_called()
        assert any(
            "Image size changed" in c[0][0] for c in warn.call_args_list
        )


class TestSuccessPath:
    def test_single_image(self, viewer, tmp_path):
        pts = viewer.add_points(np.array([[30.0, 40.0]]))
        ref = viewer.add_image(np.zeros((100, 100), dtype=np.uint8))
        widget = cdm.generate_density_maps()
        with patch.object(cdm, "show_info"):
            result = widget(
                Input_layer=pts,
                Reference_image=ref,
                Pixel_calibration=0.5,
                Density_size=10,
                Save_folder=tmp_path,
                Subfolder_name="Run",
            )
        run_dir = tmp_path / "Run"
        assert (run_dir / "01.npy").exists()
        arr = np.load(run_dir / "01.npy")
        assert arr.shape == (10, 10)
        assert arr.dtype == np.uint32
        assert arr.sum() == 1
        assert (run_dir / "metadata.txt").exists()
        assert (run_dir / "calibration.txt").exists()
        assert "Used index 01" in result

    def test_point_cell_membership(self, viewer, tmp_path):
        pts = viewer.add_points(np.array([[5.0, 5.0], [95.0, 95.0]]))
        ref = viewer.add_image(np.zeros((100, 100), dtype=np.uint8))
        widget = cdm.generate_density_maps()
        with patch.object(cdm, "show_info"):
            widget(
                Input_layer=pts,
                Reference_image=ref,
                Pixel_calibration=0.5,
                Density_size=10,
                Save_folder=tmp_path,
                Subfolder_name="Run",
            )
        arr = np.load(tmp_path / "Run" / "01.npy")
        assert arr[0, 0] == 1
        assert arr[9, 9] == 1
        assert arr.sum() == 2

    def test_boundary_point_goes_to_higher_cell(self, viewer, tmp_path):
        # exact edge between cell 0 and cell 1 at y = 10
        pts = viewer.add_points(np.array([[10.0, 5.0]]))
        ref = viewer.add_image(np.zeros((100, 100), dtype=np.uint8))
        widget = cdm.generate_density_maps()
        with patch.object(cdm, "show_info"):
            widget(
                Input_layer=pts,
                Reference_image=ref,
                Pixel_calibration=0.5,
                Density_size=10,
                Save_folder=tmp_path,
                Subfolder_name="Run",
            )
        arr = np.load(tmp_path / "Run" / "01.npy")
        assert arr[1, 0] == 1

    def test_out_of_bounds_points_dropped(self, viewer, tmp_path):
        pts = viewer.add_points(
            np.array([[-1.0, 5.0], [5.0, 5.0], [200.0, 5.0]])
        )
        ref = viewer.add_image(np.zeros((100, 100), dtype=np.uint8))
        widget = cdm.generate_density_maps()
        with patch.object(cdm, "show_info"):
            result = widget(
                Input_layer=pts,
                Reference_image=ref,
                Pixel_calibration=0.5,
                Density_size=10,
                Save_folder=tmp_path,
                Subfolder_name="Run",
            )
        assert "Dropped 2 out-of-bounds" in result

    def test_empty_points_layer(self, viewer, tmp_path):
        pts = viewer.add_points(np.empty((0, 2)))
        ref = viewer.add_image(np.zeros((100, 100), dtype=np.uint8))
        widget = cdm.generate_density_maps()
        with patch.object(cdm, "show_info"):
            widget(
                Input_layer=pts,
                Reference_image=ref,
                Pixel_calibration=0.5,
                Density_size=10,
                Save_folder=tmp_path,
                Subfolder_name="Run",
            )
        arr = np.load(tmp_path / "Run" / "01.npy")
        assert arr.sum() == 0

    def test_shapes_layer(self, viewer, tmp_path):
        rect = np.array(
            [[10.0, 20.0], [10.0, 40.0], [30.0, 40.0], [30.0, 20.0]]
        )
        shapes = viewer.add_shapes(
            [rect], shape_type="rectangle", name="Boxes"
        )
        ref = viewer.add_image(np.zeros((100, 100), dtype=np.uint8))
        widget = cdm.generate_density_maps()
        with patch.object(cdm, "show_info"):
            widget(
                Input_layer=shapes,
                Reference_image=ref,
                Pixel_calibration=0.5,
                Density_size=10,
                Save_folder=tmp_path,
                Subfolder_name="Run",
            )
        arr = np.load(tmp_path / "Run" / "01.npy")
        # bbox center = (20, 30) -> cell (2, 3)
        assert arr[2, 3] == 1

    def test_1stack_writes_all_frames(self, viewer, tmp_path):
        pts = viewer.add_points(
            np.array([[0.0, 30.0, 30.0], [2.0, 50.0, 50.0]])
        )
        ref = viewer.add_image(np.zeros((3, 100, 100), dtype=np.uint8))
        widget = cdm.generate_density_maps()
        with patch.object(cdm, "show_info"):
            widget(
                Input_layer=pts,
                Reference_image=ref,
                Pixel_calibration=0.5,
                Density_size=10,
                Save_folder=tmp_path,
                Subfolder_name="Run",
            )
        run_dir = tmp_path / "Run"
        assert (run_dir / "01_00.npy").exists()
        assert (run_dir / "01_01.npy").exists()  # empty frame still written
        assert (run_dir / "01_02.npy").exists()
        assert (run_dir / "01_01.npy").read_bytes()  # exists
        assert np.load(run_dir / "01_01.npy").sum() == 0

    def test_2stack_naming_and_guard(self, viewer, tmp_path):
        pts = viewer.add_points(
            np.array([[0.0, 0.0, 30.0, 30.0], [1.0, 1.0, 50.0, 50.0]])
        )
        ref = viewer.add_image(np.zeros((2, 2, 100, 100), dtype=np.uint8))
        widget = cdm.generate_density_maps()
        with patch.object(cdm, "show_info"):
            widget(
                Input_layer=pts,
                Reference_image=ref,
                Pixel_calibration=0.5,
                Density_size=10,
                Save_folder=tmp_path,
                Subfolder_name="Run",
            )
        run_dir = tmp_path / "Run"
        # 2-stack filenames omit the index prefix
        assert (run_dir / "00_00.npy").exists()
        assert (run_dir / "01_01.npy").exists()
        assert (run_dir / "00_01.npy").exists()
        assert (run_dir / "01_00.npy").exists()

    def test_2stack_rejects_nonempty_folder(self, viewer, tmp_path):
        run_dir = tmp_path / "Run"
        run_dir.mkdir()
        (run_dir / "existing.npy").touch()
        pts = viewer.add_points(np.array([[0.0, 0.0, 30.0, 30.0]]))
        ref = viewer.add_image(np.zeros((2, 2, 100, 100), dtype=np.uint8))
        widget = cdm.generate_density_maps()
        with patch.object(cdm, "show_modal_error") as err:
            result = widget(
                Input_layer=pts,
                Reference_image=ref,
                Pixel_calibration=0.5,
                Save_folder=tmp_path,
                Subfolder_name="Run",
            )
        err.assert_called_once()
        assert result == "Folder already contains density maps."

    def test_index_collision_falls_back(self, viewer, tmp_path):
        run_dir = tmp_path / "Run"
        run_dir.mkdir()
        (run_dir / "01.npy").touch()
        pts = viewer.add_points(np.array([[30.0, 30.0]]))
        ref = viewer.add_image(np.zeros((100, 100), dtype=np.uint8))
        widget = cdm.generate_density_maps()
        with (
            patch.object(cdm, "show_info") as info,
            patch.object(cdm, "show_modal_error"),
        ):
            widget(
                Input_layer=pts,
                Reference_image=ref,
                Pixel_calibration=0.5,
                Index=1,
                Density_size=10,
                Save_folder=tmp_path,
                Subfolder_name="Run",
            )
        assert (run_dir / "02.npy").exists()
        # fallback info toast fired
        assert any("Falling back" in c[0][0] for c in info.call_args_list)

    def test_config_persisted(self, viewer, tmp_path):
        pts = viewer.add_points(np.array([[30.0, 30.0]]))
        ref = viewer.add_image(np.zeros((100, 100), dtype=np.uint8))
        widget = cdm.generate_density_maps()
        with patch.object(cdm, "show_info"):
            widget(
                Input_layer=pts,
                Reference_image=ref,
                Pixel_calibration=0.42,
                Density_size=10,
                Save_folder=tmp_path,
                Subfolder_name="Run",
            )
        assert cdm._load_last_calibration() == 0.42
        assert cdm._load_last_folder() == tmp_path
