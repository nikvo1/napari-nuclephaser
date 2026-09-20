from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import napari_nuclephaser.compile_density_maps as cdm


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(cdm, "CONFIG_PATH", tmp_path / "config.json")


@pytest.fixture(autouse=True)
def silence_modals(monkeypatch):
    monkeypatch.setattr(cdm, "show_modal_error", MagicMock(), raising=False)
    monkeypatch.setattr(cdm, "show_modal_warning", MagicMock(), raising=False)


def _seed_source(
    folder,
    index=1,
    n_maps=3,
    shape=(10, 10),
    pixel_um=0.5,
    image_size=(100, 100),
    max_count=5,
):
    folder.mkdir(exist_ok=True)
    for i in range(n_maps):
        arr = np.zeros(shape, dtype=np.uint32)
        arr[0, 0] = max_count
        np.save(folder / f"{index:02d}_{i:02d}.npy", arr)
    (folder / "calibration.txt").write_text(
        f"pixel_size_um={pixel_um}\n"
        f"image_size={image_size[0]}x{image_size[1]}\n"
    )


class TestParseFilename:
    def test_index_only(self):
        assert cdm._parse_filename("01.npy") == (1,)

    def test_index_and_frame(self):
        assert cdm._parse_filename("01_07.npy") == (1, 7)

    def test_2stack(self):
        assert cdm._parse_filename("07_12.npy") == (7, 12)

    def test_letters(self):
        assert cdm._parse_filename("abc.npy") is None

    def test_mixed(self):
        assert cdm._parse_filename("01_abc.npy") is None


class TestFmtDensity:
    def test_zero(self):
        assert cdm._fmt_density(0) == "0"

    def test_negative(self):
        assert cdm._fmt_density(-500) == "-500"

    def test_sub_one(self):
        assert cdm._fmt_density(0.123) == "0.123"

    def test_small_fraction(self):
        assert cdm._fmt_density(0.0456) == "0.0456"

    def test_int_under_1000(self):
        assert cdm._fmt_density(1) == "1"
        assert cdm._fmt_density(123) == "123"

    def test_float_under_1000(self):
        assert cdm._fmt_density(12.7) == "12.7"

    def test_1000_to_9999(self):
        assert cdm._fmt_density(1000) == "1000"
        assert cdm._fmt_density(9999) == "9999"

    def test_k_suffix(self):
        assert "k" in cdm._fmt_density(13700)
        assert "k" in cdm._fmt_density(123000)

    def test_M_suffix(self):
        assert "M" in cdm._fmt_density(1.5e6)

    def test_G_suffix(self):
        assert "G" in cdm._fmt_density(1.5e9)


class TestWithSuffix:
    def test_integer(self):
        assert cdm._with_suffix(1000, 1e3, "k") == "1k"

    def test_single_digit_decimal(self):
        assert cdm._with_suffix(12345, 1e3, "k") == "12.3k"

    def test_double_digit_decimal(self):
        assert cdm._with_suffix(1234, 1e3, "k") == "1.23k"

    def test_above_100(self):
        assert cdm._with_suffix(123456, 1e3, "k") == "123k"


class TestPxToPt:
    def test_identity(self):
        # at 72 dpi, 1 px = 1 pt
        assert cdm._px_to_pt(72, 72) == 72.0

    def test_scales_with_dpi(self):
        # at 144 dpi, 2 px = 1 pt
        assert cdm._px_to_pt(144, 144) == 72.0


class TestTextWidth:
    def test_empty(self):
        assert cdm._text_width_px("", 20, 600) == 0.0

    def test_positive(self):
        assert cdm._text_width_px("hello", 20, 600) > 0

    def test_monotonic_in_font(self):
        small = cdm._text_width_px("hello", 10, 600)
        big = cdm._text_width_px("hello", 40, 600)
        assert big > small


class TestResolveColormap:
    def test_known(self):
        assert cdm._resolve_colormap("viridis") == "viridis"

    def test_unknown_falls_back(self):
        assert cdm._resolve_colormap("nope") == "inferno"

    def test_none(self):
        assert cdm._resolve_colormap(None) == "inferno"

    def test_empty(self):
        assert cdm._resolve_colormap("") == "inferno"

    def test_whitespace(self):
        assert cdm._resolve_colormap("   ") == "inferno"

    def test_strip(self):
        assert cdm._resolve_colormap("  viridis  ") == "viridis"


class TestOutputFolder:
    def test_missing(self, tmp_path):
        assert cdm._has_compiled_output(str(tmp_path / "nope")) is False

    def test_no_metadata(self, tmp_path):
        (tmp_path / "01.npy").touch()
        assert cdm._has_compiled_output(str(tmp_path)) is False

    def test_with_metadata(self, tmp_path):
        (tmp_path / "compile_metadata.txt").touch()
        assert cdm._has_compiled_output(str(tmp_path)) is True

    def test_fresh(self, tmp_path):
        result = cdm._resolve_output_folder(str(tmp_path), "Out")
        assert result == str(tmp_path / "Out")

    def test_reuse_when_npy_only(self, tmp_path):
        base = tmp_path / "Out"
        base.mkdir()
        (base / "01.npy").touch()
        # no compile_metadata.txt -> folder is reused
        assert cdm._resolve_output_folder(str(tmp_path), "Out") == str(base)

    def test_uniquify_when_metadata_present(self, tmp_path):
        base = tmp_path / "Out"
        base.mkdir()
        (base / "compile_metadata.txt").touch()
        assert (
            cdm._resolve_output_folder(str(tmp_path), "Out")
            == str(base) + "_1"
        )

    def test_increment(self, tmp_path):
        base = tmp_path / "Out"
        base.mkdir()
        (base / "compile_metadata.txt").touch()
        (tmp_path / "Out_1").mkdir()
        (tmp_path / "Out_1" / "compile_metadata.txt").touch()
        assert (
            cdm._resolve_output_folder(str(tmp_path), "Out")
            == str(base) + "_2"
        )


class TestBuildGridImage:
    def test_empty_returns_bg_canvas(self):
        img = cdm._build_grid_image(
            entries={},
            n_rows=2,
            n_cols=3,
            position_of=lambda k: (0, 0),
            map_shape=(10, 20),
            sep=5,
            bg=255,
        )
        assert img.shape == (35, 80, 3)
        assert (img == 255).all()

    def test_single_entry_content(self):
        tile = np.full((4, 4, 3), 128, dtype=np.uint8)
        img = cdm._build_grid_image(
            entries={"a": tile},
            n_rows=1,
            n_cols=1,
            position_of=lambda k: (0, 0),
            map_shape=(4, 4),
            sep=2,
            bg=255,
        )
        assert (img[:2, :] == 255).all()
        assert (img[2:6, 2:6] == 128).all()

    def test_multiple_entries_placement(self):
        a = np.full((2, 2, 3), 10, dtype=np.uint8)
        b = np.full((2, 2, 3), 20, dtype=np.uint8)
        positions = {"a": (0, 0), "b": (0, 1)}
        img = cdm._build_grid_image(
            entries={"a": a, "b": b},
            n_rows=1,
            n_cols=2,
            position_of=lambda k: positions[k],
            map_shape=(2, 2),
            sep=1,
            bg=255,
        )
        assert (img[1:3, 1:3] == 10).all()
        assert (img[1:3, 4:6] == 20).all()

    def test_out_of_bounds_skipped(self):
        tile = np.full((2, 2, 3), 99, dtype=np.uint8)
        img = cdm._build_grid_image(
            entries={"a": tile},
            n_rows=1,
            n_cols=1,
            position_of=lambda k: (5, 5),
            map_shape=(2, 2),
            sep=1,
            bg=255,
        )
        assert (img == 255).all()


class TestRenderScale:
    @pytest.mark.parametrize(
        "orientation",
        [
            "vertical_right",
            "vertical_left",
            "horizontal_top",
            "horizontal_bottom",
        ],
    )
    def test_writes_file(self, tmp_path, orientation):
        out = tmp_path / f"scale_{orientation}.png"
        cdm._render_scale(
            colormap_name="inferno",
            max_density=1234.0,
            orientation=orientation,
            target_long_px=400,
            output_path=str(out),
        )
        assert out.exists()
        assert out.stat().st_size > 0

    def test_zero_density(self, tmp_path):
        out = tmp_path / "scale.png"
        cdm._render_scale(
            colormap_name="inferno",
            max_density=0.0,
            orientation="horizontal_top",
            target_long_px=400,
            output_path=str(out),
        )
        assert out.exists()


def _make_source(tmp_path, **kwargs):
    src = tmp_path / "src"
    _seed_source(src, **kwargs)
    return src


class TestWidgetValidation:
    def test_missing_source(self, tmp_path):
        widget = cdm.compile_density_maps()
        with patch.object(cdm, "show_modal_error") as err:
            result = widget(
                Source_folder=tmp_path / "nope",
                Save_folder=tmp_path,
                Subfolder_name="Out",
            )
        err.assert_called_once()
        assert result == "Invalid source folder."

    def test_empty_save_folder(self, tmp_path):
        src = _make_source(tmp_path)
        widget = cdm.compile_density_maps()
        with patch.object(cdm, "show_modal_error") as err:
            result = widget(
                Source_folder=src,
                Save_folder="",
                Subfolder_name="Out",
            )
        err.assert_called_once()
        assert result == "No output parent folder selected."

    def test_empty_subfolder(self, tmp_path):
        src = _make_source(tmp_path)
        widget = cdm.compile_density_maps()
        with patch.object(cdm, "show_modal_error") as err:
            result = widget(
                Source_folder=src,
                Save_folder=tmp_path,
                Subfolder_name="   ",
            )
        err.assert_called_once()
        assert result == "No subfolder name provided."

    def test_no_output_option(self, tmp_path):
        src = _make_source(tmp_path)
        widget = cdm.compile_density_maps()
        with patch.object(cdm, "show_modal_error") as err:
            result = widget(
                Source_folder=src,
                Save_folder=tmp_path,
                Subfolder_name="Out",
                Create_individual_maps=False,
                Combine_maps_into_grid=False,
            )
        err.assert_called_once()
        assert result == "No output option selected."

    def test_no_npy_files(self, tmp_path):
        src = tmp_path / "empty"
        src.mkdir()
        (src / "calibration.txt").write_text(
            "pixel_size_um=0.5\nimage_size=100x100\n"
        )
        widget = cdm.compile_density_maps()
        with patch.object(cdm, "show_modal_error") as err:
            result = widget(
                Source_folder=src,
                Save_folder=tmp_path,
                Subfolder_name="Out",
            )
        err.assert_called_once()
        assert result == "No density maps found in given folder"

    def test_non_parsable_names(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "abc.npy").touch()
        (src / "calibration.txt").write_text(
            "pixel_size_um=0.5\nimage_size=100x100\n"
        )
        widget = cdm.compile_density_maps()
        with patch.object(cdm, "show_modal_error") as err:
            result = widget(
                Source_folder=src,
                Save_folder=tmp_path,
                Subfolder_name="Out",
            )
        err.assert_called_once()
        assert result == "No valid density maps."

    def test_mixed_dimensionality(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "01.npy").touch()
        (src / "01_01.npy").touch()
        (src / "calibration.txt").write_text(
            "pixel_size_um=0.5\nimage_size=100x100\n"
        )
        widget = cdm.compile_density_maps()
        with patch.object(cdm, "show_modal_error") as err:
            result = widget(
                Source_folder=src,
                Save_folder=tmp_path,
                Subfolder_name="Out",
            )
        err.assert_called_once()
        assert result == "Mixed dimensions."

    def test_missing_calibration(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        np.save(src / "01.npy", np.zeros((10, 10), dtype=np.uint32))
        widget = cdm.compile_density_maps()
        with patch.object(cdm, "show_modal_error") as err:
            result = widget(
                Source_folder=src,
                Save_folder=tmp_path,
                Subfolder_name="Out",
            )
        err.assert_called_once()
        assert result == "Missing pixel calibration."

    def test_missing_image_size(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        np.save(src / "01.npy", np.zeros((10, 10), dtype=np.uint32))
        (src / "calibration.txt").write_text("pixel_size_um=0.5\n")
        widget = cdm.compile_density_maps()
        with patch.object(cdm, "show_modal_error") as err:
            result = widget(
                Source_folder=src,
                Save_folder=tmp_path,
                Subfolder_name="Out",
            )
        err.assert_called_once()
        assert result == "Missing image size."

    def test_rows_exceed_maps(self, tmp_path):
        src = _make_source(tmp_path, n_maps=2)
        widget = cdm.compile_density_maps()
        with patch.object(cdm, "show_modal_error") as err:
            result = widget(
                Source_folder=src,
                Save_folder=tmp_path,
                Subfolder_name="Out",
                Number_of_rows=5,
            )
        err.assert_called_once()
        assert result == "Number of rows exceeds number of maps."

    def test_non_uniform_shapes(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        np.save(src / "01_00.npy", np.zeros((10, 10), dtype=np.uint32))
        np.save(src / "01_01.npy", np.zeros((20, 20), dtype=np.uint32))
        (src / "calibration.txt").write_text(
            "pixel_size_um=0.5\nimage_size=100x100\n"
        )
        widget = cdm.compile_density_maps()
        with patch.object(cdm, "show_modal_error") as err:
            result = widget(
                Source_folder=src,
                Save_folder=tmp_path,
                Subfolder_name="Out",
            )
        err.assert_called_once()
        assert result == "Non-uniform density map shapes."
        assert "non-uniform shapes" in err.call_args[0][0].lower()


class TestWidgetSuccess:
    def test_individual_maps_only(self, tmp_path):
        src = _make_source(tmp_path)
        widget = cdm.compile_density_maps()
        with patch.object(cdm, "show_info"):
            result = widget(
                Source_folder=src,
                Save_folder=tmp_path,
                Subfolder_name="Out",
                Create_individual_maps=True,
                Combine_maps_into_grid=False,
            )
        out = tmp_path / "Out"
        indiv = out / "Individual_maps"
        assert indiv.is_dir()
        assert (indiv / "01_00.tiff").exists()
        assert (indiv / "01_02.tiff").exists()
        assert not (out / "horizontal_maps.tif").exists()
        # scale files always written
        assert (out / "scale_vertical_right.png").exists()
        assert (out / "scale_vertical_left.png").exists()
        assert (out / "scale_horizontal_top.png").exists()
        assert (out / "scale_horizontal_bottom.png").exists()
        # metadata
        assert (out / "compile_metadata.txt").exists()
        # returned summary references the output folder
        assert "Compiled" in result
        assert str(out) in result

        def test_grid_only(self, tmp_path):
            src = _make_source(tmp_path, n_maps=4)
            widget = cdm.compile_density_maps()
            with patch.object(cdm, "show_info"):
                widget(
                    Source_folder=src,
                    Save_folder=tmp_path,
                    Subfolder_name="Out",
                    Create_individual_maps=False,
                    Combine_maps_into_grid=True,
                    Number_of_rows=2,
                )
            out = tmp_path / "Out"
            assert (out / "horizontal_maps.tif").exists()
            assert (out / "vertical_maps.tif").exists()
            assert not (out / "Individual_maps").exists()
            # 2 rows x 2 cols of 10x10 with 5 sep -> (2*10 + 3*5, 2*10 + 3*5)
            import tifffile

            h = tifffile.imread(out / "horizontal_maps.tif")
            v = tifffile.imread(out / "vertical_maps.tif")
            assert h.shape[:2] == (35, 35)
            assert v.shape[:2] == (35, 35)

    def test_grid_partial_last_row(self, tmp_path):
        # 5 maps, 2 rows -> 3 columns; last row has 1 map
        src = _make_source(tmp_path, n_maps=5)
        widget = cdm.compile_density_maps()
        with patch.object(cdm, "show_info"):
            widget(
                Source_folder=src,
                Save_folder=tmp_path,
                Subfolder_name="Out",
                Create_individual_maps=False,
                Combine_maps_into_grid=True,
                Number_of_rows=2,
            )
        out = tmp_path / "Out"
        import tifffile

        h = tifffile.imread(out / "horizontal_maps.tif")
        # 2 rows: 2*10 + 3*5 = 35 tall; 3 cols: 3*10 + 4*5 = 50 wide
        assert h.shape[:2] == (35, 50)

    def test_both_options(self, tmp_path):
        src = _make_source(tmp_path)
        widget = cdm.compile_density_maps()
        with patch.object(cdm, "show_info"):
            widget(
                Source_folder=src,
                Save_folder=tmp_path,
                Subfolder_name="Out",
                Create_individual_maps=True,
                Combine_maps_into_grid=True,
                Number_of_rows=1,
            )
        out = tmp_path / "Out"
        assert (out / "Individual_maps" / "01_00.tiff").exists()
        assert (out / "horizontal_maps.tif").exists()
        assert (out / "vertical_maps.tif").exists()

    def test_another_colormap_overrides(self, tmp_path):
        src = _make_source(tmp_path)
        widget = cdm.compile_density_maps()
        with patch.object(cdm, "show_info"):
            widget(
                Source_folder=src,
                Save_folder=tmp_path,
                Subfolder_name="Out",
                Colormap="inferno",
                Another_colormap="viridis",
                Create_individual_maps=False,
                Combine_maps_into_grid=True,
            )
        meta = (tmp_path / "Out" / "compile_metadata.txt").read_text()
        assert "Colormap: viridis" in meta

    def test_unknown_colormap_falls_back(self, tmp_path):
        src = _make_source(tmp_path)
        widget = cdm.compile_density_maps()
        with patch.object(cdm, "show_info"):
            widget(
                Source_folder=src,
                Save_folder=tmp_path,
                Subfolder_name="Out",
                Another_colormap="totally_not_a_cmap",
                Create_individual_maps=False,
                Combine_maps_into_grid=True,
            )
        meta = (tmp_path / "Out" / "compile_metadata.txt").read_text()
        assert "Colormap: inferno" in meta

    def test_zero_maps_all(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        for i in range(3):
            np.save(
                src / f"01_{i:02d}.npy",
                np.zeros((10, 10), dtype=np.uint32),
            )
        (src / "calibration.txt").write_text(
            "pixel_size_um=0.5\nimage_size=100x100\n"
        )
        widget = cdm.compile_density_maps()
        with patch.object(cdm, "show_info"):
            widget(
                Source_folder=src,
                Save_folder=tmp_path,
                Subfolder_name="Out",
                Create_individual_maps=False,
                Combine_maps_into_grid=True,
                Number_of_rows=1,
            )
        meta = (tmp_path / "Out" / "compile_metadata.txt").read_text()
        assert "Max raw count: 0" in meta

    def test_uniquify_on_rerun(self, tmp_path):
        src = _make_source(tmp_path)
        widget = cdm.compile_density_maps()
        with patch.object(cdm, "show_info"):
            widget(
                Source_folder=src,
                Save_folder=tmp_path,
                Subfolder_name="Out",
                Create_individual_maps=False,
                Combine_maps_into_grid=True,
                Number_of_rows=1,
            )
            widget(
                Source_folder=src,
                Save_folder=tmp_path,
                Subfolder_name="Out",
                Create_individual_maps=False,
                Combine_maps_into_grid=True,
                Number_of_rows=1,
            )
        assert (tmp_path / "Out").exists()
        assert (tmp_path / "Out_1").exists()

    def test_single_image_input(self, tmp_path):
        # 01.npy only -> n_comp == 1, single-image layout
        src = tmp_path / "src"
        src.mkdir()
        np.save(src / "01.npy", np.ones((10, 10), dtype=np.uint32))
        (src / "calibration.txt").write_text(
            "pixel_size_um=0.5\nimage_size=100x100\n"
        )
        widget = cdm.compile_density_maps()
        with patch.object(cdm, "show_info"):
            widget(
                Source_folder=src,
                Save_folder=tmp_path,
                Subfolder_name="Out",
                Create_individual_maps=False,
                Combine_maps_into_grid=True,
                Number_of_rows=1,
            )
        import tifffile

        h = tifffile.imread(tmp_path / "Out" / "horizontal_maps.tif")
        assert h.shape[:2] == (20, 20)  # 1 map + 2 sep = 20

    def test_config_persisted_on_success(self, tmp_path):
        src = _make_source(tmp_path)
        widget = cdm.compile_density_maps()
        with patch.object(cdm, "show_info"):
            widget(
                Source_folder=src,
                Save_folder=tmp_path,
                Subfolder_name="Out",
                Create_individual_maps=False,
                Combine_maps_into_grid=True,
            )
        # last folder should be the save folder (the last one written)
        assert cdm._load_last_folder() == tmp_path

    def test_config_not_persisted_on_early_return(self, tmp_path):
        widget = cdm.compile_density_maps()
        with patch.object(cdm, "show_modal_error"):
            widget(
                Source_folder=tmp_path / "nonexistent",
                Save_folder=tmp_path,
                Subfolder_name="Out",
            )
        # _save_last_folder never called because we bail before it
        assert not (tmp_path / "config.json").exists()
