import math
import os
import pathlib
import re
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from magicgui import magic_factory
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from napari.utils.notifications import show_error, show_info

from napari_nuclephaser.utils import show_modal_error


def _parse_filename(filename):
    stem = filename[:-4]
    parts = stem.split("_")
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        return None


def _read_calibration(folder):
    path = os.path.join(str(folder), "calibration.txt")
    if not os.path.isfile(path):
        return None, None, None
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return None, None, None

    pixel_um = None
    image_h = None
    image_w = None

    m = re.search(r"pixel_size_um\s*=\s*([0-9]*\.?[0-9]+)", content)
    if m:
        try:
            value = float(m.group(1))
            if value > 0:
                pixel_um = value
        except ValueError:
            pass

    m = re.search(r"image_size\s*=\s*(\d+)\s*[xX,]\s*(\d+)", content)
    if m:
        image_h = int(m.group(1))
        image_w = int(m.group(2))

    return pixel_um, image_h, image_w


def _fmt_density(v):
    if v == 0:
        return "0"
    if v < 0:
        return "-" + _fmt_density(-v)
    if v < 1:
        return f"{v:.3g}"
    if v < 1e3:
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.1f}"
    if v < 1e4:
        return str(int(round(v)))
    if v < 1e6:
        return _with_suffix(v, 1e3, "k")
    if v < 1e9:
        return _with_suffix(v, 1e6, "M")
    return _with_suffix(v, 1e9, "G")


def _with_suffix(v, scale, suffix):
    x = v / scale
    if abs(x - round(x)) < 1e-9:
        return f"{int(round(x))}{suffix}"
    if x < 10:
        return f"{x:.2f}{suffix}"
    if x < 100:
        return f"{x:.1f}{suffix}"
    return f"{int(round(x))}{suffix}"


def _px_to_pt(px, dpi):
    return px * 72.0 / dpi


def _text_width_px(text, font_px, dpi):
    if not text:
        return 0.0
    fp = FontProperties(size=font_px * 72.0 / dpi)
    tp = TextPath((0, 0), text, prop=fp)
    bbox = tp.get_extents()
    return bbox.width * dpi / 72.0


def _resolve_colormap(name):
    name = (name or "").strip()
    if name and name in mpl.colormaps:
        return name
    return "inferno"


def _has_compiled_output(folder):
    if not os.path.isdir(folder):
        return False
    return os.path.isfile(os.path.join(folder, "compile_metadata.txt"))


def _resolve_output_folder(parent, name):
    base = os.path.join(str(parent), str(name))
    if not _has_compiled_output(base):
        return base
    i = 1
    while _has_compiled_output(f"{base}_{i}"):
        i += 1
    return f"{base}_{i}"


def _build_grid_image(
    entries, n_rows, n_cols, position_of, map_shape, sep, bg
):
    mh, mw = map_shape
    grid_h = n_rows * mh + (n_rows + 1) * sep
    grid_w = n_cols * mw + (n_cols + 1) * sep
    canvas = np.full((grid_h, grid_w, 3), bg, dtype=np.uint8)
    for key, img in entries.items():
        r, c = position_of(key)
        if r < 0 or r >= n_rows or c < 0 or c >= n_cols:
            continue
        y0 = sep + r * (mh + sep)
        x0 = sep + c * (mw + sep)
        canvas[y0 : y0 + mh, x0 : x0 + mw] = img
    return canvas


def _render_scale(
    colormap_name, max_density, orientation, target_long_px, output_path
):
    cmap = mpl.colormaps[colormap_name]
    dpi = 600

    bar_thick_px = max(60, target_long_px // 6)
    tick_len_px = max(18, bar_thick_px // 3)
    label_gap_px = max(12, bar_thick_px // 5)
    label_font_px = max(36, int(bar_thick_px * 0.55))
    title_font_px = max(40, int(bar_thick_px * 0.6))
    pad_px = max(30, bar_thick_px // 2)
    bar_length_px = target_long_px

    title_text = "nuclei density, nuclei/mm²"
    label_strings = [
        _fmt_density(0),
        _fmt_density(max_density / 2),
        _fmt_density(max_density),
    ]

    label_max_w_px = (
        int(max(_text_width_px(s, label_font_px, dpi) for s in label_strings))
        + 8
    )
    label_h_px = int(label_font_px * 1.15)

    title_w_px = int(_text_width_px(title_text, title_font_px, dpi)) + 8
    title_h_px = int(title_font_px * 1.3)

    if orientation in ("vertical_right", "vertical_left"):
        W_px = (
            pad_px
            + title_h_px
            + bar_thick_px
            + tick_len_px
            + label_gap_px
            + label_max_w_px
            + pad_px
        )

        H_px_bar = label_h_px + bar_length_px + label_h_px + 2 * pad_px
        H_px_title = title_w_px + 2 * pad_px
        H_px = max(H_px_bar, H_px_title)

        bar_y0 = (H_px - bar_length_px) / 2
        bar_y1 = bar_y0 + bar_length_px

        if orientation == "vertical_right":
            bar_x0 = pad_px + title_h_px
            bar_x1 = bar_x0 + bar_thick_px
            tick_x_in = bar_x1
            tick_x_out = bar_x1 + tick_len_px
            label_x = bar_x1 + tick_len_px + label_gap_px
            label_ha = "left"
            title_x = pad_px + title_h_px / 2
        else:
            bar_x0 = pad_px + label_max_w_px + label_gap_px + tick_len_px
            bar_x1 = bar_x0 + bar_thick_px
            tick_x_in = bar_x0
            tick_x_out = bar_x0 - tick_len_px
            label_x = bar_x0 - tick_len_px - label_gap_px
            label_ha = "right"
            title_x = W_px - pad_px - title_h_px / 2

        fig = plt.figure(
            figsize=(W_px / dpi, H_px / dpi), dpi=dpi, facecolor="white"
        )
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W_px)
        ax.set_ylim(0, H_px)
        ax.set_axis_off()

        gradient = np.linspace(0, 1, 512).reshape(-1, 1)
        ax.imshow(
            gradient,
            extent=[bar_x0, bar_x1, bar_y0, bar_y1],
            cmap=cmap,
            aspect="auto",
            origin="lower",
            zorder=1,
        )

        tick_ys = [bar_y0, (bar_y0 + bar_y1) / 2, bar_y1]
        for y_tick, label in zip(tick_ys, label_strings, strict=False):
            ax.plot(
                [tick_x_in, tick_x_out],
                [y_tick, y_tick],
                color="black",
                lw=0.2,
                zorder=2,
            )
            ax.text(
                label_x,
                y_tick,
                label,
                va="center",
                ha=label_ha,
                fontsize=_px_to_pt(label_font_px, dpi),
                color="black",
                zorder=2,
            )

        ax.text(
            title_x,
            H_px / 2,
            title_text,
            rotation=90,
            va="center",
            ha="center",
            fontsize=_px_to_pt(title_font_px, dpi),
            color="black",
        )

        fig.savefig(output_path, dpi=dpi, facecolor="white")
        plt.close(fig)

    else:
        H_px = (
            pad_px
            + title_h_px
            + bar_thick_px
            + tick_len_px
            + label_gap_px
            + label_h_px
            + pad_px
        )

        W_bar = bar_length_px + label_max_w_px + 2 * pad_px
        W_title = title_w_px + 2 * pad_px
        W_px = max(W_bar, W_title)

        bar_x0 = (W_px - bar_length_px) / 2
        bar_x1 = bar_x0 + bar_length_px

        if orientation == "horizontal_top":
            bar_y0 = pad_px + title_h_px
            bar_y1 = bar_y0 + bar_thick_px
            tick_y_in = bar_y1
            tick_y_out = bar_y1 + tick_len_px
            label_y = bar_y1 + tick_len_px + label_gap_px
            label_va = "bottom"
            title_y = pad_px + title_h_px / 2
        else:
            bar_y0 = pad_px + label_h_px + label_gap_px + tick_len_px
            bar_y1 = bar_y0 + bar_thick_px
            tick_y_in = bar_y0
            tick_y_out = bar_y0 - tick_len_px
            label_y = pad_px + label_h_px / 2
            label_va = "center"
            title_y = bar_y1 + title_h_px / 2

        fig = plt.figure(
            figsize=(W_px / dpi, H_px / dpi), dpi=dpi, facecolor="white"
        )
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W_px)
        ax.set_ylim(0, H_px)
        ax.set_axis_off()

        gradient = np.linspace(0, 1, 512).reshape(1, -1)
        ax.imshow(
            gradient,
            extent=[bar_x0, bar_x1, bar_y0, bar_y1],
            cmap=cmap,
            aspect="auto",
            origin="lower",
            zorder=1,
        )

        tick_xs = [bar_x0, (bar_x0 + bar_x1) / 2, bar_x1]
        for x_tick, label in zip(tick_xs, label_strings, strict=False):
            ax.plot(
                [x_tick, x_tick],
                [tick_y_in, tick_y_out],
                color="black",
                lw=0.2,
                zorder=2,
            )
            ax.text(
                x_tick,
                label_y,
                label,
                va=label_va,
                ha="center",
                fontsize=_px_to_pt(label_font_px, dpi),
                color="black",
                zorder=2,
            )

        ax.text(
            W_px / 2,
            title_y,
            title_text,
            rotation=0,
            va="center",
            ha="center",
            fontsize=_px_to_pt(title_font_px, dpi),
            color="black",
        )

        fig.savefig(output_path, dpi=dpi, facecolor="white")
        plt.close(fig)


@magic_factory(
    auto_call=False,
    call_button="Compile density maps",
    result_widget=True,
    Source_folder={"mode": "d", "label": "Source folder (.npy maps)"},
    Save_folder={"mode": "d", "label": "Output parent folder"},
    Subfolder_name={"label": "Output subfolder name"},
    Create_individual_maps={
        "label": "Create individual maps",
        "value": False,
    },
    Combine_maps_into_grid={
        "label": "Combine maps into grid",
        "value": True,
    },
    Number_of_rows={
        "label": "Number of rows (grid)",
        "min": 1,
        "max": 10000,
        "value": 1,
    },
    Colormap={
        "choices": ["inferno", "viridis", "plasma", "magma", "cividis"],
        "value": "inferno",
        "label": "Colormap",
    },
    Another_colormap={
        "label": "Another colormap (name only)",
        "value": "",
    },
)
def compile_density_maps(
    Source_folder: pathlib.Path,
    Save_folder: pathlib.Path = pathlib.Path(),
    Subfolder_name: str = "CompiledDensityMaps",
    Create_individual_maps: bool = False,
    Combine_maps_into_grid: bool = True,
    Number_of_rows: int = 1,
    Colormap: str = "inferno",
    Another_colormap: str = "",
) -> str:
    if not Source_folder or not os.path.isdir(str(Source_folder)):
        show_error("Source folder does not exist.")
        return "Invalid source folder."

    if not Save_folder:
        show_error("Please select an output parent folder.")
        return "No output folder selected."

    if not Subfolder_name or not str(Subfolder_name).strip():
        show_error("Please provide an output subfolder name.")
        return "No subfolder name provided."

    if not Create_individual_maps and not Combine_maps_into_grid:
        show_modal_error(
            "Please select at least one output option: create individual "
            "maps and/or combine maps into grid."
        )
        return "No output option selected."

    npy_files = sorted(
        f for f in os.listdir(str(Source_folder)) if f.endswith(".npy")
    )
    if not npy_files:
        show_error("No density maps found in given folder")
        return "No density maps found in given folder"

    parsed = {}
    for f in npy_files:
        key = _parse_filename(f)
        if key is not None:
            parsed[f] = key

    if not parsed:
        show_error("No valid density maps found in given folder")
        return "No valid density maps."

    lengths = {len(k) for k in parsed.values()}
    if len(lengths) != 1:
        show_error(
            f"Mixed density map dimensionality in folder: {sorted(lengths)}"
        )
        return "Mixed dimensions."

    n_comp = lengths.pop()
    if n_comp not in (1, 2):
        show_error(f"Unexpected filename structure ({n_comp} components).")
        return "Invalid filenames."

    pixel_um, img_h, img_w = _read_calibration(str(Source_folder))
    if pixel_um is None:
        show_error("No pixel calibration found in source folder.")
        return "Missing pixel calibration."
    if img_h is None or img_w is None:
        show_error("No image size found in source calibration file.")
        return "Missing image size."

    maps_raw = {}
    shapes_seen = {}
    for f, _ in parsed.items():
        arr = np.load(os.path.join(str(Source_folder), f))
        if arr.ndim != 2:
            show_error(f"Unexpected array shape in {f}: {arr.shape}")
            return "Invalid array shape."
        maps_raw[f] = arr
        shapes_seen.setdefault(arr.shape, []).append(f)

    if len(shapes_seen) > 1:
        details = "; ".join(
            f"{shape}: {', '.join(files[:3])}"
            for shape, files in shapes_seen.items()
        )
        show_error(f"Density maps have non-uniform shapes: {details}")
        return "Non-uniform density map shapes."

    map_h, map_w = next(iter(shapes_seen.keys()))

    if Combine_maps_into_grid:
        n_maps = len(maps_raw)
        if Number_of_rows > n_maps:
            show_modal_error(
                f"Number of rows ({Number_of_rows}) is larger than the "
                f"number of density maps ({n_maps}). Please reduce the "
                "number of rows or disable grid creation."
            )
            return "Number of rows exceeds number of maps."
        if Number_of_rows < 1:
            show_modal_error("Number of rows must be at least 1.")
            return "Invalid number of rows."

    max_val = max(int(m.max()) for m in maps_raw.values())

    if Another_colormap.strip():
        cmap_name = _resolve_colormap(Another_colormap)
    else:
        cmap_name = _resolve_colormap(Colormap)
    cmap = mpl.colormaps[cmap_name]

    maps_rgb = {}
    for fname, raw in maps_raw.items():
        if max_val > 0:
            normalized = raw.astype(np.float32) / float(max_val)
        else:
            normalized = np.zeros_like(raw, dtype=np.float32)
        rgba = cmap(normalized)
        maps_rgb[fname] = (rgba[:, :, :3] * 255).astype(np.uint8)

    out_folder = _resolve_output_folder(str(Save_folder), str(Subfolder_name))
    os.makedirs(out_folder, exist_ok=True)

    # -------- Individual maps --------
    if Create_individual_maps:
        individual_folder = os.path.join(out_folder, "Individual_maps")
        os.makedirs(individual_folder, exist_ok=True)
        for fname, rgb in maps_rgb.items():
            out_name = fname[:-4] + ".tiff"
            tifffile.imwrite(os.path.join(individual_folder, out_name), rgb)

    # -------- Grid --------
    grid_h_shape = None
    grid_v_shape = None
    if Combine_maps_into_grid:
        sorted_keys = sorted(maps_rgb.keys(), key=lambda f: parsed[f])
        n_total = len(sorted_keys)
        n_rows_h = Number_of_rows
        n_cols_h = math.ceil(n_total / n_rows_h)

        position_h = {}
        for idx, fname in enumerate(sorted_keys):
            position_h[fname] = (idx // n_cols_h, idx % n_cols_h)

        n_rows_v = n_cols_h
        n_cols_v = n_rows_h

        position_v = {}
        for fname, (r, c) in position_h.items():
            position_v[fname] = (c, r)

        sep = 5
        bg = 255

        grid_h_img = _build_grid_image(
            maps_rgb,
            n_rows_h,
            n_cols_h,
            lambda f: position_h[f],
            (map_h, map_w),
            sep,
            bg,
        )
        grid_v_img = _build_grid_image(
            maps_rgb,
            n_rows_v,
            n_cols_v,
            lambda f: position_v[f],
            (map_h, map_w),
            sep,
            bg,
        )

        tifffile.imwrite(
            os.path.join(out_folder, "horizontal_maps.tif"), grid_h_img
        )
        tifffile.imwrite(
            os.path.join(out_folder, "vertical_maps.tif"), grid_v_img
        )

        grid_h_shape = grid_h_img.shape[:2]
        grid_v_shape = grid_v_img.shape[:2]

    # -------- Scales --------
    cell_h_px = img_h / map_h
    cell_w_px = img_w / map_w
    cell_h_mm = cell_h_px * pixel_um / 1000.0
    cell_w_mm = cell_w_px * pixel_um / 1000.0
    cell_area_mm2 = cell_h_mm * cell_w_mm
    max_density = max_val / cell_area_mm2 if cell_area_mm2 > 0 else 0.0

    if grid_h_shape is not None:
        grid_long = max(grid_h_shape)
    else:
        grid_long = max(map_h, map_w) * 4
    target_long_px = max(600, min(1500, grid_long // 2))

    _render_scale(
        cmap_name,
        max_density,
        "vertical_right",
        target_long_px,
        os.path.join(out_folder, "scale_vertical_right.png"),
    )
    _render_scale(
        cmap_name,
        max_density,
        "vertical_left",
        target_long_px,
        os.path.join(out_folder, "scale_vertical_left.png"),
    )
    _render_scale(
        cmap_name,
        max_density,
        "horizontal_top",
        target_long_px,
        os.path.join(out_folder, "scale_horizontal_top.png"),
    )
    _render_scale(
        cmap_name,
        max_density,
        "horizontal_bottom",
        target_long_px,
        os.path.join(out_folder, "scale_horizontal_bottom.png"),
    )

    # -------- Metadata --------
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    individual_str = "yes" if Create_individual_maps else "no"
    grid_str = "yes" if Combine_maps_into_grid else "no"
    grid_h_str = str(grid_h_shape) if grid_h_shape is not None else "N/A"
    grid_v_str = str(grid_v_shape) if grid_v_shape is not None else "N/A"
    rows_str = str(Number_of_rows) if Combine_maps_into_grid else "N/A"

    metadata = f"""Experiment time: {current_date}
Source folder: {Source_folder}
Number of density maps: {len(maps_rgb)}
Map shape (h, w): ({map_h}, {map_w})
Create individual maps: {individual_str}
Combine maps into grid: {grid_str}
Number of rows (grid): {rows_str}
Grid (horizontal) shape: {grid_h_str}
Grid (vertical) shape: {grid_v_str}
Max raw count: {max_val}
Pixel calibration (µm/px): {pixel_um}
Image size (h x w): {img_h} x {img_w}
Cell size (h x w, px): {cell_h_px:.4f} x {cell_w_px:.4f}
Cell area (mm^2): {cell_area_mm2:.6g}
Max density (nuclei/mm²): {_fmt_density(max_density)}
Colormap: {cmap_name}
"""
    if Create_individual_maps:
        metadata += (
            "Individual maps:\n  Individual_maps/<original_name>.tiff\n"
        )
    if Combine_maps_into_grid:
        metadata += "Grid files:\n  horizontal_maps.tif\n  vertical_maps.tif\n"
    metadata += (
        "Scale files:\n"
        "  scale_vertical_right.png\n"
        "  scale_vertical_left.png\n"
        "  scale_horizontal_top.png\n"
        "  scale_horizontal_bottom.png\n"
    )

    with open(
        os.path.join(out_folder, "compile_metadata.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(metadata)

    parts = [f"Compiled {len(maps_rgb)} density maps into {out_folder}."]
    if Combine_maps_into_grid:
        parts.append(f"Grid (horizontal): {grid_h_shape}.")
    parts.append(f"Max density: {_fmt_density(max_density)} nuclei/mm².")
    summary = " ".join(parts)
    show_info(summary)
    return summary
