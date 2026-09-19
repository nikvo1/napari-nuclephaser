import json
import os
import pathlib
import re
from datetime import datetime

import numpy as np
from magicgui import magic_factory
from napari.layers import Image, Layer, Points, Shapes
from napari.utils.notifications import show_info

from napari_nuclephaser.utils import show_modal_error, show_modal_warning

CONFIG_PATH = pathlib.Path.home() / ".napari_nuclephaser.json"


def _load_last_calibration():
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            data = json.load(f)
        value = float(data.get("last_pixel_calibration", 0.0))
        if value <= 0:
            return 0.0
        return value
    except (OSError, ValueError, TypeError):
        return 0.0


def _save_last_calibration(value):
    try:
        data = {}
        if CONFIG_PATH.is_file():
            try:
                with open(CONFIG_PATH, encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    data = {}
            except (OSError, ValueError):
                data = {}
        data["last_pixel_calibration"] = float(value)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


def _extract_frame_and_spatial_shape(image_shape):
    shape = tuple(image_shape)
    if len(shape) >= 3 and shape[-1] in (1, 3, 4):
        shape = shape[:-1]

    if len(shape) == 2:
        return (), shape
    if len(shape) == 3:
        return (shape[0],), shape[1:]
    if len(shape) == 4:
        return (shape[0], shape[1]), shape[2:]
    raise ValueError(
        f"Unsupported image shape {image_shape}. Expected a single 2D "
        "image, a 1-stack, or a 2-stack (optional trailing channel dim)."
    )


def _compute_map_shape(H, W, density_size):
    if H >= W:
        map_h = density_size
        map_w = max(1, round(W * density_size / H))
    else:
        map_w = density_size
        map_h = max(1, round(H * density_size / W))
    return map_h, map_w


def _iter_points(points_data):
    ndim = points_data.shape[1]
    if ndim == 2:
        for pt in points_data:
            yield (), float(pt[0]), float(pt[1])
    elif ndim == 3:
        for pt in points_data:
            yield (int(pt[0]),), float(pt[1]), float(pt[2])
    elif ndim == 4:
        for pt in points_data:
            yield (int(pt[0]), int(pt[1])), float(pt[2]), float(pt[3])
    else:
        raise ValueError(f"Unsupported Points dimensionality: {ndim}.")


def _iter_shapes(shapes_data):
    if len(shapes_data) == 0:
        return

    first = np.asarray(shapes_data[0])
    ndim = first.shape[1] if first.ndim == 2 else first.shape[0]

    for shape in shapes_data:
        arr = np.asarray(shape)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if ndim == 2:
            frame_key = ()
            ys = arr[:, 0]
            xs = arr[:, 1]
        elif ndim == 3:
            frame_key = (int(arr[0, 0]),)
            ys = arr[:, 1]
            xs = arr[:, 2]
        elif ndim == 4:
            frame_key = (int(arr[0, 0]), int(arr[0, 1]))
            ys = arr[:, 2]
            xs = arr[:, 3]
        else:
            raise ValueError(f"Unsupported Shapes dimensionality: {ndim}.")

        y_center = (float(ys.min()) + float(ys.max())) / 2.0
        x_center = (float(xs.min()) + float(xs.max())) / 2.0
        yield frame_key, y_center, x_center


def _find_next_index(folder, requested_index):
    used = set()
    if os.path.isdir(folder):
        for f in os.listdir(folder):
            if not f.endswith(".npy"):
                continue
            m = re.match(r"^(\d{2})(?=[_.])", f)
            if m:
                idx = int(m.group(1))
                if 1 <= idx <= 99:
                    used.add(idx)

    idx = requested_index
    while idx <= 99 and idx in used:
        idx += 1
    if idx > 99:
        raise RuntimeError(
            "All indices 1-99 are already used in the target folder."
        )
    return idx


def _folder_has_density_maps(folder):
    if not os.path.isdir(folder):
        return False
    return any(f.endswith(".npy") for f in os.listdir(folder))


def _density_filename(index, frame_key):
    if len(frame_key) == 0:
        return f"{index:02d}.npy"
    if len(frame_key) == 1:
        return f"{index:02d}_{frame_key[0]:02d}.npy"
    return "_".join(f"{k:02d}" for k in frame_key) + ".npy"


def _append_metadata(save_folder, label, lines):
    metadata_path = os.path.join(str(save_folder), "metadata.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"\n===== {label} | {timestamp} =====\n"
    with open(metadata_path, "a", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(lines))
        f.write("\n")


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

    m = re.search(
        r"^\s*pixel_size_um\s*=\s*([0-9]*\.?[0-9]+)\s*$",
        content,
        re.MULTILINE,
    )
    if m:
        try:
            value = float(m.group(1))
            if value > 0:
                pixel_um = value
        except ValueError:
            pass

    m = re.search(
        r"^\s*image_size\s*=\s*(\d+)\s*[xX,]\s*(\d+)\s*$",
        content,
        re.MULTILINE,
    )
    if m:
        image_h = int(m.group(1))
        image_w = int(m.group(2))

    return pixel_um, image_h, image_w


def _write_calibration(folder, pixel_um, image_h, image_w):
    path = os.path.join(str(folder), "calibration.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"pixel_size_um={pixel_um}\n")
        f.write(f"image_size={image_h}x{image_w}\n")


DEFAULT_CALIBRATION = _load_last_calibration()


@magic_factory(
    auto_call=False,
    call_button="Generate density maps",
    result_widget=True,
    Input_layer={"label": "Select Points or Shapes layer"},
    Reference_image={"label": "Reference image (defines shape)"},
    Density_size={
        "label": "Density map size (px per largest axis)",
        "min": 1,
        "max": 100000,
        "value": 50,
    },
    Pixel_calibration={
        "label": "Pixel calibration, µm/pixel side",
        "min": 0.0,
        "max": 1000.0,
        "step": 0.001,
        "value": DEFAULT_CALIBRATION,
    },
    Index={"label": "Index (1-99)", "min": 1, "max": 99, "value": 1},
    Save_folder={"mode": "d", "label": "Parent folder"},
    Subfolder_name={"label": "Subfolder name"},
)
def generate_density_maps(
    Input_layer: Layer,
    Reference_image: Image,
    Density_size: int = 50,
    Pixel_calibration: float = DEFAULT_CALIBRATION,
    Index: int = 1,
    Save_folder: pathlib.Path = pathlib.Path(),
    Subfolder_name: str = "DensityMaps",
) -> str:
    if not isinstance(Input_layer, Points | Shapes):
        show_modal_error(
            f"Input layer must be a Points or Shapes layer, got "
            f"{type(Input_layer).__name__}."
        )
        return "Invalid input layer."

    if not isinstance(Reference_image, Image):
        show_modal_error(
            f"Reference image must be an Image layer, got "
            f"{type(Reference_image).__name__}."
        )
        return "Invalid reference image."

    if not Save_folder:
        show_modal_error("Please select a parent folder.")
        return "No parent folder selected."

    if not Subfolder_name or not str(Subfolder_name).strip():
        show_modal_error("Please provide a subfolder name.")
        return "No subfolder name provided."

    if Density_size < 1:
        show_modal_error("Density_size must be >= 1.")
        return "Invalid Density_size."

    try:
        frame_shape, (H, W) = _extract_frame_and_spatial_shape(
            Reference_image.data.shape
        )
    except ValueError as e:
        show_modal_error(str(e))
        return "Invalid reference image shape."

    is_two_stack = len(frame_shape) == 2

    if not is_two_stack and not (1 <= Index <= 99):
        show_modal_error("Index must be in [1, 99].")
        return "Invalid index."

    subfolder = os.path.join(str(Save_folder), str(Subfolder_name))

    if is_two_stack and _folder_has_density_maps(subfolder):
        show_modal_error(
            "Given folder already contains density maps, can't process "
            "2-dimensional stack"
        )
        return "Folder already contains density maps."

    existing_pixel, existing_h, existing_w = _read_calibration(subfolder)
    field_calibration = float(Pixel_calibration)

    if existing_pixel is None:
        if field_calibration <= 0:
            show_modal_error("Please, provide pixel calibration")
            return "No pixel calibration provided."
        final_calibration = field_calibration
        calibration_action = "created"
        show_modal_warning(
            f"Attention! The calibration is set to {final_calibration}. "
            "Are you sure you didn't forget to change it for the new project?"
        )
    else:
        if field_calibration <= 0:
            final_calibration = existing_pixel
            calibration_action = "kept"
        elif field_calibration == existing_pixel:
            final_calibration = existing_pixel
            calibration_action = "unchanged"
        else:
            final_calibration = field_calibration
            calibration_action = "updated"
            show_modal_warning(
                f"Warning! Calibration was changed from "
                f"{existing_pixel} to {final_calibration}"
            )

    if (
        existing_h is not None
        and existing_w is not None
        and (existing_h != H or existing_w != W)
    ):
        show_modal_warning(
            f"Warning! Image size changed from "
            f"{existing_h}x{existing_w} to {H}x{W}"
        )

    data = Input_layer.data
    is_shapes = isinstance(Input_layer, Shapes)
    has_objects = data is not None and len(data) > 0

    if has_objects:
        if is_shapes:
            first = np.asarray(data[0])
            ndim = first.shape[1] if first.ndim == 2 else first.shape[0]
        else:
            ndim = np.asarray(data).shape[1]
    else:
        ndim = None

    expected_ndim = 2 + len(frame_shape)
    if ndim is not None and ndim != expected_ndim:
        kind = (
            "single 2D image"
            if len(frame_shape) == 0
            else f"{len(frame_shape)}-stack"
        )
        show_modal_error(
            f"Input layer has {ndim}D coordinates, but the reference "
            f"image is a {kind} (expects {expected_ndim}D coordinates)."
        )
        return "Mismatched dimensionality."

    map_h, map_w = _compute_map_shape(H, W, Density_size)

    y_edges = np.linspace(0.0, float(H), map_h + 1)
    x_edges = np.linspace(0.0, float(W), map_w + 1)

    if len(frame_shape) == 0:
        all_keys = [()]
    elif len(frame_shape) == 1:
        all_keys = [(f,) for f in range(frame_shape[0])]
    else:
        all_keys = [
            (d1, d2)
            for d1 in range(frame_shape[0])
            for d2 in range(frame_shape[1])
        ]
    key_set = set(all_keys)
    maps = {k: np.zeros((map_h, map_w), dtype=np.uint32) for k in all_keys}

    n_total = 0
    n_dropped_oob = 0
    n_dropped_oof = 0

    if has_objects:
        iterator = (
            _iter_shapes(data) if is_shapes else _iter_points(np.asarray(data))
        )
        for frame_key, y, x in iterator:
            n_total += 1
            if frame_key not in key_set:
                n_dropped_oof += 1
                continue
            if not (0.0 <= y < H and 0.0 <= x < W):
                n_dropped_oob += 1
                continue

            by = int(np.searchsorted(y_edges, y, side="right") - 1)
            bx = int(np.searchsorted(x_edges, x, side="right") - 1)
            if by < 0:
                by = 0
            elif by >= map_h:
                by = map_h - 1
            if bx < 0:
                bx = 0
            elif bx >= map_w:
                bx = map_w - 1

            maps[frame_key][by, bx] += 1

    os.makedirs(subfolder, exist_ok=True)

    if is_two_stack:
        final_index = None
        label = "2-stack"
        index_note = "Index field ignored for 2-stack."
    else:
        try:
            final_index = _find_next_index(subfolder, Index)
        except RuntimeError as e:
            show_modal_error(str(e))
            return "No free index available."

        if final_index != Index:
            show_info(
                f"Index {Index:02d} is already used in the target folder. "
                f"Falling back to index {final_index:02d}."
            )
        label = f"Index {final_index:02d}"
        index_note = None

    for key, m in maps.items():
        filename = _density_filename(final_index, key)
        np.save(os.path.join(subfolder, filename), m)

    if calibration_action in ("created", "updated"):
        _write_calibration(subfolder, final_calibration, H, W)

    _save_last_calibration(final_calibration)

    cell_h = H / map_h
    cell_w = W / map_w

    metadata_lines = [
        f"Source layer: {Input_layer.name} ({type(Input_layer).__name__})",
        f"Reference image: {Reference_image.name}",
        f"Image size (h x w): {H} x {W}",
        f"Requested Density_size: {Density_size}",
        f"Actual map shape (h, w): ({map_h}, {map_w})",
        f"Cell size along H (px): {cell_h:.4f}",
        f"Cell size along W (px): {cell_w:.4f}",
        f"Pixel calibration (µm/px): {final_calibration:.6g}",
        f"Calibration action: {calibration_action}",
        f"Requested index: {Index}",
        f"Used index: {final_index if final_index is not None else 'N/A'}",
        f"Total objects in layer: {n_total}",
        f"Objects dropped (out of spatial bounds): {n_dropped_oob}",
        f"Objects dropped (frame not in reference stack): {n_dropped_oof}",
        f"Number of density maps written: {len(maps)}",
    ]
    if index_note is not None:
        metadata_lines.append(f"Note: {index_note}")
    _append_metadata(subfolder, label, metadata_lines)

    if is_two_stack:
        summary = (
            f"Wrote {len(maps)} 2-stack map(s) of shape ({map_h}, {map_w}) "
            f"to {subfolder} (index field ignored)."
        )
    else:
        summary = (
            f"Used index {final_index:02d}. "
            f"Wrote {len(maps)} map(s) of shape ({map_h}, {map_w}) to "
            f"{subfolder}."
        )
    summary += (
        f" Pixel calibration: {final_calibration:.6g} µm/px "
        f"({calibration_action})."
    )
    if n_dropped_oob or n_dropped_oof:
        summary += (
            f" Dropped {n_dropped_oob} out-of-bounds and "
            f"{n_dropped_oof} out-of-frame object(s)."
        )
    show_info(summary)
    return summary
