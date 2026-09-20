import os
import pathlib

import numpy as np
import pandas as pd
from magicgui import magic_factory
from napari.layers import Layer, Points, Shapes
from napari.utils.notifications import show_error, show_info

from napari_nuclephaser.utils import (
    create_unique_subfolder,  # reuse existing helper
)


def _count_points(points_data):
    """Count Points layer data. Returns (result_table, summary)."""
    if len(points_data) == 0:
        return {"Frame": [], "Count": []}, "No points."

    ndim = points_data.shape[1]

    if ndim == 2:
        total = len(points_data)
        return {"Frame": [0], "Count": [total]}, f"Total points: {total}"
    elif ndim == 3:
        frames = points_data[:, 0].astype(int)
        unique_frames, counts = np.unique(frames, return_counts=True)
        return {
            "Frame": unique_frames.tolist(),
            "Count": counts.tolist(),
        }, f"Counted points across {len(unique_frames)} frames."
    elif ndim == 4:
        dim1 = points_data[:, 0].astype(int)
        dim2 = points_data[:, 1].astype(int)
        df_temp = pd.DataFrame({"dim1": dim1, "dim2": dim2})
        grouped = (
            df_temp.groupby(["dim1", "dim2"]).size().reset_index(name="Count")
        )
        return {
            "Dimension 1 frame": grouped["dim1"].tolist(),
            "Dimension 2 frame": grouped["dim2"].tolist(),
            "Count": grouped["Count"].tolist(),
        }, f"Counted points across {len(grouped)} (dim1, dim2) pairs."
    else:
        raise ValueError(
            f"Unsupported point dimensionality: {ndim}. "
            "Expected 2 (single image), 3 (1‑stack) or 4 (2‑stack)."
        )


def _count_shapes(shapes_data):
    """Count Shapes layer data (list of vertex arrays).

    The frame indices are read from the first vertex of each shape, using
    the same convention as Points layers: the last two axes are spatial
    (y, x); all leading axes are stack indices.
    """
    if len(shapes_data) == 0:
        return {"Frame": [], "Count": []}, "No boxes."

    # Determine ndim from the first shape
    first = np.asarray(shapes_data[0])
    # Single-vertex shape (ndim==1) is unlikely for boxes, but possible
    ndim = first.shape[0] if first.ndim == 1 else first.shape[1]

    def _first_vertex_frame_indices(shape):
        shape_arr = np.asarray(shape)
        coords = shape_arr if shape_arr.ndim == 1 else shape_arr[0]
        if ndim <= 2:
            return (0,)
        return tuple(int(coords[k]) for k in range(ndim - 2))

    if ndim == 2:
        total = len(shapes_data)
        return {"Frame": [0], "Count": [total]}, f"Total boxes: {total}"
    elif ndim == 3:
        frame_indices = np.array(
            [_first_vertex_frame_indices(s)[0] for s in shapes_data]
        )
        unique_frames, counts = np.unique(frame_indices, return_counts=True)
        return {
            "Frame": unique_frames.tolist(),
            "Count": counts.tolist(),
        }, f"Counted boxes across {len(unique_frames)} frames."
    elif ndim == 4:
        pairs = [_first_vertex_frame_indices(s) for s in shapes_data]
        dim1_list = [p[0] for p in pairs]
        dim2_list = [p[1] for p in pairs]
        df_temp = pd.DataFrame({"dim1": dim1_list, "dim2": dim2_list})
        grouped = (
            df_temp.groupby(["dim1", "dim2"]).size().reset_index(name="Count")
        )
        return {
            "Dimension 1 frame": grouped["dim1"].tolist(),
            "Dimension 2 frame": grouped["dim2"].tolist(),
            "Count": grouped["Count"].tolist(),
        }, f"Counted boxes across {len(grouped)} (dim1, dim2) pairs."
    else:
        raise ValueError(
            f"Unsupported shape dimensionality: {ndim}. "
            "Expected 2 (single image), 3 (1‑stack) or 4 (2‑stack)."
        )


@magic_factory(
    auto_call=False,
    call_button="Count",
    result_widget=True,
    Input_layer={"label": "Select Points or Shapes layer"},
    Save_result={"tooltip": "Save count results to a folder"},
    Experiment_name={"tooltip": "Subfolder name for the results"},
    Save_csv={"tooltip": "Save results as CSV"},
    Save_xlsx={"tooltip": "Save results as Excel"},
    Save_folder={"mode": "d", "tooltip": "Folder where results will be saved"},
)
def count_points_in_stack(
    Input_layer: Layer,
    Save_result: bool = True,
    Save_folder: pathlib.Path = pathlib.Path(),
    Experiment_name: str = "PointsCount",
    Save_csv: bool = False,
    Save_xlsx: bool = True,
) -> str:
    """
    Count points (Points layer) or boxes (Shapes layer) that represent a
    single image, a 1‑dimensional stack, or a 2‑dimensional stack.

    Coordinate layout is assumed to be:
      - (y, x) for a single image,
      - (frame, y, x) for a 1‑stack,
      - (dim1, dim2, y, x) for a 2‑stack.

    For Shapes layers, each shape is assigned to a frame using the leading
    (non‑spatial) coordinates of its first vertex.

    Returns a summary string; optionally saves a per‑frame count table
    as CSV and/or XLSX.
    """
    if not isinstance(Input_layer, Points | Shapes):
        show_error(
            "Please select a Points or Shapes layer. "
            f"Got: {type(Input_layer).__name__}."
        )
        return "Invalid layer type."

    data = Input_layer.data
    layer_name = Input_layer.name

    if data is None or len(data) == 0:
        show_error("The selected layer is empty.")
        return "Nothing to count."

    is_shapes = isinstance(Input_layer, Shapes)

    try:
        if is_shapes:
            result_table, summary = _count_shapes(data)
        else:
            result_table, summary = _count_points(np.asarray(data))
    except ValueError as e:
        show_error(str(e))
        return "Invalid data shape."

    show_info(summary)

    if Save_result:
        if not Save_folder:
            Save_folder = pathlib.Path.cwd()
        subfolder = create_unique_subfolder(
            str(Save_folder), str(Experiment_name)
        )
        df = pd.DataFrame.from_dict(result_table)

        if Save_csv:
            csv_path = os.path.join(subfolder, f"{layer_name}_counts.csv")
            df.to_csv(csv_path, index=False)
            show_info(f"Saved CSV to {csv_path}")
        if Save_xlsx:
            xlsx_path = os.path.join(subfolder, f"{layer_name}_counts.xlsx")
            df.to_excel(xlsx_path, index=False)
            show_info(f"Saved Excel to {xlsx_path}")
        if not Save_csv and not Save_xlsx:
            # default to CSV
            csv_path = os.path.join(subfolder, f"{layer_name}_counts.csv")
            df.to_csv(csv_path, index=False)
            show_info(f"Saved CSV (default) to {csv_path}")

    return summary
