import os
import pathlib
from datetime import datetime

import numpy as np
import pandas as pd
from magicgui import magic_factory
from napari.layers import Points
from napari.utils.notifications import show_error, show_info

from napari_nuclephaser.utils import (
    create_unique_subfolder,  # reuse existing helper
)


@magic_factory(
    auto_call=False,
    call_button="Count",
    result_widget=True,
    Points_layer={"label": "Select points layer"},
    Save_result={"tooltip": "Save count results to a folder"},
    Experiment_name={"tooltip": "Subfolder name for the results"},
    Save_csv={"tooltip": "Save results as CSV"},
    Save_xlsx={"tooltip": "Save results as Excel"},
    Save_folder={"mode": "d", "tooltip": "Folder where results will be saved"},
)
def count_points_in_stack(
    Points_layer: Points,
    Save_result: bool = True,
    Save_folder: pathlib.Path = pathlib.Path(),
    Experiment_name: str = "PointsCount",
    Save_csv: bool = False,
    Save_xlsx: bool = True,
) -> str:
    """
    Count points in a Points layer that represents a single image, a 1‑dimensional stack,
    or a 2‑dimensional stack. Points coordinates are assumed to be:
      - (y, x) for a single image,
      - (frame, y, x) for a 1‑stack,
      - (dim1, dim2, y, x) for a 2‑stack.
    The function returns a table of counts per frame (or per (dim1, dim2) pair) and
    optionally saves it as CSV/XLSX.
    """
    points_data = Points_layer.data

    if points_data is None or len(points_data) == 0:
        show_error("The selected Points layer is empty.")
        return "No points to count."

    # Determine dimensionality
    ndim = points_data.shape[1] if len(points_data.shape) == 2 else 0
    if ndim not in (2, 3, 4):
        show_error(
            f"Unsupported point dimensionality: {ndim}. "
            "Expected 2 (single image), 3 (1‑stack) or 4 (2‑stack)."
        )
        return "Invalid point data shape."

    if ndim == 2:
        # Single image: total count
        total = len(points_data)
        result_table = {"Frame": [0], "Count": [total]}
        summary = f"Total points: {total}"
    elif ndim == 3:
        # 1‑dimensional stack: group by first coordinate (frame index)
        frames = points_data[:, 0].astype(int)
        unique_frames, counts = np.unique(frames, return_counts=True)
        result_table = {"Frame": unique_frames, "Count": counts}
        summary = f"Counted points across {len(unique_frames)} frames."
    else:  # ndim == 4
        # 2‑dimensional stack: group by first two coordinates (dim1, dim2)
        dim1 = points_data[:, 0].astype(int)
        dim2 = points_data[:, 1].astype(int)
        # Use pandas for easy groupby
        df_temp = pd.DataFrame({"dim1": dim1, "dim2": dim2})
        grouped = (
            df_temp.groupby(["dim1", "dim2"]).size().reset_index(name="Count")
        )
        result_table = {
            "Dimension 1 frame": grouped["dim1"].values,
            "Dimension 2 frame": grouped["dim2"].values,
            "Count": grouped["Count"].values,
        }
        summary = f"Counted points across {len(grouped)} (dim1, dim2) pairs."

    # Show info in napari
    show_info(summary)

    # Save results if requested
    if Save_result:
        if not Save_folder:
            Save_folder = pathlib.Path.cwd()
        subfolder = create_unique_subfolder(
            str(Save_folder), str(Experiment_name)
        )
        df = pd.DataFrame.from_dict(result_table)
        name = Points_layer.name

        if Save_csv:
            csv_path = os.path.join(subfolder, f"{name}_counts.csv")
            df.to_csv(csv_path, index=False)
            show_info(f"Saved CSV to {csv_path}")
        if Save_xlsx:
            xlsx_path = os.path.join(subfolder, f"{name}_counts.xlsx")
            df.to_excel(xlsx_path, index=False)
            show_info(f"Saved Excel to {xlsx_path}")
        if not Save_csv and not Save_xlsx:
            # default to CSV
            csv_path = os.path.join(subfolder, f"{name}_counts.csv")
            df.to_csv(csv_path, index=False)
            show_info(f"Saved CSV (default) to {csv_path}")

        # Save a simple metadata file
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metadata = f"""Experiment time: {current_date}
Points count widget
Points layer name: {name}
Point dimensionality: {ndim}
Number of points: {len(points_data)}
"""
        metadata_path = os.path.join(subfolder, f"{name}_metadata.txt")
        with open(metadata_path, "w") as f:
            f.write(metadata)
        show_info("Metadata saved.")

    # Return a string summary that will appear in the magicgui result widget
    return summary
