import os
import pathlib
import re
import warnings
from datetime import datetime

import cv2
import napari
import numpy as np
import pandas as pd
from magicgui import magic_factory
from napari.layers import Image
from napari.utils import progress
from napari.utils.notifications import show_error, show_info
from sahi.predict import get_sliced_prediction
from torch import cuda

from napari_nuclephaser.utils import create_unique_subfolder, initialize_model

warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)

# cuda device check
cuda_available = "cuda:0" if cuda.is_available() else "cpu"

# find default models folder
models_folder = pathlib.Path(pathlib.Path(__file__).parent / "models")
first_model = next((x for x in models_folder.iterdir() if x.is_file()), None)


# ---------- Augmentation functions ----------
def _native(img):
    return img


def _resize_1_5(img):
    return cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)


def _resize_2(img):
    return cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


def _apply_clahe(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    clahe_gray = clahe.apply(gray)
    return cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2RGB)


def _adjust_gamma(img, gamma=1.5):
    if img.dtype == np.uint8:
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
        ).astype(np.uint8)
        return cv2.LUT(img, table)
    normalized = img.astype(np.float32) / 255.0
    corrected = np.power(normalized, 1.0 / gamma) * 255.0
    return corrected.astype(np.uint8)


def _invert_image(img):
    return 255 - img


def _median_filter_3(img):
    return cv2.medianBlur(img, 3)


def _bilateral_filter_10(img):
    return cv2.bilateralFilter(img, -1, 10, 10)


def _unsharp_mask(img, sigma=1.0, strength=1.5):
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return sharpened


AUGMENTATION_MAP = {
    "native": (_native, 1.0),
    "resize_1.5x": (_resize_1_5, 1 / 1.5),
    "resize_2x": (_resize_2, 0.5),
    "clahe": (_apply_clahe, 1.0),
    "gamma_1.5": (lambda x: _adjust_gamma(x, 1.5), 1.0),
    "invert": (_invert_image, 1.0),
    "median_3": (_median_filter_3, 1.0),
    "bilateral_10": (_bilateral_filter_10, 1.0),
    "sharpen": (_unsharp_mask, 1.0),
}
# ---------------------------------------------------------------------------


def _parse_metadata(metadata_path):
    """Parse metadata.txt file and return:
    - best_augmentations: list of augmentation names
    - aug_thresholds: dict {aug_name: threshold}
    - model_name: string (base name of model file)
    """
    with open(metadata_path, encoding="utf-8") as f:
        content = f.read()

    model_match = re.search(r"Phase model:\s+([^\s\(]+)", content)
    model_name = model_match.group(1) if model_match else None

    combo_match = re.search(r"Best TTA combination:\s+(.+)", content)
    if not combo_match:
        raise ValueError(
            "Metadata file does not contain 'Best TTA combination' line."
        )
    combo_str = combo_match.group(1).strip()
    best_augmentations = [aug.strip() for aug in combo_str.split("+")]

    thresholds = {}
    threshold_block_match = re.search(
        r"Per‑augmentation calibrated thresholds:\s*\n((?:  .+:\s+[\d\.]+\n?)+)",
        content,
    )
    if threshold_block_match:
        block = threshold_block_match.group(1)
        for line in block.split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split(":")
            if len(parts) == 2:
                aug_name = parts[0].strip()
                try:
                    thr = float(parts[1].strip())
                    thresholds[aug_name] = thr
                except ValueError:
                    pass
    else:
        raise ValueError(
            "Could not find per‑augmentation thresholds in metadata file."
        )

    return best_augmentations, thresholds, model_name


@magic_factory(
    Postprocess={
        "choices": ["GREEDYNMM", "NMS", "NMM"],
        "tooltip": "An algorithm to process overlapping detections. See obss/sahi library docs for more details.",
    },
    Match_metric={
        "choices": ["IOS", "IOU"],
        "tooltip": "A metric to determine when two detections are two different detections overlapping or is it a one detection. Sett obss/sahi library docs for more details",
    },
    Confidence_threshold={
        "tooltip": "Parameter that determines how many detections will model return. Use calibration widgets to determine optimal threshold for your use case."
    },
    Sahi_size={
        "max": 100000,
        "tooltip": "Slicing window inference slice. The large image will be divided into small ones with this size in pixels. See obss/sahi library for more details",
    },
    Sahi_overlap={
        "tooltip": "Relative overlap between sliding windows. See obss/sahi library docs for more details."
    },
    Intersection_threshold={
        "tooltip": "A metric to determine when to detections are overlapping. If metric is higher than threshold, detections will be merged. See obss/sahi library docs for more details."
    },
    Points_size={
        "tooltip": "Points size in results Points layer. Can be changed later by pressing Ctrl+A and moving Size slider in the layer itself"
    },
    Save_result={
        "tooltip": "If chosen, a folder will be created with .csv or .xlsx file containing quantification of objects for each frame"
    },
    Experiment_name={
        "tooltip": "Name of the subfolder that will be created for the results"
    },
    Save_csv={
        "tooltip": "If chosen, .csv format file with counting results will be saved at given folder"
    },
    Save_xlsx={
        "tooltip": "If chosen, .xlsx format file with counting results will be saved at given folder"
    },
    Use_TTA={
        "widget_type": "CheckBox",
        "tooltip": "Use test‑time augmentations defined in a calibration metadata file. Overrides Confidence_threshold and uses per‑augmentation thresholds from the file.",
    },
    TTA_metadata_file={
        "mode": "r",
        "filter": "*.txt",
        "tooltip": "Metadata .txt file generated by calibrate_points.py (containing best augmentation combination and thresholds).",
    },
    call_button="Predict",
    Save_folder={"mode": "d"},
    auto_call=False,
    result_widget=False,
)
def predict_on_two_stack(
    Select_stack: Image,
    viewer: napari.Viewer,
    Select_model=first_model,
    Confidence_threshold: float = 0.2,
    Use_TTA=False,
    TTA_metadata_file=pathlib.Path(),
    Save_result=True,
    Save_folder=pathlib.Path(),
    Experiment_name="Experiment",
    ADVANCED_SETTINGS="",
    Postprocess="NMS",
    Match_metric="IOS",
    Sahi_size=640,
    Sahi_overlap: float = 0.2,
    Intersection_threshold=0.34,
    Points_size=30,
    Save_csv=False,
    Save_xlsx=True,
):
    """Takes a 2-dimensional stack of images, YOLO model, and optional TTA metadata -> adds point layers and saves counts."""
    pic = Select_stack.data
    # Validate stack (must be 2D stack)
    if (
        len(pic.shape) == 2
        or (len(pic.shape) == 3)
        or (len(pic.shape) == 4 and pic.shape[-1] in (1, 3, 4))
    ):
        show_error(
            "Chosen image is a single frame or a 1-stack, not a 2-stack!"
        )
        return None
    if (len(pic.shape) == 5 and pic.shape[-1] not in (1, 3, 4)) or len(
        pic.shape
    ) > 5:
        show_error("Chosen image has more dimensions than 2-stack!")
        return None

    is_gray = False
    if len(pic.shape) == 4:  # shape (T1, T2, H, W) grayscale
        is_gray = True
    name = Select_stack.name
    print("Images stack is initialized successfully!")

    viewer.window._status_bar._toggle_activity_dock(True)

    # ---- TTA mode ----
    if Use_TTA:
        if not TTA_metadata_file or not TTA_metadata_file.exists():
            show_error("TTA metadata file not found or not provided.")
            viewer.window._status_bar._toggle_activity_dock(False)
            return None
        if TTA_metadata_file.suffix.lower() != ".txt":
            show_error("Selected file is not a .txt file.")
            viewer.window._status_bar._toggle_activity_dock(False)
            return None

        # Parse metadata
        try:
            best_augs, aug_thresholds, model_name_meta = _parse_metadata(
                str(TTA_metadata_file)
            )
        except (ValueError, OSError, KeyError) as e:
            show_error(f"Failed to parse metadata file: {e}")
            viewer.window._status_bar._toggle_activity_dock(False)
            return None

        # Validate model name
        selected_model_name = pathlib.Path(Select_model).name
        if model_name_meta and selected_model_name != model_name_meta:
            show_info(
                f"Warning: Selected model ({selected_model_name}) does not match model in metadata ({model_name_meta}). Continuing anyway."
            )

        # Determine dimensions
        dim1 = len(pic)
        dim2 = len(pic[0]) if dim1 > 0 else 0

        # For each augmentation, collect points (i, j, y, x) and per-frame counts (2D array)
        all_aug_points = []  # list of points arrays per augmentation
        all_aug_counts = []  # list of 2D count arrays per augmentation

        # Outer progress bar over augmentations
        with progress(
            total=len(best_augs), desc="TTA augmentations"
        ) as pbar_augs:
            for _, aug_name in enumerate(best_augs):
                if aug_name not in AUGMENTATION_MAP:
                    show_error(
                        f"Unknown augmentation '{aug_name}' in metadata. Skipping."
                    )
                    pbar_augs.update(1)
                    continue
                aug_func, scale_factor = AUGMENTATION_MAP[aug_name]
                thr = aug_thresholds.get(aug_name, Confidence_threshold)
                # Initialize model for this augmentation
                model, _ = initialize_model(
                    str(Select_model), thr, cuda_available
                )

                # Storage for this augmentation
                aug_points = []  # list of [i, j, y, x]
                aug_counts = (
                    np.zeros((dim1, dim2), dtype=int)
                    if dim1 > 0 and dim2 > 0
                    else None
                )

                # Process each frame (nested loops)
                with progress(
                    total=dim1, desc=f"Processing dim1 ({aug_name})"
                ) as pbar_i:
                    for i in range(dim1):
                        with progress(
                            total=dim2,
                            desc=f"Processing dim2 ({aug_name})",
                            leave=False,
                        ) as pbar_j:
                            for j in range(dim2):
                                frame = pic[i][j]
                                if hasattr(frame, "compute"):  # handle dask
                                    frame = frame.compute()
                                if frame.dtype == np.uint16:
                                    frame = cv2.convertScaleAbs(
                                        frame, alpha=255 / 65535
                                    )
                                    frame = frame.astype(np.uint8)
                                if is_gray:
                                    frame = cv2.cvtColor(
                                        frame, cv2.COLOR_GRAY2BGR
                                    )

                                # Apply augmentation
                                aug_frame = aug_func(frame)

                                # Sliced prediction (with nested progress callback)
                                pbar_slice = None

                                def slice_callback(
                                    current,
                                    total,
                                    aug_name=aug_name,
                                    frame_i=i,
                                    frame_j=j,
                                ):
                                    nonlocal pbar_slice
                                    if pbar_slice is None:
                                        pbar_slice = progress(
                                            total=total,
                                            desc=f"Sliced {aug_name} frame ({frame_i+1},{frame_j+1})",  # use bound variables
                                        )
                                    pbar_slice.update(1)
                                    if current == total:
                                        pbar_slice.close()

                                result = get_sliced_prediction(
                                    aug_frame,
                                    model,
                                    slice_height=Sahi_size,
                                    slice_width=Sahi_size,
                                    overlap_height_ratio=Sahi_overlap,
                                    overlap_width_ratio=Sahi_overlap,
                                    postprocess_type=Postprocess,
                                    postprocess_match_metric=Match_metric,
                                    postprocess_match_threshold=Intersection_threshold,
                                    verbose=0,
                                    force_postprocess_type=True,
                                    progress_bar=False,
                                    progress_callback=slice_callback,
                                )
                                result = result.to_coco_predictions()

                                # Convert detections
                                frame_count = 0
                                for instance in result:
                                    bbox = instance[
                                        "bbox"
                                    ]  # [x, y, width, height] in augmented image
                                    # Center point
                                    if scale_factor != 1.0:
                                        center_x = int(
                                            (bbox[0] + bbox[2] // 2)
                                            * scale_factor
                                        )
                                        center_y = int(
                                            (bbox[1] + bbox[3] // 2)
                                            * scale_factor
                                        )
                                    else:
                                        center_x = int(bbox[0] + bbox[2] // 2)
                                        center_y = int(bbox[1] + bbox[3] // 2)
                                    aug_points.append(
                                        [i, j, center_y, center_x]
                                    )  # (i, j, y, x)
                                    frame_count += 1
                                if aug_counts is not None:
                                    aug_counts[i, j] = frame_count
                                pbar_j.update(1)
                        pbar_i.update(1)

                all_aug_points.append(
                    np.array(aug_points) if aug_points else np.empty((0, 4))
                )
                all_aug_counts.append(aug_counts)
                pbar_augs.update(1)

        # Compute average counts per frame across augmentations
        avg_counts = np.zeros((dim1, dim2))
        for f in range(dim1):
            for g in range(dim2):
                counts = [
                    aug_counts[f, g]
                    for aug_counts in all_aug_counts
                    if aug_counts is not None
                ]
                avg_counts[f, g] = np.mean(counts) if counts else 0

        # Create separate points layers per augmentation
        for _, (aug_name, aug_points) in enumerate(
            zip(best_augs, all_aug_points, strict=False)
        ):
            if len(aug_points) > 0:
                viewer.add_points(
                    aug_points,  # shape (N, 4) with columns (i, j, y, x)
                    size=Points_size,
                    name=f"{len(aug_points)} points ({aug_name}) {name}",
                )
            else:
                viewer.add_points(
                    np.empty((0, 4)),
                    size=Points_size,
                    name=f"0 points ({aug_name}) {name}",
                )

        # Save averaged results if requested
        if Save_result:
            subfolder = create_unique_subfolder(
                str(Save_folder), str(Experiment_name)
            )
            # Build table: each row is (Dimension 1 frame, Dimension 2 frame, Count)
            rows = []
            for i in range(dim1):
                for j in range(dim2):
                    rows.append(
                        {
                            "Dimension 1 frame": i,
                            "Dimension 2 frame": j,
                            "Count": avg_counts[i, j],
                        }
                    )
            df = pd.DataFrame(rows)
            if Save_csv:
                df.to_csv(
                    os.path.join(subfolder, f"{name}_TTA_averaged_counts.csv"),
                    index=False,
                )
            if Save_xlsx:
                df.to_excel(
                    os.path.join(
                        subfolder, f"{name}_TTA_averaged_counts.xlsx"
                    ),
                    index=False,
                )
            if not Save_csv and not Save_xlsx:
                df.to_csv(
                    os.path.join(subfolder, f"{name}_TTA_averaged_counts.csv"),
                    index=False,
                )

            # Save metadata
            current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
            metadata = f"""Experiment time: {current_date}
TTA prediction on 2-stack
Stack napari name: {name}
Detection model: {Select_model}
Augmentations used: {' + '.join(best_augs)}
Per‑augmentation thresholds: {aug_thresholds}
Averaged counts per frame: {avg_counts.tolist()}
SAHI parameters used: size={Sahi_size}, overlap={Sahi_overlap}, postprocess={Postprocess}, match_metric={Match_metric}, iou_thr={Intersection_threshold}
"""
            metadata_path = os.path.join(subfolder, f"{name}_TTA_metadata.txt")
            with open(metadata_path, "w", encoding="utf-8") as f:
                f.write(metadata)
            show_info(f"TTA results saved in {subfolder}")
        else:
            show_info("TTA inference complete (results not saved).")

        viewer.window._status_bar._toggle_activity_dock(False)
        return

    # ---- Original (non‑TTA) mode ----
    print("Initializing model...")
    detection_model, model_type = initialize_model(
        rf"{Select_model}", Confidence_threshold, cuda_available
    )
    print(
        f"Model is initialized! Model type is {model_type}. Running on {cuda_available}"
    )

    points = []
    result_table = {
        "Dimension 1 frame": [],
        "Dimension 2 frame": [],
        "Count": [],
    }

    print("Running predictions...")

    # Helper to create a progress callback for each frame
    def make_slice_callback(frame_idx_i, frame_idx_j):
        pbar = None

        def callback(current: int, total: int):
            nonlocal pbar
            if pbar is None:
                pbar = progress(
                    total=total,
                    desc=f"Sliced prediction for frame ({frame_idx_i+1}, {frame_idx_j+1})",
                )
            pbar.update(1)
            if current == total:
                pbar.close()

        return callback

    for i in progress(range(len(pic)), desc="Loop through dimension 1"):
        for j in progress(range(len(pic[i])), desc="Loop through dimension 2"):
            frame = pic[i][j]
            if hasattr(frame, "compute"):
                frame = frame.compute()
            if frame.dtype == np.uint16:
                frame = cv2.convertScaleAbs(frame, alpha=255 / 65535)
                frame = frame.astype(np.uint8)
            if is_gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            result = get_sliced_prediction(
                frame,
                detection_model,
                slice_height=Sahi_size,
                slice_width=Sahi_size,
                overlap_height_ratio=Sahi_overlap,
                overlap_width_ratio=Sahi_overlap,
                postprocess_type=Postprocess,
                postprocess_match_metric=Match_metric,
                postprocess_match_threshold=Intersection_threshold,
                verbose=0,
                force_postprocess_type=True,
                progress_bar=False,
                progress_callback=make_slice_callback(i, j),
            )
            result = result.to_coco_predictions()
            for instance in result:
                bbox = instance["bbox"]
                Y, X = int(bbox[0] + (bbox[2] // 2)), int(
                    bbox[1] + (bbox[3] // 2)
                )
                points.append([i, j, X, Y])
            result_table["Dimension 1 frame"].append(i)
            result_table["Dimension 2 frame"].append(j)
            result_table["Count"].append(len(result))

    viewer.add_points(points, size=Points_size, name=f"Points for {name}")
    viewer.window._status_bar._toggle_activity_dock(False)
    print("Prediction is complete!")

    if Save_result:
        print("Saving results...")
        subfolder = create_unique_subfolder(
            str(Save_folder), str(Experiment_name)
        )
        df = pd.DataFrame.from_dict(result_table)
        if Save_csv:
            df.to_csv(
                os.path.join(subfolder, f"{name} count results.csv"),
                index=False,
            )
            print(".csv file created successfully")
        if Save_xlsx:
            df.to_excel(
                os.path.join(subfolder, f"{name} count results.xlsx"),
                index=False,
            )
            print(".xlsx file created successfully")
        if not Save_csv and not Save_xlsx:
            df.to_csv(
                os.path.join(subfolder, f"{name} count results.csv"),
                index=False,
            )
            print(
                "None of the options are chosen, creating .csv file as a default"
            )

        print("Creating metadata file...")
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metadata = f"""Experiment time: {current_date}
Prediction on 2-stack widget
Stack napari name: {name}
Detection_model: {Select_model}
Model type: {model_type}
Confidence threshold: {Confidence_threshold}
Postprocess algorithm: {Postprocess}
Match metric: {Match_metric}
Intersection threshold: {Intersection_threshold}
SAHI size: {Sahi_size}
SAHI overlap: {Sahi_overlap}"""
        metadata_path = os.path.join(subfolder, f"{name} count metadata.txt")
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(metadata)
        print("Metadata file is saved!")

    show_info("Made predictions for stack successfully!")
