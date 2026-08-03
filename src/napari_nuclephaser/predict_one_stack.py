import os
import pathlib
import re
import time
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

cuda_available = "cuda:0" if cuda.is_available() else "cpu"

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
    Generate_points={
        "tooltip": "If chosen, Points layer will be created with point at the center of bounding box for each detection"
    },
    Generate_bbox={
        "tooltip": "If chosen, Shapes layer will be created with rectangle representing bounding box of each detection"
    },
    Show_confidence={
        "tooltip": "If chosen, each rectangle in Shapes layer will have confidence score of each detection printed above it"
    },
    Bbox_thickness={
        "tooltip": "Thickness of the side of rectangles in Shapes layer if Generate bbox is chosen"
    },
    Score_text_size={
        "tooltip": "Font size of confidence score text if Show confidence parameter is chosen"
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
def predict_on_stack(
    Select_stack: Image,
    viewer: napari.Viewer,
    Select_model=first_model,
    Confidence_threshold: float = 0.2,
    Generate_points: bool = True,
    Generate_bbox: bool = False,
    Show_confidence: bool = False,
    Use_TTA=False,
    TTA_metadata_file=pathlib.Path(),
    Save_result=True,
    Save_folder=pathlib.Path(),
    Experiment_name="Experiment",
    ADVANCED_SETTINGS="",
    Postprocess="GREEDYNMM",
    Match_metric="IOS",
    Sahi_size=640,
    Sahi_overlap: float = 0.2,
    Intersection_threshold=0.3,
    Points_size=30,
    Bbox_thickness: int = 5,
    Score_text_size: int = 10,
    Save_csv=False,
    Save_xlsx=True,
):
    """Takes a 1-dimensional stack of images, YOLO model, and optional TTA metadata -> adds point/bbox layers and saves counts."""
    pic = Select_stack.data
    # Validate stack
    if len(pic.shape) == 2 or (
        len(pic.shape) == 3 and pic.shape[-1] in (1, 3, 4)
    ):
        show_error("Chosen image is a single frame, not a stack!")
        return None
    if (len(pic.shape) == 4 and pic.shape[-1] not in (1, 3, 4)) or len(
        pic.shape
    ) > 4:
        show_error("Chosen image has more dimensions than 1-stack!")
        return None

    is_gray = False
    if len(pic.shape) == 3:
        is_gray = True
    name = Select_stack.name
    print("Images stack is initialized successfully!")

    viewer.window._status_bar._toggle_activity_dock(True)

    # ---- Helper to create bounding box vertices in 3D (frame, y, x) ----
    def bbox_vertices(frame_idx, x1, y1, x2, y2):
        """Return 4 vertices of a rectangle in (z, y, x) order."""
        return np.array(
            [
                [frame_idx, y1, x1],
                [frame_idx, y1, x2],
                [frame_idx, y2, x2],
                [frame_idx, y2, x1],
            ]
        )

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

        try:
            best_augs, aug_thresholds, model_name_meta = _parse_metadata(
                str(TTA_metadata_file)
            )
        except (ValueError, OSError, KeyError) as e:
            show_error(f"Failed to parse metadata file: {e}")
            viewer.window._status_bar._toggle_activity_dock(False)
            return None

        selected_model_name = pathlib.Path(Select_model).name
        if model_name_meta and selected_model_name != model_name_meta:
            show_info(
                f"Warning: Selected model ({selected_model_name}) does not match model in metadata ({model_name_meta}). Continuing anyway."
            )

        # Storage per augmentation
        all_aug_points = []  # list of numpy arrays (N, 3) with (frame, y, x)
        all_aug_bboxes = []  # list of lists of 4x3 numpy arrays
        all_aug_scores = []  # list of lists of scores
        all_aug_counts = []  # list of lists of per-frame counts

        with progress(
            total=len(best_augs), desc="TTA augmentations"
        ) as pbar_augs:
            for aug_name in best_augs:
                if aug_name not in AUGMENTATION_MAP:
                    show_error(
                        f"Unknown augmentation '{aug_name}' in metadata. Skipping."
                    )
                    pbar_augs.update(1)
                    continue
                aug_func, scale_factor = AUGMENTATION_MAP[aug_name]
                thr = aug_thresholds.get(aug_name, Confidence_threshold)
                model, _ = initialize_model(
                    str(Select_model), thr, cuda_available
                )

                aug_points = []  # list of [frame, y, x]
                aug_bboxes = []  # list of 4x3 arrays
                aug_scores = []  # list of scores
                frame_counts = []

                with progress(
                    total=len(pic), desc=f"Processing frames ({aug_name})"
                ) as pbar_frames:
                    for i in range(len(pic)):
                        frame = pic[i]
                        if hasattr(frame, "compute"):
                            frame = frame.compute()
                        if frame.dtype == np.uint16:
                            frame = cv2.convertScaleAbs(
                                frame, alpha=255 / 65535
                            )
                            frame = frame.astype(np.uint8)
                        if is_gray:
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                        aug_frame = aug_func(frame)

                        pbar_slice = None

                        def slice_callback(
                            current, total, aug_name=aug_name, frame_idx=i
                        ):
                            nonlocal pbar_slice
                            if pbar_slice is None:
                                pbar_slice = progress(
                                    total=total,
                                    desc=f"Sliced prediction {frame_idx+1}",
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

                        frame_count = 0
                        for instance in result:
                            bbox = instance[
                                "bbox"
                            ]  # [x, y, w, h] in augmented image
                            score = instance["score"]
                            # Scale back to original image coordinates
                            if scale_factor != 1.0:
                                x = bbox[0] * scale_factor
                                y = bbox[1] * scale_factor
                                w = bbox[2] * scale_factor
                                h = bbox[3] * scale_factor
                            else:
                                x, y, w, h = bbox
                            x1, y1, x2, y2 = (
                                int(x),
                                int(y),
                                int(x + w),
                                int(y + h),
                            )
                            cx = int(x + w / 2)
                            cy = int(y + h / 2)
                            aug_points.append([i, cy, cx])  # (frame, row, col)
                            aug_bboxes.append(bbox_vertices(i, x1, y1, x2, y2))
                            aug_scores.append(score)
                            frame_count += 1
                        frame_counts.append(frame_count)
                        pbar_frames.update(1)

                all_aug_points.append(
                    np.array(aug_points) if aug_points else np.empty((0, 3))
                )
                all_aug_bboxes.append(aug_bboxes)
                all_aug_scores.append(aug_scores)
                all_aug_counts.append(frame_counts)
                pbar_augs.update(1)

        # Compute average counts per frame across augmentations
        n_frames = len(pic)
        avg_counts_per_frame = []
        for f in range(n_frames):
            counts = [
                aug_counts[f]
                for aug_counts in all_aug_counts
                if f < len(aug_counts)
            ]
            avg_counts_per_frame.append(np.mean(counts) if counts else 0)

        # Add layers for each augmentation
        for idx, aug_name in enumerate(best_augs):
            pts = all_aug_points[idx]
            bboxes = all_aug_bboxes[idx]
            scores = all_aug_scores[idx]
            n_det = len(pts)

            # Points
            if Generate_points or (not Generate_points and not Generate_bbox):
                if n_det > 0:
                    # pts is (N,3) -> (frame, y, x)
                    viewer.add_points(
                        pts,
                        size=Points_size,
                        name=f"{n_det} points ({aug_name}) {name}",
                    )
                else:
                    viewer.add_points(
                        np.empty((0, 3)),
                        size=Points_size,
                        name=f"0 points ({aug_name}) {name}",
                    )

            # Bounding boxes
            if Generate_bbox and n_det > 0:
                props = {"score": scores}
                text_kwargs = {}
                if Show_confidence:
                    text_kwargs = {
                        "text": {
                            "string": "{score:.2f}",
                            "size": Score_text_size,
                            "color": "red",
                            "anchor": "upper_left",
                            "translation": [-3, 0],
                        }
                    }
                viewer.add_shapes(
                    bboxes,
                    face_color="transparent",
                    edge_color="red",
                    edge_width=Bbox_thickness,
                    properties=props,
                    name=f"{n_det} bboxes ({aug_name}) {name}",
                    **text_kwargs,
                )

        # Save averaged results if requested
        if Save_result:
            subfolder = create_unique_subfolder(
                str(Save_folder), str(Experiment_name)
            )
            result_table = {
                "Frame": list(range(n_frames)),
                "Count": avg_counts_per_frame,
            }
            df = pd.DataFrame(result_table)
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

            current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
            metadata = f"""Experiment time: {current_date}
TTA prediction on stack
Stack napari name: {name}
Detection model: {Select_model}
Augmentations used: {' + '.join(best_augs)}
Per‑augmentation thresholds: {aug_thresholds}
Averaged counts per frame: {avg_counts_per_frame}
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

    # Accumulators for all detections across frames
    all_points = []  # list of [frame, y, x]
    all_bboxes = []  # list of 4x3 arrays
    all_scores = []  # list of scores
    result_table = {"Frame": [], "Count": []}

    print("Running predictions...")

    def make_slice_callback(frame_idx):
        pbar = None

        def callback(current: int, total: int):
            nonlocal pbar
            if pbar is None:
                pbar = progress(
                    total=total,
                    desc=f"Sliced prediction for frame {frame_idx+1}",
                )
            pbar.update(1)
            if current == total:
                pbar.close()

        return callback

    for i in progress(range(len(pic)), desc="Running predictions"):
        if i == 0:
            start_time = time.time()
        frame = pic[i]
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
            progress_callback=make_slice_callback(i),
        )
        result = result.to_coco_predictions()

        frame_count = 0
        for instance in result:
            bbox = instance["bbox"]  # [x, y, w, h]
            score = instance["score"]
            x1, y1, x2, y2 = (
                int(bbox[0]),
                int(bbox[1]),
                int(bbox[0] + bbox[2]),
                int(bbox[1] + bbox[3]),
            )
            cx = int(bbox[0] + bbox[2] / 2)
            cy = int(bbox[1] + bbox[3] / 2)
            all_points.append([i, cy, cx])  # (frame, row, col)
            all_bboxes.append(bbox_vertices(i, x1, y1, x2, y2))
            all_scores.append(score)
            frame_count += 1
        result_table["Frame"].append(i)
        result_table["Count"].append(frame_count)

        if i == 0:
            finish_time = time.time()
            frame_time = round(finish_time - start_time)
            print(f"First slice took {frame_time} seconds to process.")
            print(
                f"Processing whole stack will take approximately {frame_time * len(pic)} seconds"
            )
        print(f"Slice {i} is done!")

    # Add layers after processing all frames
    n_total = len(all_points)
    if Generate_points or (not Generate_points and not Generate_bbox):
        if n_total > 0:
            viewer.add_points(
                np.array(all_points),
                size=Points_size,
                name=f"{n_total} points {name}",
            )
        else:
            viewer.add_points(
                np.empty((0, 3)),
                size=Points_size,
                name=f"0 points {name}",
            )

    if Generate_bbox and n_total > 0:
        props = {"score": all_scores}
        text_kwargs = {}
        if Show_confidence:
            text_kwargs = {
                "text": {
                    "string": "{score:.2f}",
                    "size": Score_text_size,
                    "color": "red",
                    "anchor": "upper_left",
                    "translation": [-3, 0],
                }
            }
        viewer.add_shapes(
            all_bboxes,
            face_color="transparent",
            edge_color="red",
            edge_width=Bbox_thickness,
            properties=props,
            name=f"{n_total} bboxes {name}",
            **text_kwargs,
        )

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
Prediction on 1-stack
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
