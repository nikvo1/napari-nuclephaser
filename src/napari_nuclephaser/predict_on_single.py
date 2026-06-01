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

from napari_nuclephaser.utils import initialize_model

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


# Mapping from augmentation name (as stored in metadata) to function and scaling factor
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
    - best_augmentations: list of augmentation names (e.g. ['native', 'median_3'])
    - aug_thresholds: dict {aug_name: threshold}
    - model_name: string (base name of model file)
    """
    with open(metadata_path, encoding="utf-8") as f:
        content = f.read()

    # Extract model name (e.g. "ND_v11n.pt")
    model_match = re.search(r"Phase model:\s+([^\s\(]+)", content)
    model_name = model_match.group(1) if model_match else None

    # Extract best combination line
    combo_match = re.search(r"Best TTA combination:\s+(.+)", content)
    if not combo_match:
        raise ValueError(
            "Metadata file does not contain 'Best TTA combination' line."
        )
    combo_str = combo_match.group(1).strip()
    best_augmentations = [aug.strip() for aug in combo_str.split("+")]

    # Extract per‑augmentation thresholds
    thresholds = {}
    # Find the block starting with "Per‑augmentation calibrated thresholds:"
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
    Use_TTA={
        "widget_type": "CheckBox",
        "tooltip": "Use test‑time augmentations defined in a calibration metadata file. Overrides Confidence_threshold and uses per‑augmentation thresholds from the file.",
    },
    TTA_metadata_file={
        "mode": "r",
        "filter": "*.txt",
        "tooltip": "Metadata .txt file generated by calibrate_points.py (containing best augmentation combination and thresholds).",
    },
    ADVANCED_SETTINGS={},
    Generate_points={
        "tooltip": "If chosen, Points layer will be created with point at the center of bounding box for each detection"
    },
    Generate_bbox={
        "tooltip": "If chosen, Shapes layer will be created with rectangle representing bounding box of each detection"
    },
    Show_confidence={
        "tooltip": "If chosen, each rectangle in Shapes layer will have confidence score of each detection printed above it"
    },
    Confidence_threshold={
        "tooltip": "Parameter that determines how many detections will model return. Use calibration widgets to determine optimal threshold for your use case."
    },
    Sahi_size={
        "max": 100000,  # Default setting creates limit at 1000, this prevents it
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
    Bbox_thickness={
        "tooltip": "Thickness of the side of rectangles in Shapes layer if Generate bbox is chosen"
    },
    Score_text_size={
        "tooltip": "Font size of confidence score text if Show confidence parameter is chosen"
    },
    Save_result={
        "tooltip": "If chosen, a folder will be created with .csv or .xlsx file containing averaged detection counts across augmentations (TTA mode only)."
    },
    Experiment_name={
        "tooltip": "Name of the subfolder that will be created for the results (TTA mode only)."
    },
    Save_csv={
        "tooltip": "If chosen, .csv format file with counting results will be saved (TTA mode only)."
    },
    Save_xlsx={
        "tooltip": "If chosen, .xlsx format file with counting results will be saved (TTA mode only)."
    },
    Save_folder={"mode": "d"},
    call_button="Predict",
    auto_call=False,
    result_widget=False,
)
def make_points(
    Select_image: Image,
    viewer: napari.Viewer,
    Select_model=first_model,
    Confidence_threshold: float = 0.2,
    Generate_points=True,
    Generate_bbox=False,
    Show_confidence=False,
    Use_TTA=False,
    TTA_metadata_file=pathlib.Path(),
    Save_result=False,
    Save_folder=pathlib.Path(),
    Experiment_name="Experiment",
    Save_csv=False,
    Save_xlsx=True,
    ADVANCED_SETTINGS="",
    Postprocess="GREEDYNMM",
    Match_metric="IOS",
    Sahi_size=640,
    Sahi_overlap: float = 0.2,
    Intersection_threshold=0.3,
    Points_size=10,
    Bbox_thickness=5,
    Score_text_size=3,
) -> napari.types.LayerDataTuple:
    """Takes a single-frame image, YOLO model, and optional TTA metadata -> adds point/bbox layers and saves averaged counts."""
    pic = Select_image.data
    if len(pic.shape) == 2:
        pic = cv2.cvtColor(pic, cv2.COLOR_GRAY2RGB)
    if len(pic.shape) > 3 or (
        len(pic.shape) == 3 and pic.shape[-1] not in (1, 3, 4)
    ):
        show_error(
            "Image is not a single frame! Choose different widget for processing stacks of images"
        )
        return None
    name = Select_image.name
    if pic.dtype == np.uint16:
        pic = cv2.convertScaleAbs(pic, alpha=255 / 65535)
        pic = pic.astype(np.uint8)

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

        # For each augmentation in best combination, run inference
        all_counts = (
            []
        )  # store detection count per augmentation (for averaging)
        # Outer progress bar over augmentations
        with progress(
            total=len(best_augs), desc="TTA augmentations"
        ) as pbar_augs:
            for aug_name in best_augs:
                if aug_name not in AUGMENTATION_MAP:
                    show_error(
                        f"Unknown augmentation '{aug_name}' in metadata. Skipping this augmentation."
                    )
                    pbar_augs.update(1)
                    continue
                aug_func, scale_factor = AUGMENTATION_MAP[aug_name]
                # Get threshold for this augmentation from metadata
                thr = aug_thresholds.get(
                    aug_name, Confidence_threshold
                )  # fallback to user's threshold if missing
                # Initialize model with this threshold (temporary)
                model, _ = initialize_model(
                    str(Select_model), thr, cuda_available
                )

                # Apply augmentation to whole image
                aug_img = aug_func(pic)
                # Run sliced prediction with user's SAHI settings
                # Create a progress callback for sliced prediction (nested)
                pbar_inner = None

                def progress_callback(current, total, aug_name=aug_name):
                    nonlocal pbar_inner
                    if pbar_inner is None:
                        pbar_inner = progress(
                            total=total, desc=f"Sliced prediction ({aug_name})"
                        )
                    pbar_inner.update(1)
                    if current == total:
                        pbar_inner.close()

                result = get_sliced_prediction(
                    aug_img,
                    model,
                    slice_height=Sahi_size,
                    slice_width=Sahi_size,
                    overlap_height_ratio=Sahi_overlap,
                    overlap_width_ratio=Sahi_overlap,
                    postprocess_type=Postprocess,
                    postprocess_match_metric=Match_metric,
                    postprocess_match_threshold=Intersection_threshold,
                    force_postprocess_type=True,
                    progress_bar=False,
                    progress_callback=progress_callback,
                )
                result = result.to_coco_predictions()
                # Transform points and bounding boxes back to original coordinates
                points_aug = []
                bboxes_aug = []
                scores_aug = []
                for instance in result:
                    bbox = instance[
                        "bbox"
                    ]  # [x, y, width, height] in augmented image
                    score = instance["score"]
                    scores_aug.append(score)

                    # Center point
                    if scale_factor != 1.0:
                        center_x = int((bbox[0] + bbox[2] // 2) * scale_factor)
                        center_y = int((bbox[1] + bbox[3] // 2) * scale_factor)
                    else:
                        center_x = int(bbox[0] + bbox[2] // 2)
                        center_y = int(bbox[1] + bbox[3] // 2)
                    points_aug.append(
                        [center_y, center_x]
                    )  # napari uses (y, x)

                    # Bounding box rectangle
                    if scale_factor != 1.0:
                        x_orig = bbox[0] * scale_factor
                        y_orig = bbox[1] * scale_factor
                        w_orig = bbox[2] * scale_factor
                        h_orig = bbox[3] * scale_factor
                    else:
                        x_orig = bbox[0]
                        y_orig = bbox[1]
                        w_orig = bbox[2]
                        h_orig = bbox[3]

                    # Convert to napari polygon points (each point is (y, x))
                    y1 = int(x_orig)  # row of top-left
                    x1 = int(y_orig)  # col of top-left
                    y2 = int(x_orig + w_orig)  # row of bottom-right
                    x2 = int(y_orig + h_orig)  # col of bottom-right
                    polygon = np.array(
                        [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
                    )
                    bboxes_aug.append(polygon)

                n_cells = len(points_aug)

                # Add points layer if requested
                if Generate_points or (
                    not Generate_points and not Generate_bbox
                ):
                    viewer.add_points(
                        np.array(points_aug),
                        size=Points_size,
                        name=f"{n_cells} points ({aug_name}) {name}",
                    )

                # Add bounding boxes layer if requested
                if Generate_bbox:
                    properties = {"score": scores_aug}
                    if Show_confidence:
                        text_parameters = {
                            "string": "{score:.2f}",
                            "size": Score_text_size,
                            "color": "red",
                            "anchor": "upper_left",
                            "translation": [-3, 0],
                        }
                        viewer.add_shapes(
                            bboxes_aug,
                            face_color="transparent",
                            edge_color="red",
                            edge_width=Bbox_thickness,
                            properties=properties,
                            text=text_parameters,
                            name=f"{n_cells} bounding boxes ({aug_name}) {name}",
                        )
                    else:
                        viewer.add_shapes(
                            bboxes_aug,
                            face_color="transparent",
                            edge_color="red",
                            edge_width=Bbox_thickness,
                            properties=properties,
                            name=f"{n_cells} bounding boxes ({aug_name}) {name}",
                        )

                all_counts.append(n_cells)
                pbar_augs.update(1)

        # Compute average count across augmentations
        avg_count = np.mean(all_counts) if all_counts else 0
        # Save results if requested
        if Save_result:
            from napari_nuclephaser.utils import create_unique_subfolder

            subfolder = create_unique_subfolder(
                str(Save_folder), str(Experiment_name)
            )
            # Create dataframe with a single row (since it's a single image)
            result_table = {"Frame": [name], "Count": [avg_count]}
            df = pd.DataFrame.from_dict(result_table)
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
TTA prediction on single image
Image napari name: {name}
Detection model: {Select_model}
Augmentations used: {' + '.join(best_augs)}
Per‑augmentation thresholds: {aug_thresholds}
Averaged detection count: {avg_count:.2f}
SAHI parameters used: size={Sahi_size}, overlap={Sahi_overlap}, postprocess={Postprocess}, match_metric={Match_metric}, iou_thr={Intersection_threshold}
"""
            metadata_path = os.path.join(subfolder, f"{name}_TTA_metadata.txt")
            with open(metadata_path, "w", encoding="utf-8") as f:
                f.write(metadata)
            show_info(f"TTA results saved in {subfolder}")
        else:
            show_info(
                f"TTA inference complete. Averaged detection count = {avg_count:.2f}"
            )

        viewer.window._status_bar._toggle_activity_dock(False)
        return None

    # ---- Original (non‑TTA) mode ----
    initialization_pbar = progress(total=1, desc="Initializing model")
    detection_model, model_type = initialize_model(
        rf"{Select_model}", Confidence_threshold, cuda_available
    )
    print(
        f"Model is initialized! Model type is {model_type}. Running on {cuda_available}"
    )
    initialization_pbar.close()
    print("Performing sliced prediction...")

    pbar = None

    def progress_callback(current: int, total: int):
        nonlocal pbar
        if pbar is None:
            pbar = progress(total=total, desc="Processing slices")
        pbar.update(1)
        if current == total:
            pbar.close()
            # Note: we don't create a new progress bar for postprocessing here

    result = get_sliced_prediction(
        pic,
        detection_model,
        slice_height=Sahi_size,
        slice_width=Sahi_size,
        overlap_height_ratio=Sahi_overlap,
        overlap_width_ratio=Sahi_overlap,
        postprocess_type=Postprocess,
        postprocess_match_metric=Match_metric,
        postprocess_match_threshold=Intersection_threshold,
        force_postprocess_type=True,
        progress_bar=False,
        progress_callback=progress_callback,
    )
    result = result.to_coco_predictions()
    print("Prediction is done!")

    def create_points(result):
        points = []
        for instance in result:
            bbox = instance["bbox"]
            Y, X = int(bbox[0] + (bbox[2] // 2)), int(bbox[1] + (bbox[3] // 2))
            points.append([X, Y])
        n_cells = len(points)
        points = np.array(points)
        viewer.add_points(
            points, size=Points_size, name=f"{n_cells} points {name}"
        )
        return points, n_cells

    def create_bbox(result):
        bboxes = []
        scores = []
        for instance in result:
            bbox = instance["bbox"]
            score = instance["score"]
            Y1, X1, Y2, X2 = (
                int(bbox[0]),
                int(bbox[1]),
                int(bbox[0] + bbox[2]),
                int(bbox[1] + bbox[3]),
            )
            bboxes.append(np.array([[X1, Y1], [X1, Y2], [X2, Y2], [X2, Y1]]))
            scores.append(score)
        n_cells = len(scores)
        properties = {"score": scores}
        if Show_confidence:
            text_parameters = {
                "string": "{score:.2f}",
                "size": Score_text_size,
                "color": "red",
                "anchor": "upper_left",
                "translation": [-3, 0],
            }
            viewer.add_shapes(
                bboxes,
                face_color="transparent",
                edge_color="red",
                edge_width=Bbox_thickness,
                properties=properties,
                text=text_parameters,
                name=f"{n_cells} bounding boxes {name}",
            )
        else:
            viewer.add_shapes(
                bboxes,
                face_color="transparent",
                edge_color="red",
                edge_width=Bbox_thickness,
                properties=properties,
                name=f"{n_cells} bounding boxes {name}",
            )
        return bboxes, scores, n_cells

    viewer.window._status_bar._toggle_activity_dock(False)

    if Generate_points:
        print("Generating points...")
        create_points(result)
        print("Points are generated!")
    if Generate_bbox:
        print("Generating boxes...")
        create_bbox(result)
        print("Boxes are generated!")
    if not Generate_points and not Generate_bbox:
        print("None of the options are chosen, generating points as a default")
        create_points(result)
        print("Points are generated!")
    return None
