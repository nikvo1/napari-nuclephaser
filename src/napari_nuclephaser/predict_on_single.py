import os
import pathlib
import pickle
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

cuda_available = "cuda:0" if cuda.is_available() else "cpu"

models_folder = pathlib.Path(pathlib.Path(__file__).parent / "models")
first_model = next((x for x in models_folder.iterdir() if x.is_file()), None)


# ---------- Augmentation functions (unchanged) ----------
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


# ---------- Feature extraction helpers (copied from calibrate_with_dynamic_threshold) ----------
def expand_bbox(bbox, image_shape, scale=2.5):
    x1, x2, y1, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    new_w = width * scale
    new_h = height * scale
    nx1 = int(cx - new_w / 2)
    nx2 = int(cx + new_w / 2)
    ny1 = int(cy - new_h / 2)
    ny2 = int(cy + new_h / 2)
    h, w = image_shape
    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(w, nx2)
    ny2 = min(h, ny2)
    return nx1, nx2, ny1, ny2


def extract_features_grayscale(region):
    flat = region.ravel().astype(np.float64)
    h, w = region.shape
    height_width = h / w if w > 0 else 0.0
    mean = np.mean(flat)
    std = np.std(flat)
    median = np.median(flat)
    p25 = np.percentile(flat, 25)
    p75 = np.percentile(flat, 75)
    iqr = p75 - p25
    min_val = np.min(flat)
    max_val = np.max(flat)
    range_val = max_val - min_val
    rms = np.sqrt(np.mean((flat - mean) ** 2))
    lap = cv2.Laplacian(region, cv2.CV_64F, ksize=3)
    lap_var = np.var(lap.ravel())
    hist, _ = np.histogram(flat, bins=256, range=(0, 255))
    hist = hist.astype(np.float64)
    hist /= hist.sum() + 1e-12
    entropy = -np.sum(hist * np.log2(hist + 1e-12))
    energy = np.sum(flat**2)
    return {
        "height/width": height_width,
        "mean": mean,
        "std": std,
        "median": median,
        "p25": p25,
        "p75": p75,
        "iqr": iqr,
        "min": min_val,
        "max": max_val,
        "range": range_val,
        "rms_contrast": rms,
        "lap_var": lap_var,
        "entropy": entropy,
        "energy": energy,
    }


def extract_all_features(
    expanded_region, rel_x1, rel_y1, rel_x2, rel_y2, eps=1.0
):
    lap_expanded = cv2.Laplacian(expanded_region, cv2.CV_64F, ksize=3)
    flat_img = expanded_region.ravel()
    flat_lap = lap_expanded.ravel()

    ctx_mean = np.mean(flat_img)
    ctx_std = np.std(flat_img)
    ctx_median = np.median(flat_img)
    ctx_p25 = np.percentile(flat_img, 25)
    ctx_p75 = np.percentile(flat_img, 75)
    ctx_rms = np.sqrt(np.mean((flat_img - ctx_mean) ** 2))
    ctx_lap_var = np.var(flat_lap)
    hist, _ = np.histogram(flat_img, bins=256, range=(0, 255))
    hist = hist.astype(np.float64)
    hist /= hist.sum() + 1e-12
    ctx_entropy = -np.sum(hist * np.log2(hist + 1e-12))

    var_I = np.var(flat_img)
    var_L = np.var(flat_lap)
    focus_score = var_L / (var_I + eps) if (var_I + eps) > 0 else 0.0

    original_region = expanded_region[rel_y1:rel_y2, rel_x1:rel_x2]
    obj_feats = extract_features_grayscale(original_region)

    rel_mean = obj_feats["mean"] - ctx_median
    rel_median = obj_feats["median"] - ctx_median
    rel_std = obj_feats["std"] / (ctx_std + 1e-6)
    rel_contrast = obj_feats["rms_contrast"] - ctx_rms

    return {
        "height/width": obj_feats["height/width"],
        "mean": obj_feats["mean"],
        "std": obj_feats["std"],
        "median": obj_feats["median"],
        "p25": obj_feats["p25"],
        "p75": obj_feats["p75"],
        "iqr": obj_feats["iqr"],
        "min": obj_feats["min"],
        "max": obj_feats["max"],
        "range": obj_feats["range"],
        "rms_contrast": obj_feats["rms_contrast"],
        "lap_var": obj_feats["lap_var"],
        "entropy": obj_feats["entropy"],
        "energy": obj_feats["energy"],
        "context_mean": ctx_mean,
        "context_std": ctx_std,
        "context_median": ctx_median,
        "context_p25": ctx_p25,
        "context_p75": ctx_p75,
        "context_rms_contrast": ctx_rms,
        "context_lap_var": ctx_lap_var,
        "context_entropy": ctx_entropy,
        "relative_mean": rel_mean,
        "relative_median": rel_median,
        "relative_std": rel_std,
        "relative_contrast": rel_contrast,
        "focus": focus_score,
    }


# ---------- Metadata parsing (unchanged) ----------
def _parse_metadata(metadata_path):
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
    Mode={
        "choices": ["None", "TTA", "Dynamic threshold"],
        "tooltip": "Select the prediction mode: standard (None), Test-Time Augmentation (TTA), or dynamic threshold filtering.",
    },
    Mode_file={
        "mode": "r",
        "filter": "*.txt;*.pkl",
        "tooltip": "Select the TTA metadata file (.txt) or the dynamic threshold .pkl file, depending on the chosen mode.",
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
    Bbox_thickness={
        "tooltip": "Thickness of the side of rectangles in Shapes layer if Generate bbox is chosen"
    },
    Score_text_size={
        "tooltip": "Font size of confidence score text if Show confidence parameter is chosen"
    },
    Save_result={
        "tooltip": "If chosen, results will be saved in the format selected below."
    },
    Save_format={
        "choices": ["CSV", "XLSX", "Both"],
        "value": "CSV",
        "tooltip": "Select the output format for saving results (when Save_result is enabled).",
    },
    Experiment_name={
        "tooltip": "Name of the subfolder that will be created for the results (TTA mode only)."
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
    Mode="None",
    Mode_file=pathlib.Path(),
    Save_result=False,
    Save_folder=pathlib.Path(),
    Experiment_name="Experiment",
    Save_format="CSV",
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
    """Takes a single-frame image, YOLO model, and optional TTA or dynamic threshold -> adds point/bbox layers and saves averaged/filtered counts."""
    # ---------- Mode handling ----------
    use_tta = Mode == "TTA"
    use_dynamic = Mode == "Dynamic threshold"

    # If mode is not "None", the file must be provided and exist.
    if Mode != "None":
        if not Mode_file or not Mode_file.exists():
            show_error(
                f"Mode '{Mode}' requires a valid file. Please select the appropriate file."
            )
            return None

        # Check file extension vs mode
        if use_tta and Mode_file.suffix.lower() != ".txt":
            show_error("TTA mode requires a .txt metadata file.")
            return None
        if use_dynamic and Mode_file.suffix.lower() != ".pkl":
            show_error("Dynamic threshold mode requires a .pkl file.")
            return None

    # ---------- Input validation and preprocessing ----------
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

    # ---------- TTA mode ----------
    if use_tta:
        if not Mode_file or not Mode_file.exists():
            show_error("TTA metadata file not found or not provided.")
            viewer.window._status_bar._toggle_activity_dock(False)
            return None
        if Mode_file.suffix.lower() != ".txt":
            show_error("Selected file is not a .txt file.")
            viewer.window._status_bar._toggle_activity_dock(False)
            return None

        try:
            best_augs, aug_thresholds, model_name_meta = _parse_metadata(
                str(Mode_file)
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

        all_counts = []
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
                thr = aug_thresholds.get(aug_name, Confidence_threshold)
                model, _ = initialize_model(
                    str(Select_model), thr, cuda_available
                )

                aug_img = aug_func(pic)
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
                points_aug = []
                bboxes_aug = []
                scores_aug = []
                for instance in result:
                    bbox = instance["bbox"]
                    score = instance["score"]
                    scores_aug.append(score)

                    if scale_factor != 1.0:
                        center_x = int((bbox[0] + bbox[2] // 2) * scale_factor)
                        center_y = int((bbox[1] + bbox[3] // 2) * scale_factor)
                    else:
                        center_x = int(bbox[0] + bbox[2] // 2)
                        center_y = int(bbox[1] + bbox[3] // 2)
                    points_aug.append([center_y, center_x])

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

                    y1 = int(x_orig)
                    x1 = int(y_orig)
                    y2 = int(x_orig + w_orig)
                    x2 = int(y_orig + h_orig)
                    polygon = np.array(
                        [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
                    )
                    bboxes_aug.append(polygon)

                n_cells = len(points_aug)

                if Generate_points or (
                    not Generate_points and not Generate_bbox
                ):
                    viewer.add_points(
                        np.array(points_aug),
                        size=Points_size,
                        name=f"{n_cells} points ({aug_name}) {name}",
                    )

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

        avg_count = np.mean(all_counts) if all_counts else 0
        if Save_result:
            from napari_nuclephaser.utils import create_unique_subfolder

            subfolder = create_unique_subfolder(
                str(Save_folder), str(Experiment_name)
            )
            result_table = {"Frame": [name], "Count": [avg_count]}
            df = pd.DataFrame.from_dict(result_table)

            # Save according to the selected format
            if Save_format in ("CSV", "Both"):
                df.to_csv(
                    os.path.join(subfolder, f"{name}_TTA_averaged_counts.csv"),
                    index=False,
                )
            if Save_format in ("XLSX", "Both"):
                df.to_excel(
                    os.path.join(
                        subfolder, f"{name}_TTA_averaged_counts.xlsx"
                    ),
                    index=False,
                )

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

    # ---------- Dynamic threshold mode ----------
    if use_dynamic:
        if not Mode_file or not Mode_file.exists():
            show_error(
                "Dynamic threshold .pkl file not found or not provided."
            )
            viewer.window._status_bar._toggle_activity_dock(False)
            return None

        # Load the decision tree
        with open(Mode_file, "rb") as f:
            clf = pickle.load(f)

        # Feature columns used during training (must match exactly)
        feature_cols = [
            "confidence",
            "size",
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
            "context_mean",
            "context_std",
            "context_median",
            "context_p25",
            "context_p75",
            "context_rms_contrast",
            "context_lap_var",
            "context_entropy",
            "relative_mean",
            "relative_median",
            "relative_std",
            "relative_contrast",
            "focus",
        ]

        # Run inference with confidence = 0.01 (ignoring user's threshold)
        model, _ = initialize_model(str(Select_model), 0.01, cuda_available)

        # Sliced prediction progress
        pbar_slices = None

        def progress_callback(current, total):
            nonlocal pbar_slices
            if pbar_slices is None:
                pbar_slices = progress(total=total, desc="Sliced prediction")
            pbar_slices.update(1)
            if current == total:
                pbar_slices.close()

        result = get_sliced_prediction(
            pic,
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
        detections = result.object_prediction_list
        if not detections:
            show_info("No detections found by the model.")
            viewer.window._status_bar._toggle_activity_dock(False)
            return None

        # Extract features for each detection
        print("Extracting features for dynamic threshold...")
        rows = []
        # Convert the image to grayscale once (for feature extraction)
        if len(pic.shape) == 3:
            gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)
        else:
            gray = pic

        with progress(
            total=len(detections), desc="Feature extraction"
        ) as pbar_feat:
            for det in detections:
                x1, x2, y1, y2 = (
                    int(det.bbox.minx),
                    int(det.bbox.maxx),
                    int(det.bbox.miny),
                    int(det.bbox.maxy),
                )
                bbox = (x1, x2, y1, y2)
                # Expand bbox
                exp_x1, exp_x2, exp_y1, exp_y2 = expand_bbox(
                    bbox, gray.shape, scale=2.5
                )
                expanded = gray[exp_y1:exp_y2, exp_x1:exp_x2]
                rel_x1 = x1 - exp_x1
                rel_y1 = y1 - exp_y1
                rel_x2 = x2 - exp_x1
                rel_y2 = y2 - exp_y1
                feats = extract_all_features(
                    expanded, rel_x1, rel_y1, rel_x2, rel_y2
                )
                row = {
                    "confidence": det.score.value,
                    "size": (x2 - x1) * (y2 - y1),
                    **feats,
                }
                rows.append(row)
                pbar_feat.update(1)

        if not rows:
            show_info(
                "No features extracted (possible issue with detection format)."
            )
            viewer.window._status_bar._toggle_activity_dock(False)
            return None

        df = pd.DataFrame(rows)
        # Predict labels
        X = df[feature_cols]
        y_pred = clf.predict(X)
        # Keep only detections with label 1
        keep = y_pred == 1

        # Filter detections
        filtered_boxes = []
        filtered_scores = []
        filtered_points = []
        for idx, det in enumerate(detections):
            if keep[idx]:
                x1, x2, y1, y2 = (
                    int(det.bbox.minx),
                    int(det.bbox.maxx),
                    int(det.bbox.miny),
                    int(det.bbox.maxy),
                )
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                filtered_points.append([center_y, center_x])
                filtered_boxes.append(
                    np.array([[y1, x1], [y1, x2], [y2, x2], [y2, x1]])
                )
                filtered_scores.append(det.score.value)

        n_filtered = len(filtered_points)

        # Add layers
        if Generate_points or (not Generate_points and not Generate_bbox):
            viewer.add_points(
                np.array(filtered_points),
                size=Points_size,
                name=f"{n_filtered} points (dynamic) {name}",
            )

        if Generate_bbox:
            properties = {"score": filtered_scores}
            if Show_confidence:
                text_parameters = {
                    "string": "{score:.2f}",
                    "size": Score_text_size,
                    "color": "red",
                    "anchor": "upper_left",
                    "translation": [-3, 0],
                }
                viewer.add_shapes(
                    filtered_boxes,
                    face_color="transparent",
                    edge_color="red",
                    edge_width=Bbox_thickness,
                    properties=properties,
                    text=text_parameters,
                    name=f"{n_filtered} bounding boxes (dynamic) {name}",
                )
            else:
                viewer.add_shapes(
                    filtered_boxes,
                    face_color="transparent",
                    edge_color="red",
                    edge_width=Bbox_thickness,
                    properties=properties,
                    name=f"{n_filtered} bounding boxes (dynamic) {name}",
                )

        # Save results if requested
        if Save_result:
            from napari_nuclephaser.utils import create_unique_subfolder

            subfolder = create_unique_subfolder(
                str(Save_folder), str(Experiment_name)
            )
            dynamic_folder = os.path.join(
                subfolder, "Dynamic_Threshold_Results"
            )
            os.makedirs(dynamic_folder, exist_ok=True)

            # Save count
            result_table = {"Frame": [name], "Filtered_count": [n_filtered]}
            df_res = pd.DataFrame.from_dict(result_table)

            if Save_format in ("CSV", "Both"):
                df_res.to_csv(
                    os.path.join(dynamic_folder, f"{name}_dynamic_counts.csv"),
                    index=False,
                )
            if Save_format in ("XLSX", "Both"):
                df_res.to_excel(
                    os.path.join(
                        dynamic_folder, f"{name}_dynamic_counts.xlsx"
                    ),
                    index=False,
                )

            # Save the decision tree used (copy)
            import shutil

            shutil.copy2(
                Mode_file,
                os.path.join(dynamic_folder, "dynamic_threshold_used.pkl"),
            )

            # Metadata
            current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
            metadata = f"""Experiment time: {current_date}
Dynamic threshold prediction on single image
Image napari name: {name}
Detection model: {Select_model}
Dynamic threshold .pkl file: {Mode_file.name}
Number of detections before filtering: {len(detections)}
Number of detections after filtering: {n_filtered}
SAHI parameters used: size={Sahi_size}, overlap={Sahi_overlap}, postprocess={Postprocess}, match_metric={Match_metric}, iou_thr={Intersection_threshold}
"""
            metadata_path = os.path.join(
                dynamic_folder, f"{name}_dynamic_metadata.txt"
            )
            with open(metadata_path, "w", encoding="utf-8") as f:
                f.write(metadata)
            show_info(f"Dynamic threshold results saved in {dynamic_folder}")
        else:
            show_info(
                f"Dynamic threshold inference complete. Filtered count = {n_filtered}"
            )

        viewer.window._status_bar._toggle_activity_dock(False)
        return None

    # ---------- Original (non‑TTA, non‑dynamic) mode ----------
    initialization_pbar = progress(total=1, desc="Initializing model")
    print("Initializing model...")
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
