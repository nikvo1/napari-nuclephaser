import os
import pathlib
import pickle
import warnings
from collections import defaultdict
from datetime import datetime

import cv2
import matplotlib
import napari
import numpy as np
import pandas as pd
import seaborn as sns
from magicgui import magic_factory
from matplotlib import pyplot as plt
from napari.layers import Image, Shapes
from napari.utils import progress
from napari.utils.notifications import show_error, show_info
from sahi.predict import get_sliced_prediction
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Point, box
from shapely.strtree import STRtree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from torch import cuda

from napari_nuclephaser.utils import create_unique_subfolder, initialize_model

warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)
matplotlib.use("Agg")

cuda_available = "cuda:0" if cuda.is_available() else "cpu"

models_folder = pathlib.Path(pathlib.Path(__file__).parent / "models")
first_model = next((x for x in models_folder.iterdir() if x.is_file()), None)


def _ensure_numpy(arr):
    if hasattr(arr, "__module__") and arr.__module__.startswith("dask"):
        return arr.compute()
    return arr


def get_bbox_from_polygon(polygon):
    pts = polygon if isinstance(polygon, np.ndarray) else np.array(polygon)
    if pts.ndim == 2 and pts.shape[1] == 3:
        frame = int(pts[0, 0])
        y_vals = pts[:, 1]
        x_vals = pts[:, 2]
    else:
        frame = 0
        y_vals = pts[:, 0]
        x_vals = pts[:, 1]
    y1, y2 = int(np.min(y_vals)), int(np.max(y_vals))
    x1, x2 = int(np.min(x_vals)), int(np.max(x_vals))
    return (x1, x2, y1, y2), frame


def _split_image_and_boxes(
    image: np.ndarray, boxes: list[tuple[int, int, int, int]], window_size: int
) -> tuple[list[np.ndarray], list[int]]:
    height, width, _ = image.shape
    num_tiles_height = height // window_size
    num_tiles_width = width // window_size
    cropped_images = []
    boxes_per_tile = []
    for i in range(num_tiles_height):
        for j in range(num_tiles_width):
            left = j * window_size
            upper = i * window_size
            right = left + window_size
            lower = upper + window_size
            cropped = image[upper:lower, left:right, :]
            cropped_images.append(cropped)
            tile_boxes_count = 0
            for x1, x2, y1, y2 in boxes:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                if left <= cx < right and upper <= cy < lower:
                    tile_boxes_count += 1
            boxes_per_tile.append(tile_boxes_count)
    return cropped_images, boxes_per_tile


def blur_image(image: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    if sigma == 0:
        return image
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if image.dtype != np.uint8:
        raise ValueError("Input image must have dtype uint8.")
    if image.ndim not in (2, 3):
        raise ValueError("Image must be 2D or 3D.")
    img_float = image.astype(np.float32)
    sigma_blur = sigma if image.ndim == 2 else (sigma, sigma, 0)
    blurred_float = gaussian_filter(
        img_float, sigma=sigma_blur, mode="nearest"
    )
    return np.clip(blurred_float, 0, 255).astype(np.uint8)


def apply_random_augmentations(
    image, gamma_range=(0.7, 1.3), noise_sigma_range=(2, 15)
):
    return image


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


def compute_focus_score(region, eps=1.0):
    img = np.asarray(region, dtype=np.float64)
    lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
    var_I = np.var(img)
    var_L = np.var(lap)
    if var_I + eps > 0:
        return var_L / (var_I + eps)
    return 0.0


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


def match_boxes_to_boxes(
    gt_boxes,
    det_boxes,
    det_confidences=None,
    expand_scale=1.5,
    max_norm_dist=1e9,  # effectively disabled
    max_area_ratio=1e9,  # effectively disabled
):
    N_gt = len(gt_boxes)
    M_det = len(det_boxes)
    if N_gt == 0 or M_det == 0:
        return [0] * M_det

    det_centers, det_diags, det_areas = [], [], []
    for x1, x2, y1, y2 in det_boxes:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        diag = np.hypot(x2 - x1, y2 - y1) + 1e-6
        area = (x2 - x1) * (y2 - y1)
        det_centers.append((cx, cy))
        det_diags.append(diag)
        det_areas.append(area)

    expanded_det_boxes = []
    for x1, x2, y1, y2 in det_boxes:
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        new_w = w * expand_scale
        new_h = h * expand_scale
        ex1 = int(cx - new_w / 2)
        ex2 = int(cx + new_w / 2)
        ey1 = int(cy - new_h / 2)
        ey2 = int(cy + new_h / 2)
        expanded_det_boxes.append((ex1, ex2, ey1, ey2))

    det_geoms = [
        box(x1, y1, x2, y2) for (x1, x2, y1, y2) in expanded_det_boxes
    ]
    tree = STRtree(det_geoms)

    cost_matrix = np.full((N_gt, M_det), 1e9, dtype=np.float64)

    for i, (gx1, gx2, gy1, gy2) in enumerate(gt_boxes):
        g_cx = (gx1 + gx2) / 2
        g_cy = (gy1 + gy2) / 2
        g_area = (gx2 - gx1) * (gy2 - gy1)
        gt_point = Point(g_cx, g_cy)
        candidate_idxs = tree.query(gt_point, predicate="intersects")
        for j in candidate_idxs:
            ex1, ex2, ey1, ey2 = expanded_det_boxes[j]
            if ex1 <= g_cx <= ex2 and ey1 <= g_cy <= ey2:
                dcx, dcy = det_centers[j]
                dist = np.hypot(g_cx - dcx, g_cy - dcy)
                norm_dist = dist / det_diags[j]
                d_area = det_areas[j]
                area_ratio = max(d_area, g_area) / (min(d_area, g_area) + 1e-6)
                area_penalty = np.clip(
                    area_ratio - 1.0, 0, 10.0
                )  # cap to avoid extreme values
                conf = (
                    det_confidences[j] if det_confidences is not None else 1.0
                )
                conf_cost = 1.0 - conf
                # cost combines distance (0.45), confidence (0.1), area similarity (0.45)
                cost = 0.45 * norm_dist + 0.1 * conf_cost + 0.45 * area_penalty
                cost_matrix[i, j] = cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_detections = [0] * M_det
    for i, j in zip(row_ind, col_ind, strict=False):
        if cost_matrix[i, j] < 1e9:  # valid candidate existed
            matched_detections[j] = 1
    return matched_detections


def extract_detections_only_boxes(
    tile_phase_gray,
    phase_model,
    sahi_size,
    sahi_overlap,
    postprocess,
    match_metric,
    intersection_threshold,
    expand_scale=2.5,
):
    phase_rgb = cv2.cvtColor(tile_phase_gray, cv2.COLOR_GRAY2RGB)
    result = get_sliced_prediction(
        phase_rgb,
        phase_model,
        slice_height=sahi_size,
        slice_width=sahi_size,
        overlap_height_ratio=sahi_overlap,
        overlap_width_ratio=sahi_overlap,
        postprocess_type=postprocess,
        postprocess_match_metric=match_metric,
        postprocess_match_threshold=intersection_threshold,
        verbose=0,
        force_postprocess_type=True,
    )
    detections = result.object_prediction_list
    if not detections:
        return []
    dets = []
    for det in detections:
        x1, x2, y1, y2 = (
            int(det.bbox.minx),
            int(det.bbox.maxx),
            int(det.bbox.miny),
            int(det.bbox.maxy),
        )
        bbox = (x1, x2, y1, y2)
        conf = det.score.value
        exp_x1, exp_x2, exp_y1, exp_y2 = expand_bbox(
            bbox, tile_phase_gray.shape, scale=expand_scale
        )
        expanded = tile_phase_gray[exp_y1:exp_y2, exp_x1:exp_x2]
        rel_x1 = x1 - exp_x1
        rel_y1 = y1 - exp_y1
        rel_x2 = x2 - exp_x1
        rel_y2 = y2 - exp_y1
        feats = extract_all_features(expanded, rel_x1, rel_y1, rel_x2, rel_y2)
        dets.append(
            {
                "box": bbox,
                "confidence": conf,
                "features": feats,
            }
        )
    return dets


@magic_factory(
    Division_size={
        "max": 100000,
        "tooltip": "Tile size in pixels.",
    },
    Calibration_proportion={
        "tooltip": "Fraction of tiles used for calibration."
    },
    Max_blur_strength={
        "widget_type": "SpinBox",
        "min": 0,
        "max": 6,
        "value": 6,
        "tooltip": "Maximum defocus strength that the model with adapt to. The bigger - the more resistant the model to defocus, but calibration will take longer. 0 for skipping the adaptation to defocus",
    },
    Postprocess={
        "choices": ["GREEDYNMM", "NMS", "NMM"],
        "tooltip": "Algorithm to process overlapping detections (SAHI).",
    },
    Match_metric={
        "choices": ["IOS", "IOU"],
        "tooltip": "Metric to decide when two detections overlap.",
    },
    Save_folder={"mode": "d"},
    Sahi_size={
        "max": 100000,
        "tooltip": "Size of sliding window for sliced inference (pixels).",
    },
    Sahi_overlap={
        "tooltip": "Relative overlap between sliding windows.",
    },
    Intersection_threshold={
        "tooltip": "Threshold for merging overlapping detections.",
    },
    Experiment_name={
        "tooltip": "Name of the subfolder where results will be saved."
    },
    Debug={
        "widget_type": "CheckBox",
        "tooltip": "Add debug Shapes layer showing matched (green), false positives (red), and false negatives (blue).",
    },
    call_button="Calibrate Dynamic Threshold",
    auto_call=False,
    result_widget=True,
)
def calibrate_with_dynamic_threshold(
    Select_Phase_stack: Image,
    Select_Shapes_layer: Shapes,
    viewer: napari.Viewer,
    Phase_model=first_model,
    Division_size=640,
    Calibration_proportion=0.5,
    Max_blur_strength=6,
    Save_folder=pathlib.Path(),
    Experiment_name="Experiment",
    ADVANCED_SETTINGS="",
    Random_seed=42,
    Postprocess="GREEDYNMM",
    Match_metric="IOS",
    Intersection_threshold=0.3,
    Sahi_size=640,
    Sahi_overlap: float = 0.2,
    Debug: bool = False,
):
    image_data = _ensure_numpy(Select_Phase_stack.data)
    shapes_data = Select_Shapes_layer.data
    if not shapes_data:
        show_error("Shapes layer is empty!")
        return None

    boxes_per_frame = defaultdict(list)
    for poly in shapes_data:
        if len(poly) < 3:
            continue
        (x1, x2, y1, y2), frame = get_bbox_from_polygon(poly)
        boxes_per_frame[frame].append((x1, x2, y1, y2))

    if not boxes_per_frame:
        show_error("No valid bounding boxes found in Shapes layer.")
        return None

    if image_data.ndim == 2 or (
        image_data.ndim == 3 and image_data.shape[-1] in (1, 3, 4)
    ):
        n_frames = 1
        images = [image_data]
    elif (
        image_data.ndim == 3
        or image_data.ndim == 4
        and image_data.shape[-1] in (1, 3, 4)
    ):
        n_frames = image_data.shape[0]
        images = [image_data[i] for i in range(n_frames)]
    else:
        show_error("Unsupported image dimensions.")
        return None

    frames_with_images = [t for t in range(n_frames) if t in boxes_per_frame]
    if not frames_with_images:
        show_error("No frames with matching ground truth boxes found.")
        return None

    sigmas = list(range(Max_blur_strength + 1))
    if not sigmas:
        show_error("Max_blur_strength must be >= 0.")
        return None

    print("Initialising model with confidence=0.01 ...")
    phase_model_low, model_type = initialize_model(
        str(Phase_model), 0.01, cuda_available
    )
    print(f"Model ready: {model_type}, device: {cuda_available}")

    calib_data = []
    test_data = []
    viewer.window._status_bar._toggle_activity_dock(True)

    with progress(
        total=len(frames_with_images), desc="Splitting frames"
    ) as pbar:
        for t in frames_with_images:
            img = images[t]
            frame_boxes = boxes_per_frame[t]
            if not frame_boxes:
                pbar.update(1)
                continue

            if img.ndim == 3 and img.shape[-1] == 3:
                phase_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif img.ndim == 3 and img.shape[-1] == 1:
                phase_gray = img[:, :, 0]
            elif img.ndim == 2:
                phase_gray = img
            else:
                phase_gray = img

            if phase_gray.dtype == np.uint16:
                phase_gray = cv2.convertScaleAbs(phase_gray, alpha=255 / 65535)

            rgb_img = cv2.cvtColor(phase_gray, cv2.COLOR_GRAY2RGB)

            all_tiles, all_gt_counts = _split_image_and_boxes(
                rgb_img, frame_boxes, Division_size
            )
            n_tiles = len(all_tiles)
            if n_tiles == 0:
                pbar.update(1)
                continue

            tile_boxes = []
            for i in range(n_tiles):
                left = (
                    i % (rgb_img.shape[1] // Division_size)
                ) * Division_size
                upper = (
                    i // (rgb_img.shape[1] // Division_size)
                ) * Division_size
                boxes_in_tile = []
                for x1, x2, y1, y2 in frame_boxes:
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    if (
                        left <= cx < left + Division_size
                        and upper <= cy < upper + Division_size
                    ):
                        boxes_in_tile.append(
                            (x1 - left, x2 - left, y1 - upper, y2 - upper)
                        )
                tile_boxes.append(boxes_in_tile)

            np.random.seed(Random_seed)
            calib_indices = np.random.choice(
                n_tiles, int(n_tiles * Calibration_proportion), replace=False
            )
            test_indices = np.setdiff1d(np.arange(n_tiles), calib_indices)

            for idx in calib_indices:
                tile_rgb = all_tiles[idx]
                tile_gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)
                left = (
                    idx % (rgb_img.shape[1] // Division_size)
                ) * Division_size
                upper = (
                    idx // (rgb_img.shape[1] // Division_size)
                ) * Division_size
                calib_data.append(
                    {
                        "frame_idx": t,
                        "tile_gray": tile_gray,
                        "gt_boxes": tile_boxes[idx],
                        "gt_count": len(tile_boxes[idx]),
                        "tile_left": left,
                        "tile_upper": upper,
                    }
                )

            for idx in test_indices:
                tile_rgb = all_tiles[idx]
                tile_gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)
                left = (
                    idx % (rgb_img.shape[1] // Division_size)
                ) * Division_size
                upper = (
                    idx // (rgb_img.shape[1] // Division_size)
                ) * Division_size
                test_data.append(
                    {
                        "frame_idx": t,
                        "tile_gray": tile_gray,
                        "gt_boxes": tile_boxes[idx],
                        "gt_count": len(tile_boxes[idx]),
                        "tile_left": left,
                        "tile_upper": upper,
                    }
                )

            pbar.update(1)

    if not calib_data:
        show_error(
            "No calibration tiles found (no ground truth boxes in any tile)."
        )
        return None

    print("Extracting features from calibration tiles...")
    frame_detections = defaultdict(list)

    total_steps = len(calib_data) * len(sigmas)
    with progress(
        total=total_steps, desc="Calibration feature extraction"
    ) as pbar:
        for entry in calib_data:
            frame = entry["frame_idx"]
            tile_gray = entry["tile_gray"]
            gt_boxes = entry["gt_boxes"]
            tile_left = entry["tile_left"]
            tile_upper = entry["tile_upper"]

            for sigma in sigmas:
                np.random.seed(Random_seed + sigma + frame)
                aug_tile = apply_random_augmentations(
                    tile_gray, gamma_range=(0.5, 1.5)
                )
                blurred = blur_image(aug_tile, sigma)

                dets = extract_detections_only_boxes(
                    blurred,
                    phase_model_low,
                    Sahi_size,
                    Sahi_overlap,
                    Postprocess,
                    Match_metric,
                    Intersection_threshold,
                    expand_scale=2.5,
                )
                for det in dets:
                    det["gt_boxes"] = gt_boxes
                    det["tile_origin"] = (tile_left, tile_upper)
                    frame_detections[frame].append(det)
                pbar.update(1)

    if not frame_detections:
        show_error("No detections found in any calibration tile.")
        return None

    print("Matching detections to ground truth boxes...")
    samples_per_frame = defaultdict(list)
    debug_info = (
        []
    )  # (frame, gt_boxes, det_boxes, matched, tile_left, tile_upper)

    for frame, dets in frame_detections.items():
        tile_groups = defaultdict(list)
        for det in dets:
            key = frozenset(det["gt_boxes"])
            tile_groups[key].append(det)

        for gt_boxes_key, group_dets in tile_groups.items():
            gt_boxes = list(gt_boxes_key)
            # get tile origin from first detection (all share same origin)
            tile_left, tile_upper = group_dets[0]["tile_origin"]
            if not gt_boxes:
                for det in group_dets:
                    det["ground_truth"] = 0
                    samples_per_frame[frame].append(det)
                continue

            det_boxes = [d["box"] for d in group_dets]
            confs = [d["confidence"] for d in group_dets]
            matched = match_boxes_to_boxes(
                gt_boxes,
                det_boxes,
                det_confidences=confs,
                expand_scale=1.05,
                max_norm_dist=0.6,
                max_area_ratio=2.0,
            )
            for det, gt in zip(group_dets, matched, strict=False):
                det["ground_truth"] = gt
                samples_per_frame[frame].append(det)

            debug_info.append(
                (frame, gt_boxes, det_boxes, matched, tile_left, tile_upper)
            )

    if Debug:
        print("Creating debug Shapes layer with matched boxes...")
        matched_vertices = []
        for frame, _, det_boxes, matched, tile_left, tile_upper in debug_info:
            for det_box, m in zip(det_boxes, matched, strict=False):
                if m:
                    x1, x2, y1, y2 = det_box
                    # shift to global coordinates
                    gx1 = x1 + tile_left
                    gx2 = x2 + tile_left
                    gy1 = y1 + tile_upper
                    gy2 = y2 + tile_upper
                    matched_vertices.append(
                        np.array(
                            [
                                [frame, gy1, gx1],
                                [frame, gy1, gx2],
                                [frame, gy2, gx2],
                                [frame, gy2, gx1],
                            ]
                        )
                    )
        if matched_vertices:
            viewer.add_shapes(
                matched_vertices,
                face_color="transparent",
                edge_color="green",
                edge_width=2,
                name="Matched boxes (TP)",
            )
    all_rows = []
    for _, dets in samples_per_frame.items():
        for det in dets:
            row = {
                "confidence": det["confidence"],
                "size": (det["box"][1] - det["box"][0])
                * (det["box"][3] - det["box"][2]),
                "ground_truth": det["ground_truth"],
                **det["features"],
            }
            all_rows.append(row)
    if not all_rows:
        show_error("No matching results after applying box matching.")
        return None
    full_df = pd.DataFrame(all_rows)
    tp = full_df[full_df["ground_truth"] == 1]
    fp = full_df[full_df["ground_truth"] == 0]
    if len(tp) == 0:
        show_error("No true positive samples found. Check matching.")
        return None
    if len(tp) > len(fp):
        tp = tp.sample(n=len(fp), random_state=Random_seed)
    elif len(fp) > len(tp):
        fp = fp.sample(n=len(tp), random_state=Random_seed)
    balanced_df = pd.concat([tp, fp], ignore_index=True).sample(
        frac=1, random_state=Random_seed
    )
    if balanced_df.empty:
        show_error("Balanced training set is empty. Cannot train classifier.")
        return None
    print(
        f"Balanced training set size: {len(balanced_df)} (TP: {len(tp)}, FP: {len(fp)})"
    )
    calib_pbar = progress(total=0, desc="Calibrating dynamic threshold")
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
    X = balanced_df[feature_cols]
    y = balanced_df["ground_truth"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=Random_seed, stratify=y
    )
    print(f"Training set size before cleaning: {len(X_train)}")
    temp_rf = RandomForestClassifier(
        n_estimators=500,
        criterion="log_loss",
        random_state=Random_seed,
        n_jobs=-1,
    )
    oof_pred = cross_val_predict(
        temp_rf,
        X_train,
        y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=Random_seed),
        method="predict",
    )
    y_train_array = y_train.values
    keep_mask = []
    for i in range(len(X_train)):
        true_label = y_train_array[i]
        pred_label = oof_pred[i]
        is_correct = pred_label == true_label
        keep_mask.append(is_correct)
    keep_mask = np.array(keep_mask)
    X_train_cleaned = X_train[keep_mask]
    y_train_cleaned = y_train[keep_mask]
    print(f"Training set size after cleaning: {len(X_train_cleaned)}")
    print(
        f"Dropped {len(X_train) - len(X_train_cleaned)} samples ({(1 - len(X_train_cleaned)/len(X_train))*100:.1f}%)"
    )
    min_samples_leaf_value = max(1, int(0.03 * len(X_train_cleaned)))
    print(f"min_samples_leaf set to: {min_samples_leaf_value}")
    final_rf = RandomForestClassifier(
        n_estimators=500,
        criterion="log_loss",
        max_depth=None,
        min_samples_leaf=min_samples_leaf_value,
        random_state=Random_seed,
        n_jobs=-1,
        oob_score=True,
    )
    final_rf.fit(X_train_cleaned, y_train_cleaned)
    y_val_pred = final_rf.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred)
    calib_pbar.close()
    print("\n========== FINAL RESULTS ==========")
    print(f"Final model trained on {len(X_train_cleaned)} clean samples")
    print(f"Validation F1 Score: {val_f1:.4f}")
    print(f"OOB Score (on cleaned data): {final_rf.oob_score_:.4f}")
    print("Testing on test tiles with different blurs...")
    results = []
    with progress(total=len(test_data) * len(sigmas), desc="Testing") as pbar:
        for entry in test_data:
            tile_gray = entry["tile_gray"]
            gt_count = entry["gt_count"]
            frame = entry["frame_idx"]
            for sigma in sigmas:
                np.random.seed(Random_seed + sigma + frame)
                blurred = blur_image(tile_gray, sigma)
                phase_rgb = cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)
                result = get_sliced_prediction(
                    phase_rgb,
                    phase_model_low,
                    slice_height=Sahi_size,
                    slice_width=Sahi_size,
                    overlap_height_ratio=Sahi_overlap,
                    overlap_width_ratio=Sahi_overlap,
                    postprocess_type=Postprocess,
                    postprocess_match_metric=Match_metric,
                    postprocess_match_threshold=Intersection_threshold,
                    verbose=0,
                    force_postprocess_type=True,
                )
                detections = result.object_prediction_list
                if not detections:
                    pred_count = 0
                else:
                    rows = []
                    for det in detections:
                        x1, x2, y1, y2 = (
                            int(det.bbox.minx),
                            int(det.bbox.maxx),
                            int(det.bbox.miny),
                            int(det.bbox.maxy),
                        )
                        box_ = (x1, x2, y1, y2)
                        exp_x1, exp_x2, exp_y1, exp_y2 = expand_bbox(
                            box_, blurred.shape, scale=2.5
                        )
                        expanded = blurred[exp_y1:exp_y2, exp_x1:exp_x2]
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
                    if rows:
                        df_test = pd.DataFrame(rows)
                        X_test = df_test[feature_cols]
                        y_pred = final_rf.predict(X_test)
                        pred_count = np.sum(y_pred)
                    else:
                        pred_count = 0
                results.append(
                    {
                        "sigma": sigma,
                        "frame": frame,
                        "gt_count": gt_count,
                        "pred_count": pred_count,
                    }
                )
                pbar.update(1)
    results_df = pd.DataFrame(results)
    mape_per_sigma = {}
    for sigma in sigmas:
        sub = results_df[results_df["sigma"] == sigma]
        sub = sub[sub["gt_count"] > 0]
        if not sub.empty:
            pe = (
                np.abs((sub["gt_count"] - sub["pred_count"]) / sub["gt_count"])
                * 100
            )
            mape = pe.mean()
        else:
            mape = np.nan
        mape_per_sigma[sigma] = mape
    subfolder = create_unique_subfolder(str(Save_folder), str(Experiment_name))
    dynamic_folder = os.path.join(subfolder, "Dynamic Confidence Threshold")
    os.makedirs(dynamic_folder, exist_ok=True)
    model_path = os.path.join(dynamic_folder, "dynamic_threshold_rf.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(final_rf, f)
    ref_boxes = []
    for frame, boxes in boxes_per_frame.items():
        for x1, x2, y1, y2 in boxes:
            ref_boxes.append([frame, x1, x2, y1, y2])
    if ref_boxes:
        pd.DataFrame(
            ref_boxes, columns=["frame", "x1", "x2", "y1", "y2"]
        ).to_csv(
            os.path.join(dynamic_folder, "reference_boxes.csv"), index=False
        )
    sns.set(rc={"figure.dpi": 150, "savefig.dpi": 150})
    for sigma in sigmas:
        sub = results_df[results_df["sigma"] == sigma]
        if sub.empty:
            continue
        fig, ax = plt.subplots()
        sns.scatterplot(
            data=sub,
            x="gt_count",
            y="pred_count",
            hue="frame",
            palette="tab10",
            ax=ax,
        )
        max_val = max(sub["gt_count"].max(), sub["pred_count"].max())
        sns.lineplot(
            x=np.arange(0, max_val + 1),
            y=np.arange(0, max_val + 1),
            color="red",
            ax=ax,
        )
        mape = mape_per_sigma.get(sigma, np.nan)
        ax.set_title(
            f"Blur strength = {sigma}\nMAPE = {mape:.2f}%"
            if not np.isnan(mape)
            else f"Blur strength = {sigma}\nMAPE = N/A"
        )
        ax.legend(title="Frame", bbox_to_anchor=(1.05, 1), loc="upper left")
        plot_path = os.path.join(dynamic_folder, f"scatter_sigma_{sigma}.png")
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    metadata = f"""Experiment time: {current_date}
Calibration method: dynamic threshold (Random Forest with noise cleaning) using GROUND TRUTH BOXES
Phase image stack: {Select_Phase_stack.name}, shape {image_data.shape}
Phase model: {Phase_model.name} ({model_type})
Division size: {Division_size}
Calibration proportion: {Calibration_proportion}
Random seed: {Random_seed}
Postprocess: {Postprocess}
Match metric: {Match_metric}
Intersection threshold: {Intersection_threshold}
SAHI size: {Sahi_size}
SAHI overlap: {Sahi_overlap}
Maximum blur strength: {Max_blur_strength} (blur strengths tested: {sigmas})

--- Training details ---
Number of frames used for calibration: {len(samples_per_frame)}
Total samples after balancing (TP + FP): {len(balanced_df)}
Number of true positives used: {len(tp)}
Number of false positives used: {len(fp)}
Per-frame TP counts after balancing:
"""
    for frame, df in samples_per_frame.items():
        if isinstance(df, pd.DataFrame):
            tp_count = df["ground_truth"].sum()
            metadata += f"  Frame {frame}: {tp_count} TP samples\n"
    metadata += f"""
Random Forest parameters:
  n_estimators: 500
  criterion: log_loss
  max_depth: None (controlled by min_samples_leaf)
  min_samples_leaf: {min_samples_leaf_value} (3% of cleaned training set)
  oob_score: True
Noise cleaning used 5-fold cross-validation, dropping samples with OOF prediction != true label
Validation F1 score: {val_f1:.4f}
OOB score: {final_rf.oob_score_:.4f}

--- Testing results ---
Per-sigma MAPE (over tiles with gt>0):
"""
    for sigma, mape in mape_per_sigma.items():
        metadata += f"  Blur strength {sigma}: {mape:.2f}%\n"
    metadata_path = os.path.join(dynamic_folder, "metadata.txt")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write(metadata)
    viewer.window._status_bar._toggle_activity_dock(False)
    show_info("Dynamic threshold calibration completed.")
    summary = f"Dynamic threshold calibrated. Random Forest trained on {len(X_train_cleaned)} clean samples.\n"
    for sigma, mape in mape_per_sigma.items():
        summary += f"Blur strength {sigma} MAPE: {mape:.2f}%\n"
    return summary
