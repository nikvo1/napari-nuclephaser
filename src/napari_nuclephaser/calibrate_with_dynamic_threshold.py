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
from napari.layers import Image, Points
from napari.utils import progress
from napari.utils.notifications import show_error, show_info
from sahi.predict import get_sliced_prediction
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from shapely.geometry import box
from shapely.strtree import STRtree
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from torch import cuda

from napari_nuclephaser.utils import create_unique_subfolder, initialize_model

warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)
matplotlib.use("Agg")

cuda_available = "cuda:0" if cuda.is_available() else "cpu"

models_folder = pathlib.Path(pathlib.Path(__file__).parent / "models")
first_model = next((x for x in models_folder.iterdir() if x.is_file()), None)
model_type_list = ("ultralytics", "yolov5", "yolov8", "yolov11", "yolo11")


# ---------- Helpers (from original) ----------
def _ensure_numpy(arr):
    if hasattr(arr, "__module__") and arr.__module__.startswith("dask"):
        return arr.compute()
    return arr


def _split_image_and_points(
    image: np.ndarray, points: list[tuple[float, float]], window_size: int
) -> tuple[list[np.ndarray], list[int]]:
    height, width, _ = image.shape
    num_tiles_height = height // window_size
    num_tiles_width = width // window_size
    cropped_images = []
    points_per_tile = []

    for i in range(num_tiles_height):
        for j in range(num_tiles_width):
            left = j * window_size
            upper = i * window_size
            right = left + window_size
            lower = upper + window_size
            cropped = image[upper:lower, left:right, :]
            cropped_images.append(cropped)
            tile_points_count = 0
            for y, x in points:
                if left <= x < right and upper <= y < lower:
                    tile_points_count += 1
            points_per_tile.append(tile_points_count)

    return cropped_images, points_per_tile


def _prepare_frame(
    image: np.ndarray,
    points_2d: list[tuple[float, float]],
    division_size: int,
    calibration_proportion: float,
    random_seed: int,
) -> tuple[list[np.ndarray], list[int], list[np.ndarray], list[int]]:
    image = _ensure_numpy(image)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.dtype == np.uint16:
        image = cv2.convertScaleAbs(image, alpha=255 / 65535).astype(np.uint8)

    all_tiles, all_gt_counts = _split_image_and_points(
        image, points_2d, division_size
    )
    n_tiles = len(all_tiles)
    if n_tiles == 0:
        return [], [], [], []

    n_calib = int(n_tiles * calibration_proportion)
    np.random.seed(random_seed)
    calib_indices = np.random.choice(n_tiles, n_calib, replace=False)
    test_indices = np.setdiff1d(np.arange(n_tiles), calib_indices)

    calib_tiles = [all_tiles[i] for i in calib_indices]
    calib_gt = [all_gt_counts[i] for i in calib_indices]
    test_tiles = [all_tiles[i] for i in test_indices]
    test_gt = [all_gt_counts[i] for i in test_indices]

    return calib_tiles, calib_gt, test_tiles, test_gt


# ---------- Dynamic threshold helpers ----------
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
    original_dtype = image.dtype
    if np.issubdtype(original_dtype, np.integer):
        img_float = image.astype(np.float32) / 255.0
        max_intensity = 255.0
    else:
        img_float = image.astype(np.float32)
        max_intensity = 1.0
    img_float = np.clip(img_float, 0.0, 1.0)

    gamma = np.random.uniform(gamma_range[0], gamma_range[1])
    img_gamma = np.power(img_float, gamma)

    noise_sigma = np.random.uniform(noise_sigma_range[0], noise_sigma_range[1])
    noise_sigma_normalized = noise_sigma / max_intensity
    noise = np.random.normal(
        loc=0.0, scale=noise_sigma_normalized, size=img_gamma.shape
    )
    img_noisy = img_gamma + noise
    img_noisy = np.clip(img_noisy, 0.0, 1.0)

    if np.issubdtype(original_dtype, np.integer):
        img_out = (img_noisy * 255.0).astype(original_dtype)
    else:
        img_out = img_noisy.astype(original_dtype)
    return img_out


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


def match_points_to_boxes(points, boxes, confidences):
    N = len(points)
    M = len(boxes)
    if N == 0 or M == 0:
        return [0] * M

    from shapely.geometry import Point

    box_geoms = [box(x1, y1, x2, y2) for (x1, x2, y1, y2) in boxes]
    tree = STRtree(box_geoms)

    BIG = 100.0
    cost_matrix = np.full((N, M + N), BIG)

    for i, (py, px) in enumerate(points):
        pt = Point(px, py)
        candidate_idxs = tree.query(pt, predicate="intersects")
        for j in candidate_idxs:
            x1, x2, y1, y2 = boxes[j]
            if x1 <= px <= x2 and y1 <= py <= y2:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
                diag = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + 1e-6
                norm_dist = dist / diag
                conf = confidences[j] if confidences is not None else 1.0
                conf_cost = 1.0 - conf
                cost = 0.5 * norm_dist + 0.5 * conf_cost
                cost_matrix[i, j] = cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_boxes = [0] * M
    for i in range(N):
        j = col_ind[i]
        if j < M:
            matched_boxes[j] = 1
    return matched_boxes


def rotate_tile_and_points(tile, points, angle_deg):
    if angle_deg == 0:
        return tile, points
    h, w = tile.shape[:2]
    if angle_deg == 90:
        rotated = cv2.rotate(tile, cv2.ROTATE_90_CLOCKWISE)
    elif angle_deg == 180:
        rotated = cv2.rotate(tile, cv2.ROTATE_180)
    elif angle_deg == 270:
        rotated = cv2.rotate(tile, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError("angle_deg must be 0, 90, 180, or 270")

    new_points = []
    for y, x in points:
        if angle_deg == 90:
            new_y = x
            new_x = h - 1 - y
        elif angle_deg == 180:
            new_y = h - 1 - y
            new_x = w - 1 - x
        else:  # 270
            new_y = w - 1 - x
            new_x = y
        new_y = max(0, min(h - 1, int(round(new_y))))
        new_x = max(0, min(w - 1, int(round(new_x))))
        new_points.append((new_y, new_x))
    return rotated, new_points


def extract_detections_and_features(
    tile_phase_gray,
    points_in_tile,
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
        return pd.DataFrame()

    boxes_list = []
    confs = []
    for det in detections:
        x1, x2, y1, y2 = (
            int(det.bbox.minx),
            int(det.bbox.maxx),
            int(det.bbox.miny),
            int(det.bbox.maxy),
        )
        boxes_list.append((x1, x2, y1, y2))
        confs.append(det.score.value)

    matched = match_points_to_boxes(points_in_tile, boxes_list, confs)

    rows = []
    for _idx, (bbox, conf, gt) in enumerate(
        zip(boxes_list, confs, matched, strict=False)
    ):
        x1, x2, y1, y2 = bbox
        exp_x1, exp_x2, exp_y1, exp_y2 = expand_bbox(
            bbox, tile_phase_gray.shape, scale=expand_scale
        )
        expanded = tile_phase_gray[exp_y1:exp_y2, exp_x1:exp_x2]
        rel_x1 = x1 - exp_x1
        rel_y1 = y1 - exp_y1
        rel_x2 = x2 - exp_x1
        rel_y2 = y2 - exp_y1
        feats = extract_all_features(expanded, rel_x1, rel_y1, rel_x2, rel_y2)
        row = {
            "confidence": conf,
            "size": (x2 - x1) * (y2 - y1),
            "ground_truth": gt,
            **feats,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def save_debug_rotation_image(
    tile_gray,
    points,
    boxes,
    matched_boxes,
    output_path,
    angle_deg,
    sigma,
):
    """
    Save a debug image showing rotated tile, ground‑truth points (green),
    all detection boxes (blue), and matched boxes (red).
    """
    # Convert grayscale to BGR for drawing
    img_color = cv2.cvtColor(tile_gray, cv2.COLOR_GRAY2BGR)

    # Draw all detection boxes in blue
    for x1, x2, y1, y2 in boxes:
        cv2.rectangle(img_color, (x1, y1), (x2, y2), (255, 0, 0), 1)

    # Draw matched boxes in red
    for idx, matched in enumerate(matched_boxes):
        if matched:
            x1, x2, y1, y2 = boxes[idx]
            cv2.rectangle(img_color, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Draw ground‑truth points in green
    for py, px in points:
        cv2.circle(img_color, (int(px), int(py)), 3, (0, 255, 0), -1)

    # Add text with angle and sigma
    cv2.putText(
        img_color,
        f"Rotation {angle_deg}°, Sigma {sigma}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )

    cv2.imwrite(output_path, img_color)


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
    call_button="Calibrate Dynamic Threshold",
    auto_call=False,
    result_widget=True,
)
def calibrate_with_dynamic_threshold(
    Select_Phase_stack: Image,
    Select_Points_layer: Points,
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
):
    """
    Calibrate a dynamic threshold using a decision tree classifier.
    Replaces fixed confidence threshold with feature-based filtering.
    """
    # ---------- Input validation ----------
    image_data = _ensure_numpy(Select_Phase_stack.data)
    points_data = _ensure_numpy(Select_Points_layer.data)

    if len(points_data) == 0:
        show_error("Points layer is empty!")
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

    if points_data.ndim == 2:
        if points_data.shape[1] == 2:
            if n_frames != 1:
                show_error("Points have 2 columns but multiple frames.")
                return None
            points_per_frame = {0: [(y, x) for y, x in points_data]}
        elif points_data.shape[1] == 3:
            points_per_frame = {}
            for pt in points_data:
                t, y, x = int(pt[0]), pt[1], pt[2]
                points_per_frame.setdefault(t, []).append((y, x))
        else:
            show_error(
                f"Points layer has {points_data.shape[1]} columns. Expected 2 or 3."
            )
            return None
    else:
        show_error("Points layer must be 2D.")
        return None

    for t in range(n_frames):
        if t not in points_per_frame:
            points_per_frame[t] = []

    frames_with_images = [t for t in range(n_frames) if t < len(images)]
    if not frames_with_images:
        show_error("No valid frames found.")
        return None

    sigmas = list(range(Max_blur_strength + 1))
    if not sigmas:
        show_error("Max_blur_strength must be >= 0.")
        return None

    # ---------- Initialise model (low confidence) ----------
    print("Initialising model with confidence=0.01 ...")
    phase_model_low, model_type = initialize_model(
        str(Phase_model), 0.01, cuda_available
    )
    print(f"Model ready: {model_type}, device: {cuda_available}")

    # ---------- Split into tiles ----------
    calib_data = []  # each: {frame_idx, tile_gray, points, gt_count}
    test_data = []  # same

    viewer.window._status_bar._toggle_activity_dock(True)

    with progress(
        total=len(frames_with_images), desc="Splitting frames"
    ) as pbar:
        for t in frames_with_images:
            img = images[t]
            pts = points_per_frame.get(t, [])

            # Extract grayscale phase image
            if img.ndim == 3 and img.shape[-1] == 3:
                phase_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif img.ndim == 3 and img.shape[-1] == 1:
                phase_gray = img[:, :, 0]
            elif img.ndim == 2:
                phase_gray = img
            else:
                phase_gray = img  # fallback

            if phase_gray.dtype == np.uint16:
                phase_gray = cv2.convertScaleAbs(phase_gray, alpha=255 / 65535)

            # Build RGB version for splitting (points counting)
            rgb_img = cv2.cvtColor(phase_gray, cv2.COLOR_GRAY2RGB)

            # Get tiles and their point counts
            all_tiles, all_gt_counts = _split_image_and_points(
                rgb_img, pts, Division_size
            )
            n_tiles = len(all_tiles)
            if n_tiles == 0:
                pbar.update(1)
                continue

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
                tile_points = []
                for y, x in pts:
                    if (
                        left <= x < left + Division_size
                        and upper <= y < upper + Division_size
                    ):
                        tile_points.append((y - upper, x - left))
                calib_data.append(
                    {
                        "frame_idx": t,
                        "tile_gray": tile_gray,
                        "points": tile_points,
                        "gt_count": len(tile_points),
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
                tile_points = []
                for y, x in pts:
                    if (
                        left <= x < left + Division_size
                        and upper <= y < upper + Division_size
                    ):
                        tile_points.append((y - upper, x - left))
                test_data.append(
                    {
                        "frame_idx": t,
                        "tile_gray": tile_gray,
                        "points": tile_points,
                        "gt_count": len(tile_points),
                    }
                )

            pbar.update(1)

    if not calib_data:
        show_error("No calibration tiles found.")
        return None

    # ---------- Calibration: extract features with blur augmentation ----------
    print(
        "Extracting features from calibration tiles (including blur augmentation)..."
    )
    samples_per_frame = defaultdict(list)
    tile_data_per_frame = defaultdict(
        list
    )  # original tiles for rotation (stored once)

    total_steps = len(calib_data) * len(sigmas)
    with progress(
        total=total_steps, desc="Calibration feature extraction"
    ) as pbar:
        for entry in calib_data:
            frame = entry["frame_idx"]
            tile_gray = entry["tile_gray"]
            points = entry["points"]

            # Store original tile once for possible rotation later
            tile_data_per_frame[frame].append((tile_gray, points))

            for sigma in sigmas:
                np.random.seed(Random_seed + sigma + frame)
                aug_tile = apply_random_augmentations(
                    tile_gray, gamma_range=(0.5, 1.5)
                )
                blurred = blur_image(aug_tile, sigma)

                df = extract_detections_and_features(
                    blurred,
                    points,
                    phase_model_low,
                    Sahi_size,
                    Sahi_overlap,
                    Postprocess,
                    Match_metric,
                    Intersection_threshold,
                    expand_scale=2.5,
                )
                if not df.empty:
                    samples_per_frame[frame].append(df)
                pbar.update(1)

    # ---------- Combine per frame into DataFrames (with safety checks) ----------
    print("Combining samples per frame...")
    for frame in list(samples_per_frame.keys()):
        # If it's already a DataFrame, skip
        if isinstance(samples_per_frame[frame], pd.DataFrame):
            continue
        # If it's a list of DataFrames, concatenate
        if (
            isinstance(samples_per_frame[frame], list)
            and samples_per_frame[frame]
        ):
            samples_per_frame[frame] = pd.concat(
                samples_per_frame[frame], ignore_index=True
            )
        else:
            # Empty list or unexpected type -> remove
            del samples_per_frame[frame]

    # Final safety: ensure all values are DataFrames and not empty
    for frame in list(samples_per_frame.keys()):
        if (
            not isinstance(samples_per_frame[frame], pd.DataFrame)
            or samples_per_frame[frame].empty
        ):
            del samples_per_frame[frame]

    if not samples_per_frame:
        show_error("No detections found in calibration tiles.")
        return None

    # ---------- Balance true positives across frames using rotations (as needed) ----------
    print("Balancing true positives across frames...")
    # Compute TP counts safely
    tp_counts = {}
    for frame, df in samples_per_frame.items():
        if isinstance(df, pd.DataFrame):
            tp_counts[frame] = df["ground_truth"].sum()
        else:
            del samples_per_frame[frame]

    if not tp_counts:
        show_error("No valid frames with samples for balancing.")
        return None

    max_tp = max(tp_counts.values())

    # Estimate total steps for progress bar
    total_rotation_steps = 0
    for frame, df in samples_per_frame.items():
        if isinstance(df, pd.DataFrame) and tp_counts[frame] < max_tp:
            total_rotation_steps += (
                len(tile_data_per_frame[frame]) * len(sigmas) * 3
            )  # 3 angles

    # Debug flag – set to True to save the first rotated tile
    DEBUG_ROTATION = True
    debug_saved = False

    with progress(
        total=total_rotation_steps, desc="Rotation balancing"
    ) as pbar:
        for frame, df in list(samples_per_frame.items()):
            if not isinstance(df, pd.DataFrame):
                continue
            current_tp = tp_counts[frame]
            if current_tp >= max_tp:
                continue

            added_dfs = []
            for angle in [90, 180, 270]:
                if current_tp >= max_tp:
                    break
                for tile_gray, points in tile_data_per_frame[frame]:
                    rot_tile, rot_points = rotate_tile_and_points(
                        tile_gray, points, angle
                    )

                    # Debug: save the first rotated tile with points and matched boxes
                    if DEBUG_ROTATION and not debug_saved:
                        # Run detection once with sigma=0 to get boxes and matches
                        # We'll use extract_detections_and_features but it only returns DataFrame,
                        # so we'll call detection separately.
                        phase_rgb = cv2.cvtColor(rot_tile, cv2.COLOR_GRAY2RGB)
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
                        dets = result.object_prediction_list
                        boxes_list = []
                        confs = []
                        for det in dets:
                            x1, x2, y1, y2 = (
                                int(det.bbox.minx),
                                int(det.bbox.maxx),
                                int(det.bbox.miny),
                                int(det.bbox.maxy),
                            )
                            boxes_list.append((x1, x2, y1, y2))
                            confs.append(det.score.value)
                        matched_boxes = match_points_to_boxes(
                            rot_points, boxes_list, confs
                        )

                        # Create debug folder
                        debug_folder = os.path.join(Save_folder, "debug")
                        os.makedirs(debug_folder, exist_ok=True)
                        debug_path = os.path.join(
                            debug_folder,
                            f"first_rotation_frame{frame}_angle{angle}_sigma0.png",
                        )
                        save_debug_rotation_image(
                            rot_tile,
                            rot_points,
                            boxes_list,
                            matched_boxes,
                            debug_path,
                            angle,
                            0,
                        )
                        debug_saved = True

                    # Now process all sigmas for this rotated tile
                    for sigma in sigmas:
                        np.random.seed(Random_seed + sigma + frame + angle)
                        aug_rot = apply_random_augmentations(
                            rot_tile, gamma_range=(0.5, 1.5)
                        )
                        blurred = blur_image(aug_rot, sigma)
                        df_rot = extract_detections_and_features(
                            blurred,
                            rot_points,
                            phase_model_low,
                            Sahi_size,
                            Sahi_overlap,
                            Postprocess,
                            Match_metric,
                            Intersection_threshold,
                            expand_scale=2.5,
                        )
                        if not df_rot.empty:
                            rot_tp = df_rot["ground_truth"].sum()
                            added_dfs.append(df_rot)
                            current_tp += rot_tp
                            if current_tp >= max_tp:
                                break
                        pbar.update(1)
                    if current_tp >= max_tp:
                        break
                if current_tp >= max_tp:
                    break

            if added_dfs:
                new_df = pd.concat(added_dfs, ignore_index=True)
                samples_per_frame[frame] = pd.concat(
                    [df, new_df], ignore_index=True
                )
                # Update tp_counts for this frame
                tp_counts[frame] = samples_per_frame[frame][
                    "ground_truth"
                ].sum()

    # Downsample TP per frame to max_tp (if overshoot)
    for frame in list(samples_per_frame.keys()):
        df = samples_per_frame[frame]
        if not isinstance(df, pd.DataFrame):
            del samples_per_frame[frame]
            continue
        tp_df = df[df["ground_truth"] == 1]
        fp_df = df[df["ground_truth"] == 0]
        if len(tp_df) > max_tp:
            tp_df = tp_df.sample(n=max_tp, random_state=Random_seed)
        samples_per_frame[frame] = pd.concat([tp_df, fp_df], ignore_index=True)

    # Combine all frames and balance classes globally
    all_dfs = [
        df for df in samples_per_frame.values() if isinstance(df, pd.DataFrame)
    ]
    if not all_dfs:
        show_error("No valid DataFrames after balancing.")
        return None
    full_df = pd.concat(all_dfs, ignore_index=True)
    tp = full_df[full_df["ground_truth"] == 1]
    fp = full_df[full_df["ground_truth"] == 0]

    if len(tp) == 0:
        show_error(
            "No true positive samples found. Check point-to-box matching."
        )
        return None

    # Balance classes globally: downsample the majority class to match the minority
    if len(tp) > len(fp):
        tp = tp.sample(n=len(fp), random_state=Random_seed)
    elif len(fp) > len(tp):
        fp = fp.sample(n=len(tp), random_state=Random_seed)
    # Now tp and fp have equal length

    balanced_df = pd.concat([tp, fp], ignore_index=True).sample(
        frac=1, random_state=Random_seed
    )

    if balanced_df.empty:
        show_error("Balanced training set is empty. Cannot train classifier.")
        return None

    print(
        f"Balanced training set size: {len(balanced_df)} (TP: {len(tp)}, FP: {len(fp)})"
    )

    # ---------- Train decision tree ----------
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

    param_grid = {
        "max_depth": [5, 7, 10, None],
        "min_samples_leaf": [1, 5, 10, 20, 50],
    }
    clf = DecisionTreeClassifier(random_state=Random_seed)
    grid_search = GridSearchCV(
        clf, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    best_clf = grid_search.best_estimator_

    y_val_pred = best_clf.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred)

    print(f"Best params: {grid_search.best_params_}")
    print(f"Validation F1: {val_f1:.4f}")

    # ---------- Testing phase ----------
    print("Testing on test tiles with different blurs...")
    results = []  # list of dicts: sigma, frame, gt_count, pred_count

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
                        y_pred = best_clf.predict(X_test)
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

    # ---------- Save results ----------
    subfolder = create_unique_subfolder(str(Save_folder), str(Experiment_name))
    dynamic_folder = os.path.join(subfolder, "Dynamic Confidence Threshold")
    os.makedirs(dynamic_folder, exist_ok=True)

    # Save decision tree
    model_path = os.path.join(dynamic_folder, "dynamic_threshold.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_clf, f)

    # Save reference points
    if points_data.ndim == 2:
        if points_data.shape[1] == 2:
            points_to_save = np.column_stack(
                (np.zeros(len(points_data), dtype=int), points_data)
            )
        else:
            points_to_save = points_data
    else:
        points_to_save = points_data
    pd.DataFrame(points_to_save, columns=["frame", "y", "x"]).to_csv(
        os.path.join(dynamic_folder, "reference_points.csv"), index=False
    )

    # Scatter plots per sigma
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

    # Metadata
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    metadata = f"""Experiment time: {current_date}
Calibration method: dynamic threshold (decision tree)
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
Best hyperparameters: {grid_search.best_params_}
Cross-validation F1 score (best): {grid_search.best_score_:.4f}
Validation F1 score: {val_f1:.4f}

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

    summary = f"Dynamic threshold calibrated. Best tree: {grid_search.best_params_}\n"
    for sigma, mape in mape_per_sigma.items():
        summary += f"Blur strength {sigma} MAPE: {mape:.2f}%\n"
    return summary
