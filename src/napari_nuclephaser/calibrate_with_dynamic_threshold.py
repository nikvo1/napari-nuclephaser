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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from torch import cuda

try:
    from skimage.util import view_as_windows
except ImportError:
    from numpy.lib.stride_tricks import sliding_window_view as view_as_windows

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


def compute_tile_features(tile_gray, detections):
    stats = extract_features_grayscale(tile_gray)
    features = {
        "height/width": stats["height/width"],
        "mean": stats["mean"],
        "std": stats["std"],
        "median": stats["median"],
        "p25": stats["p25"],
        "p75": stats["p75"],
        "iqr": stats["iqr"],
        "min": stats["min"],
        "max": stats["max"],
        "range": stats["range"],
        "rms_contrast": stats["rms_contrast"],
        "lap_var": stats["lap_var"],
        "entropy": stats["entropy"],
        "energy": stats["energy"],
    }

    h, w = tile_gray.shape
    area = h * w
    thresholds = [0.01, 0.1, 0.2, 0.3]
    confs = [d[4] for d in detections]
    for thr in thresholds:
        count = sum(1 for c in confs if c >= thr)
        features[f"density_{thr:.2f}"] = count / area if area > 0 else 0.0

    if detections:
        sorted_dets = sorted(detections, key=lambda x: x[4], reverse=True)
        n_top = max(1, int(0.1 * len(sorted_dets)))
        top_dets = sorted_dets[:n_top]
        areas = [(x2 - x1) * (y2 - y1) for (x1, x2, y1, y2, _) in top_dets]
        features["top10_area"] = np.mean(areas) if areas else 0.0
    else:
        features["top10_area"] = 0.0

    return features


def find_optimal_threshold(detections, gt_count):
    if not detections:
        return None
    confs = [d[4] for d in detections]
    best_thr = 0.01
    best_error = float("inf")
    for thr in np.arange(0.01, 1.0, 0.01):
        pred = sum(1 for c in confs if c >= thr)
        error = abs(pred - gt_count)
        if error < best_error:
            best_error = error
            best_thr = round(thr, 2)
    return best_thr


def find_static_threshold(all_confs, all_gts):
    if not all_confs:
        return 0.0, float("inf")
    best_thr = 0.01
    best_error = float("inf")
    for thr in np.arange(0.01, 1.0, 0.01):
        total_error = 0
        for confs, gt in zip(all_confs, all_gts, strict=False):
            pred = sum(1 for c in confs if c >= thr)
            total_error += abs(pred - gt)
        if total_error < best_error:
            best_error = total_error
            best_thr = round(thr, 2)
    return best_thr, best_error


def build_threshold_map(
    tile_gray, detections, regressor, scaler, feature_names, win_size, stride
):
    h, w = tile_gray.shape
    if win_size > h or win_size > w:
        win_size = max(h, w)
        stride = win_size

    x_starts = list(range(0, w - win_size + 1, stride))
    y_starts = list(range(0, h - win_size + 1, stride))
    if x_starts[-1] + win_size < w:
        x_starts.append(w - win_size)
    if y_starts[-1] + win_size < h:
        y_starts.append(h - win_size)

    windows = view_as_windows(tile_gray, (win_size, win_size), step=stride)
    n_y, n_x, win_h, win_w = windows.shape

    windows_flat = windows.reshape(-1, win_h, win_w).astype(np.float64)

    means = np.mean(windows_flat, axis=(1, 2))
    stds = np.std(windows_flat, axis=(1, 2))
    medians = np.median(windows_flat, axis=(1, 2))
    p25s = np.percentile(windows_flat, 25, axis=(1, 2))
    p75s = np.percentile(windows_flat, 75, axis=(1, 2))
    iqrs = p75s - p25s
    mins = np.min(windows_flat, axis=(1, 2))
    maxs = np.max(windows_flat, axis=(1, 2))
    ranges = maxs - mins
    rms_contrasts = np.sqrt(
        np.mean((windows_flat - means[:, None, None]) ** 2, axis=(1, 2))
    )

    lap_full = cv2.Laplacian(tile_gray, cv2.CV_64F, ksize=3)
    lap_windows = view_as_windows(lap_full, (win_size, win_size), step=stride)
    lap_windows_flat = lap_windows.reshape(-1, win_size, win_size)
    lap_vars = np.var(lap_windows_flat, axis=(1, 2))

    entropies = np.zeros(windows_flat.shape[0])
    for idx, win in enumerate(windows_flat):
        hist, _ = np.histogram(win, bins=256, range=(0, 255))
        hist = hist.astype(np.float64)
        hist /= hist.sum() + 1e-12
        entropies[idx] = -np.sum(hist * np.log2(hist + 1e-12))

    energies = np.sum(windows_flat**2, axis=(1, 2))
    height_width_ratio = win_size / win_size

    feature_arrays = {
        "height/width": np.full(n_y * n_x, height_width_ratio),
        "mean": means,
        "std": stds,
        "median": medians,
        "p25": p25s,
        "p75": p75s,
        "iqr": iqrs,
        "min": mins,
        "max": maxs,
        "range": ranges,
        "rms_contrast": rms_contrasts,
        "lap_var": lap_vars,
        "entropy": entropies,
        "energy": energies,
    }

    centers = np.array(
        [((x1 + x2) / 2, (y1 + y2) / 2) for (x1, x2, y1, y2, _) in detections]
    )
    confs = np.array([conf for (_, _, _, _, conf) in detections])

    x_indices = np.searchsorted(x_starts, centers[:, 0], side="right") - 1
    y_indices = np.searchsorted(y_starts, centers[:, 1], side="right") - 1
    x_indices = np.clip(x_indices, 0, n_x - 1)
    y_indices = np.clip(y_indices, 0, n_y - 1)

    thresholds = [0.01, 0.1, 0.2, 0.3]
    density_grids = {}
    for thr in thresholds:
        mask = confs >= thr
        counts = np.zeros((n_y, n_x), dtype=np.float64)
        if np.any(mask):
            np.add.at(counts, (y_indices[mask], x_indices[mask]), 1)
        density_grids[f"density_{thr:.2f}"] = counts / (win_size * win_size)

    for thr in thresholds:
        key = f"density_{thr:.2f}"
        feature_arrays[key] = density_grids[key].ravel()

    sorted_dets = sorted(detections, key=lambda d: d[4], reverse=True)
    n_top = max(1, int(0.1 * len(sorted_dets)))
    top_dets = sorted_dets[:n_top]
    areas = [(x2 - x1) * (y2 - y1) for (x1, x2, y1, y2, _) in top_dets]
    top10_area = np.mean(areas) if areas else 0.0
    feature_arrays["top10_area"] = np.full(n_y * n_x, top10_area)

    X_win = np.column_stack([feature_arrays[name] for name in feature_names])
    X_win_scaled = scaler.transform(X_win)
    thresholds_pred = regressor.predict(X_win_scaled)
    threshold_grid = thresholds_pred.reshape(n_y, n_x)

    window_centers = []
    for yi, y0 in enumerate(y_starts):
        for xi, x0 in enumerate(x_starts):
            win_cx = x0 + win_size / 2
            win_cy = y0 + win_size / 2
            thr = threshold_grid[yi, xi]
            window_centers.append((win_cx, win_cy, thr))
    return window_centers


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
        "tooltip": "Maximum defocus strength that the model will adapt to. 0 skips blur augmentation.",
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
    Postprocess="NMS",
    Match_metric="IOS",
    Intersection_threshold=0.34,
    Sahi_size=640,
    Sahi_overlap: float = 0.2,
):
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
                show_error(
                    "Points have 2 columns but multiple frames. Select the valid points layer."
                )
                return None
            points_per_frame = {0: [(y, x) for y, x in points_data]}
        elif points_data.shape[1] == 3:
            points_per_frame = defaultdict(list)
            for pt in points_data:
                t, y, x = int(pt[0]), pt[1], pt[2]
                points_per_frame[t].append((y, x))
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

    viewer.window._status_bar._toggle_activity_dock(True)

    initialization_pbar = progress(total=0, desc="Initializing model")
    phase_model_low, model_type = initialize_model(
        str(Phase_model), 0.01, cuda_available
    )
    initialization_pbar.close()

    calib_data = []
    test_data = []

    with progress(
        total=len(frames_with_images), desc="Splitting frames"
    ) as pbar:
        for t in frames_with_images:
            img = images[t]
            pts = points_per_frame[t]
            if len(pts) == 0:
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

            all_tiles, all_gt_counts = _split_image_and_points(
                rgb_img, pts, Division_size
            )
            n_tiles = len(all_tiles)
            if n_tiles == 0:
                pbar.update(1)
                continue

            tile_infos = []
            for i in range(n_tiles):
                left = (
                    i % (rgb_img.shape[1] // Division_size)
                ) * Division_size
                upper = (
                    i // (rgb_img.shape[1] // Division_size)
                ) * Division_size
                gt_count = all_gt_counts[i]
                tile_gray = cv2.cvtColor(all_tiles[i], cv2.COLOR_RGB2GRAY)
                tile_infos.append(
                    {
                        "tile_gray": tile_gray,
                        "gt_count": gt_count,
                        "left": left,
                        "upper": upper,
                    }
                )

            np.random.seed(Random_seed)
            indices = np.arange(n_tiles)
            np.random.shuffle(indices)
            n_calib = int(n_tiles * Calibration_proportion)
            calib_indices = indices[:n_calib]
            test_indices = indices[n_calib:]

            for idx in calib_indices:
                info = tile_infos[idx]
                calib_data.append(
                    {
                        "frame_idx": t,
                        "tile_gray": info["tile_gray"],
                        "gt_count": info["gt_count"],
                        "tile_left": info["left"],
                        "tile_upper": info["upper"],
                    }
                )

            for idx in test_indices:
                info = tile_infos[idx]
                test_data.append(
                    {
                        "frame_idx": t,
                        "tile_gray": info["tile_gray"],
                        "gt_count": info["gt_count"],
                        "tile_left": info["left"],
                        "tile_upper": info["upper"],
                    }
                )

            pbar.update(1)

    if not calib_data:
        show_error("No calibration tiles found (no points in any tile).")
        return None

    samples_X = []
    samples_y = []

    static_calib_confs = []
    static_calib_gts = []

    total_steps = len(calib_data) * len(sigmas)
    with progress(total=total_steps, desc="Running calibration") as pbar:
        for entry in calib_data:
            tile_gray = entry["tile_gray"]
            gt_count = entry["gt_count"]
            frame = entry["frame_idx"]

            if gt_count == 0:
                for _ in sigmas:
                    pbar.update(1)
                continue

            for sigma in sigmas:
                np.random.seed(Random_seed + sigma + frame)
                aug_tile = apply_random_augmentations(tile_gray)
                blurred = blur_image(aug_tile, sigma)

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
                detections = []
                for det in result.object_prediction_list:
                    x1, x2, y1, y2 = (
                        int(det.bbox.minx),
                        int(det.bbox.maxx),
                        int(det.bbox.miny),
                        int(det.bbox.maxy),
                    )
                    conf = det.score.value
                    detections.append((x1, x2, y1, y2, conf))

                if sigma == 0:
                    confs = [d[4] for d in detections]
                    static_calib_confs.append(confs)
                    static_calib_gts.append(gt_count)

                features = compute_tile_features(blurred, detections)
                opt_thr = find_optimal_threshold(detections, gt_count)
                if opt_thr is None:
                    pbar.update(1)
                    continue

                feature_vector = [features[k] for k in sorted(features.keys())]
                samples_X.append(feature_vector)
                samples_y.append(opt_thr)

                pbar.update(1)

    if not samples_X:
        show_error(
            "No training samples collected (no detections or no reference points)."
        )
        return None

    X = np.array(samples_X)
    y = np.array(samples_y)

    minimal_threshold = min(samples_y) if samples_y else 0.01
    static_threshold, _ = find_static_threshold(
        static_calib_confs, static_calib_gts
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    regressor = KNeighborsRegressor(n_neighbors=1)
    regressor.fit(X_scaled, y)

    feature_names = sorted(
        compute_tile_features(np.zeros((10, 10), dtype=np.uint8), []).keys()
    )

    results_dynamic = []
    test_data_by_sigma = defaultdict(list)

    win_size = Sahi_size // 2
    stride = win_size // 2

    with progress(
        total=len(test_data) * len(sigmas), desc="Running test"
    ) as pbar:
        for entry in test_data:
            tile_gray = entry["tile_gray"]
            gt_count = entry["gt_count"]
            frame = entry["frame_idx"]
            tile_left = entry["tile_left"]
            tile_upper = entry["tile_upper"]

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
                detections = []
                for det in result.object_prediction_list:
                    x1, x2, y1, y2 = (
                        int(det.bbox.minx),
                        int(det.bbox.maxx),
                        int(det.bbox.miny),
                        int(det.bbox.maxy),
                    )
                    conf = det.score.value
                    detections.append((x1, x2, y1, y2, conf))

                window_centers = build_threshold_map(
                    blurred,
                    detections,
                    regressor,
                    scaler,
                    feature_names,
                    win_size,
                    stride,
                )

                sigma_val = win_size / 4.0
                pred_count = 0
                for x1, x2, y1, y2, conf in detections:
                    px = (x1 + x2) / 2
                    py = (y1 + y2) / 2
                    total_weight = 0.0
                    weighted_thr = 0.0
                    for cx, cy, thr in window_centers:
                        half = win_size / 2
                        if (cx - half) <= px <= (cx + half) and (
                            cy - half
                        ) <= py <= (cy + half):
                            dist2 = (px - cx) ** 2 + (py - cy) ** 2
                            weight = np.exp(-dist2 / (2 * sigma_val**2))
                            weighted_thr += weight * thr
                            total_weight += weight
                    if total_weight > 0:
                        final_thr = weighted_thr / total_weight
                    else:
                        min_dist = float("inf")
                        final_thr = 0.5
                        for cx, cy, thr in window_centers:
                            dist = (px - cx) ** 2 + (py - cy) ** 2
                            if dist < min_dist:
                                min_dist = dist
                                final_thr = thr
                    if conf >= final_thr:
                        pred_count += 1

                results_dynamic.append(
                    {
                        "sigma": sigma,
                        "frame": frame,
                        "gt_count": gt_count,
                        "pred_count": pred_count,
                    }
                )

                confs = [d[4] for d in detections]
                test_data_by_sigma[sigma].append(
                    {
                        "gt_count": gt_count,
                        "confs": confs,
                        "frame": frame,
                        "tile_left": tile_left,
                        "tile_upper": tile_upper,
                    }
                )

                pbar.update(1)

    df_dynamic = pd.DataFrame(results_dynamic)

    mape_per_sigma_dynamic = {}
    for sigma in sigmas:
        sub = df_dynamic[df_dynamic["sigma"] == sigma]
        sub = sub[sub["gt_count"] > 0]
        if not sub.empty:
            pe = (
                np.abs((sub["gt_count"] - sub["pred_count"]) / sub["gt_count"])
                * 100
            )
            mape_per_sigma_dynamic[sigma] = pe.mean()
        else:
            mape_per_sigma_dynamic[sigma] = np.nan

    overall_mape_dynamic = (
        np.nanmean(list(mape_per_sigma_dynamic.values()))
        if mape_per_sigma_dynamic
        else np.nan
    )

    results_static = []
    for sigma, data in test_data_by_sigma.items():
        for d in data:
            pred = sum(1 for c in d["confs"] if c >= static_threshold)
            results_static.append(
                {
                    "sigma": sigma,
                    "frame": d["frame"],
                    "gt_count": d["gt_count"],
                    "pred_count": pred,
                }
            )
    df_static = pd.DataFrame(results_static)

    mape_per_sigma_static = {}
    for sigma in sigmas:
        sub = df_static[df_static["sigma"] == sigma]
        sub = sub[sub["gt_count"] > 0]
        if not sub.empty:
            pe = (
                np.abs((sub["gt_count"] - sub["pred_count"]) / sub["gt_count"])
                * 100
            )
            mape_per_sigma_static[sigma] = pe.mean()
        else:
            mape_per_sigma_static[sigma] = np.nan

    overall_mape_static = (
        np.nanmean(list(mape_per_sigma_static.values()))
        if mape_per_sigma_static
        else np.nan
    )

    subfolder = create_unique_subfolder(str(Save_folder), str(Experiment_name))
    dynamic_folder = os.path.join(subfolder, "Dynamic Confidence Threshold")
    os.makedirs(dynamic_folder, exist_ok=True)

    sns.set(rc={"figure.dpi": 150, "savefig.dpi": 150})
    for sigma in sigmas:
        sub_dyn = df_dynamic[df_dynamic["sigma"] == sigma]
        sub_stat = df_static[df_static["sigma"] == sigma]
        if sub_dyn.empty and sub_stat.empty:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        if not sub_dyn.empty:
            sns.scatterplot(
                data=sub_dyn,
                x="gt_count",
                y="pred_count",
                hue="frame",
                palette="tab10",
                ax=ax1,
            )
            max_val = max(
                sub_dyn["gt_count"].max(), sub_dyn["pred_count"].max()
            )
            sns.lineplot(
                x=np.arange(0, max_val + 1),
                y=np.arange(0, max_val + 1),
                color="red",
                ax=ax1,
            )
            mape_dyn = mape_per_sigma_dynamic.get(sigma, np.nan)
            ax1.set_title(
                f"Dynamic (kNN threshold map)\nMAPE = {mape_dyn:.2f}%"
                if not np.isnan(mape_dyn)
                else "Dynamic (kNN threshold map)\nMAPE = N/A"
            )
            ax1.legend(
                title="Frame", bbox_to_anchor=(1.05, 1), loc="upper left"
            )
        else:
            ax1.text(
                0.5,
                0.5,
                "No dynamic data",
                ha="center",
                va="center",
                transform=ax1.transAxes,
            )

        if not sub_stat.empty:
            sns.scatterplot(
                data=sub_stat,
                x="gt_count",
                y="pred_count",
                hue="frame",
                palette="tab10",
                ax=ax2,
            )
            max_val = max(
                sub_stat["gt_count"].max(), sub_stat["pred_count"].max()
            )
            sns.lineplot(
                x=np.arange(0, max_val + 1),
                y=np.arange(0, max_val + 1),
                color="red",
                ax=ax2,
            )
            mape_stat = mape_per_sigma_static.get(sigma, np.nan)
            ax2.set_title(
                f"Static (global threshold = {static_threshold:.2f})\nMAPE = {mape_stat:.2f}%"
                if not np.isnan(mape_stat)
                else f"Static (global threshold = {static_threshold:.2f})\nMAPE = N/A"
            )
            ax2.legend(
                title="Frame", bbox_to_anchor=(1.05, 1), loc="upper left"
            )
        else:
            ax2.text(
                0.5,
                0.5,
                "No static data",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )

        plt.tight_layout()
        plot_path = os.path.join(
            dynamic_folder, f"Error_plot_for_blur_sigma_{sigma}.png"
        )
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)

    model_path = os.path.join(dynamic_folder, "dynamic_threshold.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "regressor": regressor,
                "scaler": scaler,
                "feature_names": feature_names,
                "minimal_threshold": minimal_threshold,
            },
            f,
        )

    points_to_save = []
    for frame, pts in points_per_frame.items():
        for y, x in pts:
            points_to_save.append([frame, y, x])
    if points_to_save:
        pd.DataFrame(points_to_save, columns=["frame", "y", "x"]).to_csv(
            os.path.join(dynamic_folder, "reference_points.csv"), index=False
        )

    comparison_summary = ""
    for sigma in sigmas:
        mape_dyn = mape_per_sigma_dynamic.get(sigma, np.nan)
        mape_stat = mape_per_sigma_static.get(sigma, np.nan)
        comparison_summary += f"Sigma {sigma}: Dynamic MAPE = {mape_dyn:.2f}%, Static MAPE = {mape_stat:.2f}%\n"

    current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    metadata = f"""Experiment time: {current_date}
Calibration method: dynamic threshold with points
Image: {Select_Phase_stack.name}, shape {image_data.shape}
Model: {Phase_model.name} ({model_type})
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
Number of training samples (tiles × sigmas): {len(X)}
Feature names: {feature_names}
k‑NN k = 1
Minimal optimal threshold (for inference initialisation): {minimal_threshold:.2f}

--- Static threshold (from sigma=0 calibration) ---
Static threshold: {static_threshold:.2f}

--- Testing results (using threshold map) ---
Dynamic threshold per-sigma MAPE (over tiles with gt>0):
"""
    for sigma, mape in mape_per_sigma_dynamic.items():
        metadata += f"  Blur strength {sigma}: {mape:.2f}%\n"

    metadata += f"\nOverall Dynamic MAPE: {overall_mape_dynamic:.2f}%\n\n"

    metadata += "Static threshold per-sigma MAPE (over tiles with gt>0):\n"
    for sigma, mape in mape_per_sigma_static.items():
        metadata += f"  Blur strength {sigma}: {mape:.2f}%\n"

    metadata += f"\nOverall Static MAPE: {overall_mape_static:.2f}%\n\n"

    metadata += "--- Comparison summary ---\n"
    metadata += comparison_summary

    metadata_path = os.path.join(dynamic_folder, "metadata.txt")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write(metadata)

    viewer.window._status_bar._toggle_activity_dock(False)
    show_info("Dynamic threshold calibration completed.")

    summary = (
        f"Calibration completed.\n"
        f"Static threshold = {static_threshold:.2f} (overall MAPE = {overall_mape_static:.2f}%)\n"
        f"Dynamic (kNN threshold map) overall MAPE = {overall_mape_dynamic:.2f}%\n"
        f"Minimal optimal threshold = {minimal_threshold:.2f}\n"
        f"Comparison:\n{comparison_summary}"
    )
    return summary
