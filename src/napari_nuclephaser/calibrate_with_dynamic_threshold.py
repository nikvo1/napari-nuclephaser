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
    """
    Split a single image into tiles of size window_size x window_size.
    For each tile, count how many of the given points fall inside it
    (points are given as (y, x) relative to the full image).
    """
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
    """
    Compute global image features + detection‑based features.
    detections: list of (x1, x2, y1, y2, conf)
    Returns: dict of feature names -> values
    """
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
    """
    Scan thresholds from 0.01 to 0.99 (step 0.01) and pick the one that
    minimises absolute counting error |pred_count - gt_count|.
    If no detections, return None.
    """
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
    """
    Calibrate a dynamic threshold using patch‑level features and k‑NN regression.
    Ground truth is provided as points (cell centers). For each tile and blur sigma,
    we extract global image features, detection densities, and the optimal threshold
    (minimising counting error). A KNeighborsRegressor(k=1) is trained to predict
    the optimal threshold from features.
    """
    image_data = _ensure_numpy(Select_Phase_stack.data)
    points_data = _ensure_numpy(Select_Points_layer.data)

    if len(points_data) == 0:
        show_error("Points layer is empty!")
        return None

    # Determine number of frames
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

    # Parse points per frame
    if points_data.ndim == 2:
        if points_data.shape[1] == 2:
            if n_frames != 1:
                show_error(
                    "Points have 2 columns but multiple frames. Use (frame, y, x)."
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

    # Ensure each frame has a list (even if empty)
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

    print("Initialising model with confidence=0.01 ...")
    phase_model_low, model_type = initialize_model(
        str(Phase_model), 0.01, cuda_available
    )
    print(f"Model ready: {model_type}, device: {cuda_available}")

    # Split frames into calibration/test tiles
    calib_data = (
        []
    )  # each: {frame_idx, tile_gray, gt_count, tile_left, tile_upper}
    test_data = []
    viewer.window._status_bar._toggle_activity_dock(True)

    with progress(
        total=len(frames_with_images), desc="Splitting frames"
    ) as pbar:
        for t in frames_with_images:
            img = images[t]
            pts = points_per_frame[t]
            if len(pts) == 0:
                pbar.update(1)
                continue

            # Convert to grayscale and RGB
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

            # Split into calibration and test
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

    print("Collecting training samples from calibration tiles...")
    samples_X = []
    samples_y = []

    total_steps = len(calib_data) * len(sigmas)
    with progress(
        total=total_steps, desc="Calibration feature extraction"
    ) as pbar:
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
            "No training samples collected (no detections or all gt=0)."
        )
        return None

    X = np.array(samples_X)
    y = np.array(samples_y)

    print(f"Collected {len(X)} training samples.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    regressor = KNeighborsRegressor(n_neighbors=1)
    regressor.fit(X_scaled, y)

    feature_names = sorted(
        compute_tile_features(np.zeros((10, 10), dtype=np.uint8), []).keys()
    )

    # --- Testing phase ---
    print("Testing on test tiles...")
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

                features = compute_tile_features(blurred, detections)
                feat_vec = [features[k] for k in feature_names]
                X_test = np.array([feat_vec])
                X_test_scaled = scaler.transform(X_test)
                predicted_thr = regressor.predict(X_test_scaled)[0]

                confs = [d[4] for d in detections]
                pred_count = sum(1 for c in confs if c >= predicted_thr)

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
            mape_per_sigma[sigma] = pe.mean()
        else:
            mape_per_sigma[sigma] = np.nan

    overall_mape = (
        np.nanmean(list(mape_per_sigma.values())) if mape_per_sigma else np.nan
    )

    print(f"Overall MAPE: {overall_mape:.2f}%")

    # --- Save results ---
    subfolder = create_unique_subfolder(str(Save_folder), str(Experiment_name))
    dynamic_folder = os.path.join(subfolder, "Dynamic Confidence Threshold")
    os.makedirs(dynamic_folder, exist_ok=True)

    # Save regressor and scaler
    model_path = os.path.join(dynamic_folder, "knn_threshold_regressor.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "regressor": regressor,
                "scaler": scaler,
                "feature_names": feature_names,
            },
            f,
        )

    # Save reference points
    points_to_save = []
    for frame, pts in points_per_frame.items():
        for y, x in pts:
            points_to_save.append([frame, y, x])
    if points_to_save:
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
Calibration method: dynamic threshold (k‑NN regression on patch features) using POINTS
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
Number of training samples (tiles × sigmas): {len(X)}
Feature names: {feature_names}
k‑NN k = 1

--- Testing results ---
Per-sigma MAPE (over tiles with gt>0):
"""
    for sigma, mape in mape_per_sigma.items():
        metadata += f"  Blur strength {sigma}: {mape:.2f}%\n"

    metadata += f"\nOverall MAPE: {overall_mape:.2f}%\n"

    metadata_path = os.path.join(dynamic_folder, "metadata.txt")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write(metadata)

    viewer.window._status_bar._toggle_activity_dock(False)
    show_info("Dynamic threshold calibration (kNN) completed.")

    summary = f"Calibration completed. Overall MAPE = {overall_mape:.2f}%"
    return summary
