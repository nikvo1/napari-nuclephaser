import itertools
import os
import pathlib
import warnings
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
from torch import cuda

from napari_nuclephaser.utils import create_unique_subfolder, initialize_model

warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)

matplotlib.use("Agg")

# cuda device check
cuda_available = "cuda:0" if cuda.is_available() else "cpu"

# find default models folder
models_folder = pathlib.Path(pathlib.Path(__file__).parent / "models")
first_model = next((x for x in models_folder.iterdir() if x.is_file()), None)
model_type_list = ("ultralytics", "yolov5", "yolov8", "yolov11", "yolo11")


def _ensure_numpy(arr):
    """Convert dask array to numpy if necessary."""
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

    Returns:
        cropped_images: list of tile image arrays (RGB)
        points_per_tile: list of integer point counts per tile
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


def _prepare_frame(
    image: np.ndarray,
    points_2d: list[tuple[float, float]],
    division_size: int,
    calibration_proportion: float,
    random_seed: int,
) -> tuple[list[np.ndarray], list[int], list[np.ndarray], list[int]]:
    """
    For a single frame:
        - convert image to RGB uint8 if needed
        - split into tiles
        - split tiles into calibration and test sets
    Returns:
        calibration_tiles, calibration_gt_counts, test_tiles, test_gt_counts
    """
    # Ensure image is a numpy array (convert from dask if needed)
    image = _ensure_numpy(image)

    # Convert to RGB if necessary
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.dtype == np.uint16:
        image = cv2.convertScaleAbs(image, alpha=255 / 65535).astype(np.uint8)

    # Split into tiles
    all_tiles, all_gt_counts = _split_image_and_points(
        image, points_2d, division_size
    )
    n_tiles = len(all_tiles)
    if n_tiles == 0:
        return [], [], [], []

    # Determine calibration / test split
    n_calib = int(n_tiles * calibration_proportion)
    np.random.seed(random_seed)
    calib_indices = np.random.choice(n_tiles, n_calib, replace=False)
    test_indices = np.setdiff1d(np.arange(n_tiles), calib_indices)

    calib_tiles = [all_tiles[i] for i in calib_indices]
    calib_gt = [all_gt_counts[i] for i in calib_indices]
    test_tiles = [all_tiles[i] for i in test_indices]
    test_gt = [all_gt_counts[i] for i in test_indices]

    return calib_tiles, calib_gt, test_tiles, test_gt


def _find_best_threshold_for_frame(
    tiles: list[np.ndarray],
    ground_truth_counts: list[int],
    model,
    sahi_size: int,
    sahi_overlap: float,
    postprocess: str,
    match_metric: str,
    intersection_threshold: float,
    frame_idx: int,
) -> float | None:
    """
    For a collection of tiles from a single frame, run the model with
    confidence thresholds from 0.01 to 0.99 and find the threshold that
    minimises the absolute difference between predicted and ground truth counts.
    Returns the best threshold (float). If no detections at all, returns None.
    """
    # Collect all detection confidences from all tiles
    all_confidences = []
    with progress(
        total=len(tiles), desc=f"Frame {frame_idx} calibration"
    ) as pbar:
        for tile in tiles:
            result = get_sliced_prediction(
                tile,
                model,
                slice_height=sahi_size,
                slice_width=sahi_size,
                overlap_height_ratio=sahi_overlap,
                overlap_width_ratio=sahi_overlap,
                postprocess_type=postprocess,
                postprocess_match_metric=match_metric,
                postprocess_match_threshold=intersection_threshold,
                verbose=0,
            )
            for box in result.object_prediction_list:
                all_confidences.append(box.score.value)
            pbar.update(1)

    if len(all_confidences) == 0:
        return None

    # Ground truth total for these tiles
    total_gt = sum(ground_truth_counts)

    best_threshold = 0.0
    best_diff = float("inf")
    for thr in np.arange(0.01, 1.0, 0.01):
        pred_count = sum(1 for c in all_confidences if c > thr)
        diff = abs(pred_count - total_gt)
        if diff < best_diff:
            best_diff = diff
            best_threshold = round(thr, 2)

    return best_threshold


def _test_frame(
    tiles: list[np.ndarray],
    ground_truth_counts: list[int],
    frame_idx: int,
    model,
    threshold: float,
    sahi_size: int,
    sahi_overlap: float,
    postprocess: str,
    match_metric: str,
    intersection_threshold: float,
) -> pd.DataFrame:
    """
    Run the calibrated model on a set of test tiles from one frame.
    Returns a DataFrame with columns: Frame, Ground_truth_count, Predicted_count.
    """
    results = []
    with progress(total=len(tiles), desc=f"Frame {frame_idx} testing") as pbar:
        for tile, gt in zip(tiles, ground_truth_counts, strict=False):
            result = get_sliced_prediction(
                tile,
                model,
                slice_height=sahi_size,
                slice_width=sahi_size,
                overlap_height_ratio=sahi_overlap,
                overlap_width_ratio=sahi_overlap,
                postprocess_type=postprocess,
                postprocess_match_metric=match_metric,
                postprocess_match_threshold=intersection_threshold,
                verbose=0,
            )
            pred = len(result.object_prediction_list)
            results.append(
                {
                    "Frame": frame_idx,
                    "Ground_truth_count": gt,
                    "Predicted_count": pred,
                }
            )
            pbar.update(1)
    return pd.DataFrame(results)


@magic_factory(
    Division_size={
        "max": 100000,
        "tooltip": "Tile size in pixels. Each image will be divided into small tiles of this size.",
    },
    Calibration_proportion={
        "tooltip": "Fraction of tiles used for calibration; the rest are used for testing."
    },
    Test_with_TTA={
        "widget_type": "CheckBox",
        "tooltip": "Run test‑time augmentation search to improve counting accuracy sacrificing the inference time.",
    },
    Postprocess={
        "choices": ["GREEDYNMM", "NMS", "NMM"],
        "tooltip": "Algorithm to process overlapping detections (see SAHI docs).",
    },
    Match_metric={
        "choices": ["IOS", "IOU"],
        "tooltip": "Metric to decide when two detections overlap (see SAHI docs).",
    },
    Save_folder={"mode": "d"},
    Sahi_size={
        "max": 100000,
        "tooltip": "Size of sliding window for sliced inference (pixels).",
    },
    Sahi_overlap={
        "tooltip": "Relative overlap between sliding windows (see SAHI docs).",
    },
    Intersection_threshold={
        "tooltip": "Threshold for merging overlapping detections (see SAHI docs).",
    },
    Experiment_name={
        "tooltip": "Name of the subfolder where results will be saved."
    },
    call_button="Calibrate",
    auto_call=False,
    result_widget=True,
)
def calibrate_with_points(
    Select_Phase_stack: Image,
    Select_Points_layer: Points,
    viewer: napari.Viewer,
    Phase_model=first_model,
    Division_size=640,
    Calibration_proportion=0.1,
    Test_with_TTA: bool = False,
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
    Calibrate a YOLO model on a stack of phase‑contrast images (or other modalities)
    using manually annotated points (3D points layer: frame, y, x).

    For each frame:
        - Split the image into tiles (Division_size).
        - Split tiles into calibration and test sets (Calibration_proportion).
        - Find the confidence threshold that gives the closest total detection count
          to the number of points in the calibration tiles.
    The final threshold is the average over all frames.
    Then test the model on all test tiles, compute overall and per‑frame MAPE,
    and produce a colour‑coded scatter plot.

    Returns a string summarising the best threshold and overall MAPE.
    """
    # --- Validate inputs -------------------------------------------------
    image_data = _ensure_numpy(Select_Phase_stack.data)
    points_data = _ensure_numpy(Select_Points_layer.data)

    if len(points_data) == 0:
        show_error("Points layer is empty! Can't proceed further")
        return None

    # Determine number of frames
    if image_data.ndim == 2 or (
        image_data.ndim == 3 and image_data.shape[-1] in (1, 3, 4)
    ):
        # Single image (2D or 3D with colour channel)
        n_frames = 1
        images = [image_data]
    elif image_data.ndim == 3:
        # Assume (T, H, W)
        n_frames = image_data.shape[0]
        images = [image_data[i] for i in range(n_frames)]
    elif image_data.ndim == 4 and image_data.shape[-1] in (1, 3, 4):
        # Assume (T, H, W, C)
        n_frames = image_data.shape[0]
        images = [image_data[i] for i in range(n_frames)]
    else:
        show_error(
            "Unsupported image dimensions. Provide a 2D image, a 3D stack (T, H, W), or (T, H, W, C)."
        )
        return None

    # Prepare points per frame
    if points_data.ndim == 2:
        if points_data.shape[1] == 2:
            # Single image points: (N, 2) -> all belong to frame 0
            if n_frames != 1:
                show_error(
                    "Points layer has 2 columns (y,x) but image stack has multiple frames. "
                    "Please use a 3D points layer with columns (frame, y, x)."
                )
                return None
            points_per_frame = {0: [(y, x) for y, x in points_data]}
        elif points_data.shape[1] == 3:
            # (N, 3) where each point = (frame, y, x)
            points_per_frame = {}
            for pt in points_data:
                t, y, x = int(pt[0]), pt[1], pt[2]
                points_per_frame.setdefault(t, []).append((y, x))
        else:
            show_error(
                f"Points layer has {points_data.shape[1]} columns. Expected 2 (y,x) for a single frame or 3 (frame,y,x) for a stack."
            )
            return None
    else:
        show_error(
            f"Points layer must be 2D (N,2) for a single image or 2D (N,3) for a stack. Got ndim={points_data.ndim}."
        )
        return None

    # Ensure we have points for each frame that exists in the image stack
    for t in range(n_frames):
        if t not in points_per_frame:
            points_per_frame[t] = []
            print(f"Warning: No points for frame {t}. Skipping this frame.")
    # Remove frames that have no images (just in case)
    frames_with_images = [t for t in range(n_frames) if t < len(images)]
    if not frames_with_images:
        show_error("No valid frames found.")
        return None

    # --- Initialise model (temporary low confidence for calibration) -----
    print("Initialising model for calibration...")
    phase_model, model_type = initialize_model(
        str(Phase_model), 0.01, cuda_available
    )
    print(f"Model ready. Type: {model_type}, device: {cuda_available}")

    # --- Per‑frame preparation and threshold finding ---------------------
    frame_data = []  # each element: {frame_idx, test_tiles, test_gt}
    thresholds_per_frame = []

    viewer.window._status_bar._toggle_activity_dock(True)

    # Outer progress bar for frames in calibration
    with progress(
        total=len(frames_with_images), desc="Running calibration"
    ) as pbar_outer:
        for t in frames_with_images:
            img = images[t]
            pts = points_per_frame.get(t, [])
            if len(pts) == 0:
                print(f"Frame {t} has no points, skipping.")
                pbar_outer.update(1)
                continue

            # Split into calibration and test tiles
            calib_tiles, calib_gt, test_tiles, test_gt = _prepare_frame(
                img, pts, Division_size, Calibration_proportion, Random_seed
            )
            if not calib_tiles:
                print(
                    f"Frame {t}: no calibration tiles (image too small or no points in tiles). Skipping."
                )
                pbar_outer.update(1)
                continue

            # Find best threshold for this frame (inner progress inside function)
            best_thr = _find_best_threshold_for_frame(
                calib_tiles,
                calib_gt,
                phase_model,
                Sahi_size,
                Sahi_overlap,
                Postprocess,
                Match_metric,
                Intersection_threshold,
                frame_idx=t,  # pass frame index for inner progress display
            )
            if best_thr is None:
                print(
                    f"Frame {t}: model made no detections on calibration tiles. Skipping."
                )
                pbar_outer.update(1)
                continue

            thresholds_per_frame.append(best_thr)
            frame_data.append(
                {
                    "frame_idx": t,
                    "test_tiles": test_tiles,
                    "test_gt": test_gt,
                }
            )
            print(f"Frame {t}: best threshold = {best_thr:.3f}")
            pbar_outer.update(1)

    if not thresholds_per_frame:
        show_error(
            "No valid calibration data. Model might not detect anything on your images."
        )
        return None

    # --- Average threshold ------------------------------------------------
    overall_threshold = np.mean(thresholds_per_frame)
    print(
        f"Calibration complete. Overall best threshold = {overall_threshold:.3f}"
    )

    # --- Initialise calibrated model -------------------------------------
    calibrated_model, _ = initialize_model(
        str(Phase_model), overall_threshold, cuda_available
    )

    # --- Test on all test tiles ------------------------------------------
    all_test_results = []
    # Outer progress bar for frames in testing
    with progress(
        total=len(frame_data), desc="Testing images"
    ) as pbar_outer_test:
        for fd in frame_data:
            df_frame = _test_frame(
                fd["test_tiles"],
                fd["test_gt"],
                fd["frame_idx"],
                calibrated_model,
                overall_threshold,
                Sahi_size,
                Sahi_overlap,
                Postprocess,
                Match_metric,
                Intersection_threshold,
            )
            all_test_results.append(df_frame)
            pbar_outer_test.update(1)

    # Filter out any empty DataFrames (e.g., frames with no test tiles)
    non_empty_results = [df for df in all_test_results if not df.empty]
    if not non_empty_results:
        print("No test tiles available. Skipping evaluation.")
        # Still save reference points and metadata (without plot)
        subfolder = create_unique_subfolder(
            str(Save_folder), str(Experiment_name)
        )

        # Save points (same as original)
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
            os.path.join(subfolder, "reference_points.csv"), index=False
        )

        # Save metadata without MAPE/plot
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metadata = f"""Experiment time: {current_date}
Calibration method: from points (multi‑frame)
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
Frames used: {[fd["frame_idx"] for fd in frame_data]}
Per‑frame thresholds: {dict(zip([fd["frame_idx"] for fd in frame_data], thresholds_per_frame, strict=False))}
Overall threshold: {overall_threshold:.3f}
Overall MAPE: No test data
Per‑frame MAPE: No test data
"""
        metadata_path = os.path.join(subfolder, "metadata.txt")
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(metadata)

        show_info("Calibration completed (no test tiles).")
        return f"Best threshold = {overall_threshold:.3f} (no test data)"

    # Otherwise, proceed with normal evaluation using non_empty_results
    test_df = pd.concat(non_empty_results, ignore_index=True)

    # --- Compute MAPE (skip tiles with ground truth = 0) -----------------
    test_df["AbsPercentageError"] = np.where(
        test_df["Ground_truth_count"] > 0,
        np.abs(
            (test_df["Ground_truth_count"] - test_df["Predicted_count"])
            / test_df["Ground_truth_count"]
        )
        * 100,
        np.nan,
    )
    overall_mape = test_df[
        "AbsPercentageError"
    ].mean()  # NaN are automatically skipped

    # Per‑frame MAPE
    per_frame_mape = (
        test_df.groupby("Frame")["AbsPercentageError"].mean().to_dict()
    )

    for t, mape in per_frame_mape.items():
        print(f"Frame {t}: MAPE = {mape:.2f}%")

    # --- Generate scatter plot with colour by frame ----------------------
    print("Drawing error plot...")
    sns.set(rc={"figure.dpi": 150, "savefig.dpi": 150})
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=test_df,
        x="Ground_truth_count",
        y="Predicted_count",
        hue="Frame",
        palette="tab10",
        ax=ax,
    )
    max_val = max(
        test_df["Ground_truth_count"].max(), test_df["Predicted_count"].max()
    )
    sns.lineplot(
        x=np.arange(0, max_val + 1),
        y=np.arange(0, max_val + 1),
        color="red",
        ax=ax,
    )
    ax.set_title(
        f"{Phase_model.name}\nThreshold = {overall_threshold:.3f}, Overall MAPE = {overall_mape:.2f}%"
    )
    ax.legend(title="Frame", bbox_to_anchor=(1.05, 1), loc="upper left")

    # --- Save results ----------------------------------------------------
    subfolder = create_unique_subfolder(str(Save_folder), str(Experiment_name))
    plot_path = os.path.join(subfolder, "Calibration_error_plot.png")
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    if points_data.ndim == 2:
        if points_data.shape[1] == 2:
            # add a dummy frame column for single frame
            points_to_save = np.column_stack(
                (np.zeros(len(points_data), dtype=int), points_data)
            )
        else:
            points_to_save = points_data
    else:
        points_to_save = points_data
    pd.DataFrame(points_to_save, columns=["frame", "y", "x"]).to_csv(
        os.path.join(subfolder, "reference_points.csv"), index=False
    )

    # Save metadata
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    metadata = f"""Experiment time: {current_date}
Calibration method: from points (multi‑frame)
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
Frames used: {[fd["frame_idx"] for fd in frame_data]}
Per‑frame thresholds: {dict(zip([fd["frame_idx"] for fd in frame_data], thresholds_per_frame, strict=False))}
Overall threshold: {overall_threshold:.3f}
Overall MAPE: {overall_mape:.2f}%
Per‑frame MAPE: {per_frame_mape}
"""
    metadata_path = os.path.join(subfolder, "metadata.txt")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write(metadata)

    if not Test_with_TTA:
        viewer.window._status_bar._toggle_activity_dock(False)
        return f"Best threshold = {overall_threshold:.3f}, Overall MAPE = {overall_mape:.2f}%"

    # ========================= TTA BLOCK =========================

    def native(img):
        return img

    def resize_1_5(img):
        return cv2.resize(
            img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC
        )

    def resize_2(img):
        return cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    def apply_clahe(img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        clahe_gray = clahe.apply(gray)
        return cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2RGB)

    def adjust_gamma(img, gamma=1.5):
        if img.dtype == np.uint8:
            inv_gamma = 1.0 / gamma
            table = np.array(
                [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
            ).astype(np.uint8)
            return cv2.LUT(img, table)
        normalized = img.astype(np.float32) / 255.0
        corrected = np.power(normalized, 1.0 / gamma) * 255.0
        return corrected.astype(np.uint8)

    def invert_image(img):
        return 255 - img

    def median_filter_3(img):
        return cv2.medianBlur(img, 3)

    def bilateral_filter_10(img):
        return cv2.bilateralFilter(img, -1, 10, 10)

    def unsharp_mask(img, sigma=1.0, strength=1.5):
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
        return sharpened

    augmentations = [
        (native, "native"),
        (resize_1_5, "resize_1.5x"),
        (resize_2, "resize_2x"),
        (apply_clahe, "clahe"),
        (adjust_gamma, "gamma_1.5"),
        (invert_image, "invert"),
        (median_filter_3, "median_3"),
        (bilateral_filter_10, "bilateral_10"),
        (unsharp_mask, "sharpen"),
    ]

    tta_thresholds = {}

    all_test_tiles_original = []
    all_test_gt = []
    all_frame_indices = []

    frame_calib_tiles = []
    frame_calib_gt = []
    frame_test_tiles = []
    frame_test_gt = []
    frame_indices = []

    with progress(
        total=len(frames_with_images), desc="Preparing tiles for TTA"
    ) as pbar:
        for t in frames_with_images:
            img = images[t]
            pts = points_per_frame.get(t, [])
            if len(pts) == 0:
                pbar.update(1)
                continue
            calib_tiles, calib_gt, test_tiles, test_gt = _prepare_frame(
                img, pts, Division_size, Calibration_proportion, Random_seed
            )
            if not calib_tiles or not test_tiles:
                pbar.update(1)
                continue
            frame_calib_tiles.append(calib_tiles)
            frame_calib_gt.append(calib_gt)
            frame_test_tiles.append(test_tiles)
            frame_test_gt.append(test_gt)
            frame_indices.append(t)

            # Also collect all test tiles (flat) for later combination search
            for tile, gt in zip(test_tiles, test_gt, strict=False):
                all_test_tiles_original.append(tile)
                all_test_gt.append(gt)
                all_frame_indices.append(t)
            pbar.update(1)

    if not all_test_tiles_original:
        show_error("No test tiles available for TTA. Aborting TTA.")
        return f"Native: threshold={overall_threshold:.3f}, MAPE={overall_mape:.2f}% (TTA skipped)"

    # For each augmentation, calibrate threshold using all frames' calibration tiles
    for aug_func, aug_name in augmentations:
        print(f"TTA calibration for: {aug_name}")
        thresholds_per_frame = []
        # Progress bar over frames for this augmentation
        with progress(
            total=len(frame_calib_tiles), desc=f"Calibration with {aug_name}"
        ) as pbar_cal:
            for idx, (calib_tiles, calib_gt) in enumerate(
                zip(frame_calib_tiles, frame_calib_gt, strict=False)
            ):
                aug_calib_tiles = [aug_func(tile) for tile in calib_tiles]
                best_thr = _find_best_threshold_for_frame(
                    aug_calib_tiles,
                    calib_gt,
                    phase_model,
                    Sahi_size,
                    Sahi_overlap,
                    Postprocess,
                    Match_metric,
                    Intersection_threshold,
                    frame_idx=frame_indices[idx],
                )
                if best_thr is not None:
                    thresholds_per_frame.append(best_thr)
                pbar_cal.update(1)
        if thresholds_per_frame:
            overall_thr = np.mean(thresholds_per_frame)
            tta_thresholds[aug_name] = overall_thr
            print(f"  {aug_name}: threshold = {overall_thr:.3f}")
        else:
            print(f"  {aug_name}: no valid calibration, skipping")
            tta_thresholds[aug_name] = None

    # Remove augmentations that failed calibration
    valid_augs = [
        name for name, thr in tta_thresholds.items() if thr is not None
    ]
    if not valid_augs:
        show_error("No augmentation produced a valid threshold. TTA aborted.")
        return f"Native: threshold={overall_threshold:.3f}, MAPE={overall_mape:.2f}% (TTA failed)"

    n_tiles = len(all_test_tiles_original)
    tile_predictions = np.zeros((n_tiles, len(valid_augs)), dtype=int)

    for i, aug_name in enumerate(valid_augs):
        thr = tta_thresholds[aug_name]
        aug_func = next(
            func for func, name in augmentations if name == aug_name
        )
        print(f"Testing augmentation: {aug_name} (thr={thr:.3f})")
        model_aug, _ = initialize_model(str(Phase_model), thr, cuda_available)
        pred_counts = []
        # Progress bar over tiles for this augmentation
        with progress(total=n_tiles, desc=f"Testing {aug_name}") as pbar:
            for tile in all_test_tiles_original:
                aug_tile = aug_func(tile)
                result = get_sliced_prediction(
                    aug_tile,
                    model_aug,
                    slice_height=Sahi_size,
                    slice_width=Sahi_size,
                    overlap_height_ratio=Sahi_overlap,
                    overlap_width_ratio=Sahi_overlap,
                    postprocess_type=Postprocess,
                    postprocess_match_metric=Match_metric,
                    postprocess_match_threshold=Intersection_threshold,
                    verbose=0,
                )
                pred_counts.append(len(result.object_prediction_list))
                pbar.update(1)
        tile_predictions[:, i] = pred_counts

    gt_array = np.array(all_test_gt)
    nonzero_mask = gt_array > 0

    best_mape = float("inf")
    best_combo = None
    best_avg_pred = None

    print("Searching for best combination of augmentations...")
    for r in range(1, len(valid_augs) + 1):
        for combo_indices in itertools.combinations(range(len(valid_augs)), r):
            avg_pred = np.mean(
                tile_predictions[:, list(combo_indices)], axis=1
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                pe = np.abs((gt_array - avg_pred) / gt_array) * 100
            mape = np.nanmean(pe[nonzero_mask])
            if mape < best_mape:
                best_mape = mape
                best_combo = [valid_augs[i] for i in combo_indices]
                best_avg_pred = avg_pred

    print(f"Best TTA combination: {' + '.join(best_combo)}")
    print(f"TTA MAPE: {best_mape:.2f}%")

    tta_subfolder = os.path.join(subfolder, "TTA")
    os.makedirs(tta_subfolder, exist_ok=True)

    fig_tta, ax_tta = plt.subplots()
    plot_df = pd.DataFrame(
        {
            "Ground_truth_count": gt_array,
            "Predicted_count": best_avg_pred,
            "Frame": all_frame_indices,
        }
    )
    sns.scatterplot(
        data=plot_df,
        x="Ground_truth_count",
        y="Predicted_count",
        hue="Frame",
        palette="tab10",
        ax=ax_tta,
    )
    max_val = max(
        plot_df["Ground_truth_count"].max(), plot_df["Predicted_count"].max()
    )
    sns.lineplot(
        x=np.arange(0, max_val + 1),
        y=np.arange(0, max_val + 1),
        color="red",
        ax=ax_tta,
    )
    ax_tta.set_title(
        f"TTA best combination: {' + '.join(best_combo)}\nMAPE = {best_mape:.2f}%"
    )
    ax_tta.legend(title="Frame", bbox_to_anchor=(1.05, 1), loc="upper left")
    tta_plot_path = os.path.join(
        tta_subfolder, "TTA_best_combination_plot.png"
    )
    fig_tta.savefig(tta_plot_path, bbox_inches="tight")
    plt.close(fig_tta)

    # Save metadata
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    tta_metadata = f"""Experiment time: {current_date}
TTA mode enabled
Native MAPE: {overall_mape:.2f}%
Best TTA combination: {' + '.join(best_combo)}
Best TTA MAPE: {best_mape:.2f}%

Per‑augmentation calibrated thresholds:
"""
    for name, thr in tta_thresholds.items():
        if thr is not None:
            tta_metadata += f"  {name}: {thr:.3f}\n"
        else:
            tta_metadata += f"  {name}: calibration failed\n"

    tta_metadata += f"""
All test tiles: {n_tiles} tiles (from frames {sorted(set(all_frame_indices))})
Random seed used: {Random_seed}
Division size: {Division_size}
Calibration proportion: {Calibration_proportion}
SAHI parameters: size={Sahi_size}, overlap={Sahi_overlap}
Postprocess: {Postprocess}, match_metric={Match_metric}, iou_thr={Intersection_threshold}
"""
    tta_metadata_path = os.path.join(tta_subfolder, "metadata_TTA.txt")
    with open(tta_metadata_path, "w", encoding="utf-8") as f:
        f.write(tta_metadata)

    show_info(
        f"TTA completed. Best MAPE: {best_mape:.2f}% (native: {overall_mape:.2f}%)"
    )

    # Return comparison string
    improvement = overall_mape - best_mape
    direction = "better" if improvement > 0 else "worse"

    viewer.window._status_bar._toggle_activity_dock(False)

    return (
        f"Native threshold = {overall_threshold:.3f}, MAPE = {overall_mape:.2f}%\n"
        f"TTA best combo ({' + '.join(best_combo)}) MAPE = {best_mape:.2f}% ({abs(improvement):.2f}% {direction})"
    )
