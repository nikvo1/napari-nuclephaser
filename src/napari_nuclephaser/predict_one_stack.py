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

try:
    from skimage.util import view_as_windows
except ImportError:
    from numpy.lib.stride_tricks import sliding_window_view as view_as_windows

from napari_nuclephaser.utils import create_unique_subfolder, initialize_model

warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)

cuda_available = "cuda:0" if cuda.is_available() else "cpu"

models_folder = pathlib.Path(pathlib.Path(__file__).parent / "models")
first_model = next((x for x in models_folder.iterdir() if x.is_file()), None)


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

    # top10_area computed from detections (per frame)
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
    Postprocess={
        "choices": ["GREEDYNMM", "NMS", "NMM"],
        "tooltip": "An algorithm to process overlapping detections. See obss/sahi library docs for more details.",
    },
    Match_metric={
        "choices": ["IOS", "IOU"],
        "tooltip": "A metric to determine when two detections are two different detections overlapping or is it a one detection. Sett obss/sahi library docs for more details",
    },
    Detection_mode={
        "choices": [
            "Regular detection",
            "Detection with TTA",
            "Detection with Dynamic threshold",
        ],
        "tooltip": "Select the prediction Detection_mode: standard (None), Test-Time Augmentation (TTA), or dynamic threshold filtering.",
    },
    Mode_file={
        "mode": "r",
        "filter": "*.txt;*.pkl",
        "tooltip": "Select the TTA metadata file (.txt) or the dynamic threshold .pkl file, depending on the chosen Detection_mode.",
    },
    ADVANCED_SETTINGS={},
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
    Save_format={
        "choices": ["CSV", "XLSX", "Both"],
        "value": "CSV",
        "tooltip": "Select the output format for saving results (when Save_result is enabled).",
    },
    Save_folder={"mode": "d"},
    call_button="Predict",
    auto_call=False,
    result_widget=False,
)
def predict_on_stack(
    Select_stack: Image,
    viewer: napari.Viewer,
    Select_model=first_model,
    Confidence_threshold: float = 0.2,
    Detection_mode="Regular detection",
    Mode_file=pathlib.Path(),
    Save_result=True,
    Save_folder=pathlib.Path(),
    Experiment_name="Experiment",
    Save_format="CSV",
    ADVANCED_SETTINGS="",
    Postprocess="NMS",
    Match_metric="IOS",
    Sahi_size=640,
    Sahi_overlap: float = 0.2,
    Intersection_threshold=0.34,
    Points_size=10,
):
    use_tta = Detection_mode == "Detection with TTA"
    use_dynamic = Detection_mode == "Detection with Dynamic threshold"

    if Detection_mode != "Regular detection":
        if not Mode_file or not Mode_file.exists():
            show_error(
                f"Detection mode '{Detection_mode}' requires a valid mode file. See docs for more details."
            )
            return None

        if use_tta and Mode_file.suffix.lower() != ".txt":
            show_error(
                "TTA detection mode requires a .txt metadata file. See docs for more details."
            )
            return None
        if use_dynamic and Mode_file.suffix.lower() != ".pkl":
            show_error(
                "Dynamic threshold detection mode requires a .pkl file. See docs for more details."
            )
            return None

    pic = Select_stack.data
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

    if use_tta:
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

        all_aug_points = []
        all_aug_counts = []

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

                aug_points = []
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
                                    desc=f"Sliced prediction ({aug_name}) frame {frame_idx+1}",
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
                            bbox = instance["bbox"]
                            if scale_factor != 1.0:
                                center_x = int(
                                    (bbox[0] + bbox[2] // 2) * scale_factor
                                )
                                center_y = int(
                                    (bbox[1] + bbox[3] // 2) * scale_factor
                                )
                            else:
                                center_x = int(bbox[0] + bbox[2] // 2)
                                center_y = int(bbox[1] + bbox[3] // 2)
                            aug_points.append([i, center_y, center_x])
                            frame_count += 1
                        frame_counts.append(frame_count)
                        pbar_frames.update(1)

                all_aug_points.append(
                    np.array(aug_points) if aug_points else np.empty((0, 3))
                )
                all_aug_counts.append(frame_counts)
                pbar_augs.update(1)

        n_frames = len(pic)
        avg_counts_per_frame = []
        for f in range(n_frames):
            counts = [
                aug_counts[f]
                for aug_counts in all_aug_counts
                if f < len(aug_counts)
            ]
            avg_counts_per_frame.append(np.mean(counts) if counts else 0)

        for idx, aug_name in enumerate(best_augs):
            pts = all_aug_points[idx]
            n_det = len(pts)
            viewer.add_points(
                pts,
                size=Points_size,
                name=f"{n_det} points ({aug_name}) {name}",
            )

        if Save_result:
            subfolder = create_unique_subfolder(
                str(Save_folder), str(Experiment_name)
            )
            result_table = {
                "Frame": list(range(n_frames)),
                "Count": avg_counts_per_frame,
            }
            df = pd.DataFrame(result_table)
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
TTA prediction on stack
Image name: {name}
Detection model: {Select_model}
Augmentations used: {' + '.join(best_augs)}
Per‑augmentation thresholds: {aug_thresholds}
Averaged detection count: {avg_counts_per_frame}
SAHI parameters used: size={Sahi_size}, overlap={Sahi_overlap}, postprocess={Postprocess}, match_metric={Match_metric}, iou_thr={Intersection_threshold}
"""
            metadata_path = os.path.join(subfolder, f"{name}_TTA_metadata.txt")
            with open(metadata_path, "w", encoding="utf-8") as f:
                f.write(metadata)
            show_info(f"TTA results saved in {subfolder}")

        viewer.window._status_bar._toggle_activity_dock(False)
        return

    if use_dynamic:
        with open(Mode_file, "rb") as f:
            model_data = pickle.load(f)
        regressor = model_data["regressor"]
        scaler = model_data["scaler"]
        feature_names = model_data["feature_names"]
        minimal_threshold = model_data.get("minimal_threshold", 0.01)

        viewer.window._status_bar._toggle_activity_dock(True)

        initialization_pbar = progress(total=0, desc="Initializing model")
        model, _ = initialize_model(
            str(Select_model), minimal_threshold, cuda_available
        )
        initialization_pbar.close()

        all_points = []
        result_table = {"Frame": [], "Count": []}
        total_filtered = 0
        win_size = Sahi_size // 2
        stride = win_size // 2

        with progress(total=len(pic), desc="Processing frames") as pbar_frames:
            for i in range(len(pic)):
                frame = pic[i]
                if hasattr(frame, "compute"):
                    frame = frame.compute()
                if frame.dtype == np.uint16:
                    frame = cv2.convertScaleAbs(frame, alpha=255 / 65535)
                    frame = frame.astype(np.uint8)
                if is_gray:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                pbar_slices = None
                post_pbar_dyn = None

                def progress_callback(current, total, frame_idx=i):
                    nonlocal pbar_slices, post_pbar_dyn
                    if pbar_slices is None:
                        pbar_slices = progress(
                            total=total,
                            desc=f"Sliced prediction frame {frame_idx+1}",
                        )
                    pbar_slices.update(1)
                    if current == total:
                        pbar_slices.close()
                        post_pbar_dyn = progress(
                            total=0,
                            desc=f"Running postprocessing frame {frame_idx+1}",
                        )

                result = get_sliced_prediction(
                    frame,
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
                if post_pbar_dyn is not None:
                    post_pbar_dyn.close()

                detections = result.object_prediction_list
                if not detections:
                    result_table["Frame"].append(i)
                    result_table["Count"].append(0)
                    pbar_frames.update(1)
                    continue

                dynamic_threshold_pbar = progress(
                    total=0, desc="Applying dynamic threshold"
                )

                det_list = []
                for det in detections:
                    x1, x2, y1, y2 = (
                        int(det.bbox.minx),
                        int(det.bbox.maxx),
                        int(det.bbox.miny),
                        int(det.bbox.maxy),
                    )
                    conf = det.score.value
                    det_list.append((x1, x2, y1, y2, conf))

                # Convert frame to grayscale for threshold map
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                else:
                    gray = frame

                # Build threshold map for this frame
                window_centers = build_threshold_map(
                    gray,
                    det_list,
                    regressor,
                    scaler,
                    feature_names,
                    win_size,
                    stride,
                )

                sigma = win_size / 4.0
                filtered_points = []
                for x1, x2, y1, y2, conf in det_list:
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
                            weight = np.exp(-dist2 / (2 * sigma**2))
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
                        filtered_points.append(
                            [i, int((y1 + y2) / 2), int((x1 + x2) / 2)]
                        )
                dynamic_threshold_pbar.close()

                all_points.extend(filtered_points)
                result_table["Frame"].append(i)
                result_table["Count"].append(len(filtered_points))
                total_filtered += len(filtered_points)
                pbar_frames.update(1)

        if all_points:
            viewer.add_points(
                np.array(all_points),
                size=Points_size,
                name=f"{total_filtered} points (dynamic) {name}",
            )
        else:
            viewer.add_points(
                np.empty((0, 3)),
                size=Points_size,
                name=f"0 points (dynamic) {name}",
            )

        if Save_result:
            subfolder = create_unique_subfolder(
                str(Save_folder), str(Experiment_name)
            )
            dynamic_folder = os.path.join(
                subfolder, "Dynamic_Threshold_Results"
            )
            os.makedirs(dynamic_folder, exist_ok=True)

            df = pd.DataFrame.from_dict(result_table)
            if Save_format in ("CSV", "Both"):
                df.to_csv(
                    os.path.join(dynamic_folder, f"{name}_dynamic_counts.csv"),
                    index=False,
                )
            if Save_format in ("XLSX", "Both"):
                df.to_excel(
                    os.path.join(
                        dynamic_folder, f"{name}_dynamic_counts.xlsx"
                    ),
                    index=False,
                )

            current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
            metadata = f"""Experiment time: {current_date}
Dynamic threshold prediction on 1-stack
Image name: {name}
Detection model: {Select_model}
Dynamic threshold .pkl file: {Mode_file.name}
Number of detections before filtering: {len(det_list)}
Number of detections after filtering: {total_filtered}
SAHI parameters used: size={Sahi_size}, overlap={Sahi_overlap}, postprocess={Postprocess}, match_metric={Match_metric}, iou_thr={Intersection_threshold}
"""
            metadata_path = os.path.join(
                dynamic_folder, f"{name}_dynamic_metadata.txt"
            )
            with open(metadata_path, "w", encoding="utf-8") as f:
                f.write(metadata)
            show_info(f"Dynamic threshold results saved in {dynamic_folder}")

        viewer.window._status_bar._toggle_activity_dock(False)
        return

    # Regular detection
    initialization_pbar = progress(total=0, desc="Initializing model")
    detection_model, model_type = initialize_model(
        rf"{Select_model}", Confidence_threshold, cuda_available
    )
    initialization_pbar.close()

    all_points = []
    result_table = {"Frame": [], "Count": []}

    with progress(total=len(pic), desc="Processing frames") as pbar_frames:
        for i in range(len(pic)):
            frame = pic[i]
            if hasattr(frame, "compute"):
                frame = frame.compute()
            if frame.dtype == np.uint16:
                frame = cv2.convertScaleAbs(frame, alpha=255 / 65535)
                frame = frame.astype(np.uint8)
            if is_gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            pbar_slices = None

            def slice_callback(current, total, frame_idx=i):
                nonlocal pbar_slices
                if pbar_slices is None:
                    pbar_slices = progress(
                        total=total,
                        desc=f"Sliced prediction frame {frame_idx+1}",
                    )
                pbar_slices.update(1)
                if current == total:
                    pbar_slices.close()

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
                progress_callback=slice_callback,
            )
            result = result.to_coco_predictions()

            frame_count = 0
            for instance in result:
                bbox = instance["bbox"]
                center_x = int(bbox[0] + bbox[2] // 2)
                center_y = int(bbox[1] + bbox[3] // 2)
                all_points.append([i, center_y, center_x])
                frame_count += 1
            result_table["Frame"].append(i)
            result_table["Count"].append(frame_count)
            pbar_frames.update(1)

    if all_points:
        viewer.add_points(
            np.array(all_points),
            size=Points_size,
            name=f"{len(all_points)} points {name}",
        )
    else:
        viewer.add_points(
            np.empty((0, 3)),
            size=Points_size,
            name=f"0 points {name}",
        )

    viewer.window._status_bar._toggle_activity_dock(False)
    print("Prediction is complete!")

    if Save_result:
        print("Saving results...")
        subfolder = create_unique_subfolder(
            str(Save_folder), str(Experiment_name)
        )
        df = pd.DataFrame.from_dict(result_table)
        if Save_format in ("CSV", "Both"):
            df.to_csv(
                os.path.join(subfolder, f"{name} count results.csv"),
                index=False,
            )
            print(".csv file created successfully")
        if Save_format in ("XLSX", "Both"):
            df.to_excel(
                os.path.join(subfolder, f"{name} count results.xlsx"),
                index=False,
            )
            print(".xlsx file created successfully")

        print("Creating metadata file...")
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metadata = f"""Experiment time: {current_date}
Prediction on 1-stack
Image name: {name}
Detection model: {Select_model}
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
