import logging
import time
from collections.abc import Callable

import numpy as np
import tqdm
from PIL import Image
from sahi.models.base import DetectionModel
from sahi.predict import (
    LOW_MODEL_CONFIDENCE,
    POSTPROCESS_NAME_TO_CLASS,
    filter_predictions,
    get_prediction,
    slice_image,
)
from sahi.prediction import PredictionResult

try:
    from sahi.utils.logger import logger
except ImportError:
    logger = logging.getLogger(__name__)


def custom_get_sliced_prediction(
    image: str | np.ndarray | Image.Image,
    detection_model: DetectionModel | None = None,
    slice_height: int | None = None,
    slice_width: int | None = None,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    perform_standard_pred: bool = True,
    postprocess_type: str = "GREEDYNMM",
    postprocess_match_metric: str = "IOS",
    postprocess_match_threshold: float = 0.5,
    postprocess_class_agnostic: bool = False,
    verbose: int = 1,
    merge_buffer_length: int | None = None,
    auto_slice_resolution: bool = True,
    slice_export_prefix: str | None = None,
    slice_dir: str | None = None,
    exclude_classes_by_name: list[str] | None = None,
    exclude_classes_by_id: list[int] | None = None,
    progress_bar: bool = False,
    progress_callback: Callable | None = None,
    batch_size: int = 1,
    force_postprocess_type: bool = False,
    confidence_threshold: float | None = None,
    # ----- NEW PARAMETERS -----
    filter_border_touching_detections: bool = True,
    border_touching_width: int = 5,
) -> PredictionResult:
    """Standard get_sliced_prediction function with edge cases filtering"""
    print("I'm using custom function!")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if detection_model is None:
        raise ValueError(
            "detection_model must be provided for sliced prediction"
        )

    original_confidence_threshold = detection_model.confidence_threshold
    if confidence_threshold is not None:
        detection_model.confidence_threshold = confidence_threshold

    try:
        durations_in_seconds = {}

        # create slices from full image
        time_start = time.perf_counter()
        slice_image_result = slice_image(
            image=image,
            output_file_name=slice_export_prefix,
            output_dir=slice_dir,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            auto_slice_resolution=auto_slice_resolution,
        )

        from sahi.models.ultralytics import UltralyticsDetectionModel

        num_slices = len(slice_image_result)
        durations_in_seconds["slice"] = time.perf_counter() - time_start

        if (
            not force_postprocess_type
            and detection_model.confidence_threshold < LOW_MODEL_CONFIDENCE
            and postprocess_type != "NMS"
        ):
            logger.warning(
                "Switching postprocess type/metric to NMS/IOU since model confidence "
                "threshold is low (%s).",
                detection_model.confidence_threshold,
            )
            postprocess_type = "NMS"
            postprocess_match_metric = "IOU"

        if (
            isinstance(detection_model, UltralyticsDetectionModel)
            and detection_model.is_obb
        ):
            postprocess_type = "NMS"

        if postprocess_type not in POSTPROCESS_NAME_TO_CLASS:
            raise ValueError(
                f"postprocess_type should be one of {list(POSTPROCESS_NAME_TO_CLASS.keys())} "
                f"but given as {postprocess_type}"
            )
        postprocess_constructor = POSTPROCESS_NAME_TO_CLASS[postprocess_type]
        postprocess = postprocess_constructor(
            match_threshold=postprocess_match_threshold,
            match_metric=postprocess_match_metric,
            class_agnostic=postprocess_class_agnostic,
        )

        postprocess_time = 0.0
        time_start = time.perf_counter()
        num_batches = (num_slices + batch_size - 1) // batch_size
        if verbose == 1 or verbose == 2:
            tqdm.tqdm.write(f"Performing prediction on {num_slices} slices.")

        if progress_bar:
            slice_iterator = tqdm.tqdm(
                range(num_batches), desc="Processing slices", total=num_batches
            )
        else:
            slice_iterator = range(num_batches)

        full_shape: list[int | float] = [
            slice_image_result.original_image_height,
            slice_image_result.original_image_width,
        ]
        object_prediction_list = []
        slices_processed = 0
        for batch_ind in slice_iterator:
            batch_start = batch_ind * batch_size
            batch_end = min(batch_start + batch_size, num_slices)
            batch_images = [
                slice_image_result.images[i]
                for i in range(batch_start, batch_end)
            ]
            batch_shifts: list[list[int | float]] = [
                list(slice_image_result.starting_pixels[i])
                for i in range(batch_start, batch_end)
            ]
            current_batch_size = len(batch_images)

            detection_model.perform_batch_inference(
                [np.ascontiguousarray(img) for img in batch_images]
            )
            detection_model.convert_original_predictions(
                shift_amount=batch_shifts,
                full_shape=[full_shape] * current_batch_size,
            )

            # MODIFIED LOOP: apply border filtering
            for idx, image_preds in enumerate(
                detection_model.object_prediction_list_per_image
            ):
                # Get the actual slice index and its shift
                slice_idx = batch_start + idx
                shift_x, shift_y = batch_shifts[idx]

                # Retrieve the slice image to get its actual dimensions
                slice_img = slice_image_result.images[slice_idx]
                # Assume numpy array; get height and width
                actual_slice_h = slice_img.shape[0]
                actual_slice_w = slice_img.shape[1]

                # Filter by excluded classes first
                filtered_preds = filter_predictions(
                    image_preds, exclude_classes_by_name, exclude_classes_by_id
                )

                for object_prediction in filtered_preds:
                    if object_prediction is None:
                        continue

                    # Edge‑touching filter (applied only when enabled)
                    if filter_border_touching_detections:
                        # Bounding box in slice coordinates
                        x_min = object_prediction.bbox.minx
                        y_min = object_prediction.bbox.miny
                        x_max = object_prediction.bbox.maxx
                        y_max = object_prediction.bbox.maxy

                        # Does it touch the slice border?
                        touches_slice_border = (
                            x_min <= border_touching_width
                            or y_min <= border_touching_width
                            or (actual_slice_w - x_max)
                            <= border_touching_width
                            or (actual_slice_h - y_max)
                            <= border_touching_width
                        )

                        if touches_slice_border:
                            # Compute global coordinates (before shifting)
                            full_x_min = x_min + shift_x
                            full_y_min = y_min + shift_y
                            full_x_max = x_max + shift_x
                            full_y_max = y_max + shift_y

                            # Does it touch the border of the whole large image?
                            touches_full_border = (
                                full_x_min <= border_touching_width
                                or full_y_min <= border_touching_width
                                or (
                                    slice_image_result.original_image_width
                                    - full_x_max
                                )
                                <= border_touching_width
                                or (
                                    slice_image_result.original_image_height
                                    - full_y_max
                                )
                                <= border_touching_width
                            )

                            # If it touches a slice border but NOT the full border, discard it
                            if not touches_full_border:
                                continue

                    # Keep this detection (shift to global coordinates)
                    object_prediction_list.append(
                        object_prediction.get_shifted_object_prediction()
                    )

            slices_processed += current_batch_size

            if (
                merge_buffer_length is not None
                and len(object_prediction_list) > merge_buffer_length
            ):
                postprocess_time_start = time.time()
                object_prediction_list = postprocess(object_prediction_list)
                postprocess_time += time.time() - postprocess_time_start

            if progress_callback is not None:
                progress_callback(slices_processed, num_slices)

        if num_slices > 1 and perform_standard_pred:
            prediction_result = get_prediction(
                image=image,
                detection_model=detection_model,
                shift_amount=[0, 0],
                full_shape=[
                    slice_image_result.original_image_height,
                    slice_image_result.original_image_width,
                ],
                postprocess=None,
                exclude_classes_by_name=exclude_classes_by_name,
                exclude_classes_by_id=exclude_classes_by_id,
            )
            object_prediction_list.extend(
                prediction_result.object_prediction_list
            )

        if len(object_prediction_list) > 1:
            postprocess_time_start = time.time()
            object_prediction_list = postprocess(object_prediction_list)
            postprocess_time += time.time() - postprocess_time_start

        time_end = time.perf_counter() - time_start
        durations_in_seconds["prediction"] = time_end - postprocess_time
        durations_in_seconds["postprocess"] = postprocess_time

        if verbose == 2:
            print(
                "Slicing performed in",
                durations_in_seconds["slice"],
                "seconds.",
            )
            print(
                "Prediction performed in",
                durations_in_seconds["prediction"],
                "seconds.",
            )
            print(
                "Postprocessing performed in",
                durations_in_seconds["postprocess"],
                "seconds.",
            )
    finally:
        detection_model.confidence_threshold = original_confidence_threshold

    return PredictionResult(
        image=image,
        object_prediction_list=object_prediction_list,
        durations_in_seconds=durations_in_seconds,
    )
