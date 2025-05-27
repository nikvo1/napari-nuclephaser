import pathlib

import cv2
import napari
import numpy as np
from magicgui import magic_factory
from napari.layers import Image
from napari.utils.notifications import show_error, show_info
from sahi.predict import get_sliced_prediction
from torch import cuda

from napari_nuclephaser.utils import initialize_model

# cuda device check
cuda_available = "cuda:0" if cuda.is_available() else "cpu"

# find default models folder
models_folder = pathlib.Path(pathlib.Path(__file__).parent / "models")
first_model = next((x for x in models_folder.iterdir() if x.is_file()), None)
model_type_list = ("yolov5", "ultralytics", "yolov8", "yolov11", "yolo11")


@magic_factory(
    Postprocess={
        "choices": ["GREEDYNMM", "NMS", "NMM"],
        "tooltip": "An algorithm to process overlapping detections. See obss/sahi library docs for more details.",
    },
    Match_metric={
        "choices": ["IOS", "IOU"],
        "tooltip": "A metric to determine when two detections are two different detections overlapping or is it a one detection. Sett obss/sahi library docs for more details",
    },
    Calibration_number={
        "max": 10000000,
        "tooltip": "The ground truth number of objects on the image. Widget will return model confidence threshold that returns closest number of objects to this number",
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
    call_button="Calibrate",
    auto_call=False,
    result_widget=True,
)
def calibrate_with_known_number(
    Select_image: Image,
    viewer: napari.Viewer,
    Select_model=first_model,
    Calibration_number=100,
    ADVANCED_SETTINGS="",
    Postprocess="GREEDYNMM",
    Match_metric="IOS",
    Intersection_threshold=0.3,
    Sahi_size=640,
    Sahi_overlap: float = 0.2,
):
    """takes a single-frame image, a model_name.pt, sahi parameters and calibration number (number of objects on the image counted in advance)
    -> returns a confidence threshold for given model that returns closest number to the given calibration number)
    """
    #####pic = viewer.layers[0].data
    pic = Select_image.data
    if (
        len(pic.shape) == 2
    ):  # Check if image is single channel. YOLO models work only with RGB images.
        pic = cv2.cvtColor(pic, cv2.COLOR_GRAY2RGB)
    if len(pic.shape) > 3 or (
        len(pic.shape) == 3 and pic.shape[-1] not in (1, 3, 4)
    ):  # Check whether image is single-frame, otherwise return error and stop the function
        show_error(
            "Image is not a single frame! Can't calibrate on a stack of images"
        )
        return None
    if pic.dtype == np.uint16:
        pic = cv2.convertScaleAbs(pic, alpha=255 / 65535)
        pic = pic.astype(np.uint8)

    print("Initializing model...")
    detection_model, model_type = initialize_model(
        rf"{Select_model}",
        0.01,
        # Initialize model with lowest confidence threshold for calibration
        cuda_available,
    )
    print(
        f"Model is initialized! Model type is {model_type}. Running on {cuda_available}"
    )

    print("Running prediction for calibration...")
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
    )
    result = result.to_coco_predictions()
    print("Prediction is complete!")

    print("Calibrating...")
    scores = []
    for instance in result:
        score = instance["score"]
        scores.append(score)
    scores = np.array(scores)

    minimal_difference = np.inf
    best_threshold = 0
    # Loop for finding the best confidence threshold.
    for i in np.arange(0.01, 1, 0.01):
        number = np.count_nonzero(scores >= i)
        difference = abs(number - Calibration_number)
        if difference <= minimal_difference:
            minimal_difference = difference
            best_threshold = round(i, 2)
    show_info(
        f"Calibrated successfully! Best threshold for model {Select_model} is {best_threshold}"
    )
    return f"Best threshold for model {Select_model} is {best_threshold}"
