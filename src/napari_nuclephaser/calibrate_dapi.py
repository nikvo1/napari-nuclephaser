import os
import pathlib
from datetime import datetime

import cv2
import matplotlib
import napari
import numpy as np
import pandas as pd
import seaborn as sns
from magicgui import magic_factory
from matplotlib import pyplot as plt
from napari.layers import Image
from napari.utils import progress
from napari.utils.notifications import show_error
from sahi.predict import get_sliced_prediction
from torch import cuda

from napari_nuclephaser.utils import create_unique_subfolder, initialize_model

matplotlib.use("Agg")
# cuda device check
cuda_available = "cuda:0" if cuda.is_available() else "cpu"

# find default models folder
models_folder = pathlib.Path(pathlib.Path(__file__).parent / "models")
first_model = next((x for x in models_folder.iterdir() if x.is_file()), None)
model_type_list = ("yolov5", "ultralytics", "yolov8", "yolov11", "yolo11")


@magic_factory(
    Division_size={
        "max": 100000,
        "tooltip": "A small image size in pixel, the whole image will be divided into small ones with this size",
    },
    Calibration_proportion={
        "tooltip": "Determines which part of the result stack of small images will be used for calibration. The rest will be used for test"
    },
    Random_seed={
        "tooltip": "Number used for random number generator, use the same random seeds for exact reproduction of results."
    },
    Postprocess={
        "choices": ["GREEDYNMM", "NMS", "NMM"],
        "tooltip": "An algorithm to process overlapping detections. See obss/sahi library docs for more details.",
    },
    Match_metric={
        "choices": ["IOS", "IOU"],
        "tooltip": "A metric to determine when two detections are two different detections overlapping or is it a one detection. Sett obss/sahi library docs for more details",
    },
    DAPI_confidence_threshold={
        "tooltip": "Parameter that determines how many detections will DAPI model return. Use calibration widgets to determine optimal threshold for your use case."
    },
    Save_folder={"mode": "d"},
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
    Experiment_name={
        "tooltip": "Name of the subfolder that will be created for the results"
    },
    call_button="Calibrate",
    auto_call=False,
    result_widget=True,
)
def calibrate_with_dapi_image(
    Select_Phase_image: Image,
    Select_DAPI_image: Image,
    viewer: napari.Viewer,
    Phase_model=first_model,
    DAPI_model=first_model,
    Division_size=640,
    Calibration_proportion=0.1,
    DAPI_confidence_threshold=0.5,
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
    """Takes a single-frame phase-contrast image (or other microscopy methods), corresponding fluorescent nuclei image (DAPI, for example),
    path to YOLO model to calibrate (Phase_model), path to YOLO model that detects nuclei on fluorescent images (DAPI_model), SAHI options
    -> splits images into small chunks with Division_size size, then splits result stack into calibration and test subsets with given proportion (Calibration_proportion),
    then finds a Confidence threshold for Phase model that returns closest number of detected objects to one given by DAPI model. Finds a best threshold for each image in calibration subset and averages them.
    Then initializes Phase model with calibrated confidence threshold and runs it on test subset against DAPI model to evaluate accuracy.
    -> returns best threshold for Phase model, saves error scatterplot.png and metadata.txt files at given folder and subfolder name; creates new subfolder if given already exists
    """
    phase_pic = Select_Phase_image.data
    if len(phase_pic.shape) > 3 or (
        len(phase_pic.shape) == 3 and phase_pic.shape[-1] not in (1, 3, 4)
    ):  # Check whether image is single-frame, otherwise return error and stop the function
        show_error(
            "Phase image is not a single frame! Can't calibrate on a stack of images"
        )
        return None
    if len(phase_pic.shape) == 3:
        phase_pic = cv2.cvtColor(phase_pic, cv2.COLOR_RGB2GRAY)
    if phase_pic.dtype == np.uint16:
        phase_pic = cv2.convertScaleAbs(phase_pic, alpha=255 / 65535)
        phase_pic = phase_pic.astype(np.uint8)
    image_shape = phase_pic.shape

    dapi_pic = Select_DAPI_image.data
    if len(dapi_pic.shape) > 3 or (
        len(dapi_pic.shape) == 3 and dapi_pic.shape[-1] not in (1, 3, 4)
    ):  # Check whether image is single-frame, otherwise return error and stop the function
        show_error(
            "DAPI image is not a single frame! Can't calibrate on a stack of images"
        )
        return None
    if len(dapi_pic.shape) == 3:
        dapi_pic = cv2.cvtColor(dapi_pic, cv2.COLOR_RGB2GRAY)
    if dapi_pic.dtype == np.uint16:
        dapi_pic = cv2.convertScaleAbs(dapi_pic, alpha=255 / 65535)
        dapi_pic = dapi_pic.astype(np.uint8)

    if phase_pic.shape != dapi_pic.shape:
        show_error(
            f"Phase and DAPI images have different dimensions! Phase image is {phase_pic.shape} and DAPI is {dapi_pic.shape}"
        )
        print(
            f"Phase and DAPI images have different dimensions! Phase image is {phase_pic.shape} and DAPI is {dapi_pic.shape}"
        )
        return None
    merged = np.zeros((2, image_shape[0], image_shape[1]), dtype=np.uint8)

    merged[0, :, :] = phase_pic
    merged[1, :, :] = dapi_pic

    def split_image(image, size):
        # Function that splits images into stack of small images with given size
        _, height, width = image.shape
        new_width = size
        new_height = size
        width_factor = width // new_width
        height_factor = height // new_height
        images = []

        for i in range(height_factor):
            for j in range(width_factor):
                left = j * new_width
                upper = i * new_height
                right = left + new_width
                lower = upper + new_height
                cropped_image = image[:, upper:lower, left:right]
                images.append(cropped_image)

        return np.array(images)

    stack = split_image(merged, Division_size)

    n = int(len(stack) * Calibration_proportion)

    # Randomly select the indices for the first part
    np.random.seed(Random_seed)
    indices = np.random.choice(len(stack), n, replace=False)

    # Split the array into two parts
    calibration_part = stack[indices]
    test_part = np.delete(stack, indices, axis=0)
    print(
        f"Images initialized successfully! With {Division_size} window size image is split into {len(stack)} small ones"
    )

    print("Initializing DAPI model...")
    dapi_model, dapi_model_type = initialize_model(
        rf"{DAPI_model}", DAPI_confidence_threshold, cuda_available
    )
    print(
        f"DAPI model is initialized! Model type is {dapi_model_type}. Running on {cuda_available}"
    )

    print("Initializing Phase model...")
    phase_model, phase_model_type = initialize_model(
        rf"{Phase_model}", 0.01, cuda_available
    )
    print(
        f"Model is initialized! Model type is {phase_model_type}. Running on {cuda_available}"
    )
    print("Phase model is initialized!")

    print(f"Running calibration on {len(calibration_part)} images...")
    thresholds = []
    viewer.window._status_bar._toggle_activity_dock(True)
    for i in progress(
        range(len(calibration_part)), desc="Running calibration"
    ):
        image = calibration_part[i]
        phase = cv2.cvtColor(image[0], cv2.COLOR_GRAY2RGB)
        dapi = cv2.cvtColor(image[1], cv2.COLOR_GRAY2RGB)

        dapi_result = get_sliced_prediction(
            dapi,
            dapi_model,
            slice_height=Sahi_size,
            slice_width=Sahi_size,
            overlap_height_ratio=Sahi_overlap,
            overlap_width_ratio=Sahi_overlap,
            postprocess_type=Postprocess,
            postprocess_match_metric=Match_metric,
            postprocess_match_threshold=Intersection_threshold,
            verbose=0,
        )

        dapi_count = int(len(dapi_result.object_prediction_list))

        phase_result = get_sliced_prediction(
            phase,
            phase_model,
            slice_height=Sahi_size,
            slice_width=Sahi_size,
            overlap_height_ratio=Sahi_overlap,
            overlap_width_ratio=Sahi_overlap,
            postprocess_type=Postprocess,
            postprocess_match_metric=Match_metric,
            postprocess_match_threshold=Intersection_threshold,
            verbose=0,
        )

        detection_confidences = []

        if len(phase_result.object_prediction_list) > 0:
            for box in phase_result.object_prediction_list:
                detection_confidences.append(box.score.value)

        best_threshold = 0
        best_difference = 100000

        if len(detection_confidences) == 0:
            continue

        for i in np.arange(0.01, 1, 0.01):
            phase_count = int(sum(1 for x in detection_confidences if x > i))
            difference = abs(phase_count - dapi_count)
            if difference < best_difference:
                best_difference = difference
                best_threshold = round(i, 2)
        thresholds.append(best_threshold)

    if len(thresholds) == 0:
        print("Couldn't calibrate! Model didn't detect any objects")
        show_error("Couldn't calibrate! Model didn't detect any objects")

    best_threshold = np.array(thresholds).mean()
    print(
        f"Calibration is complete! Best threshold for {Phase_model} is {best_threshold:.3f}"
    )

    if len(test_part) == 0:
        print("There are no images in the test part! Skipping tests...")
        print(f"Best threshold for {Phase_model} is {best_threshold:.3f}")
        return None

    print("Running test. Initializing calibrated model for testing...")
    test_results = {"Predicted_count": [], "DAPI_count": []}
    calibrated_model, calibrated_model_type = initialize_model(
        rf"{Phase_model}", best_threshold, cuda_available
    )
    print(
        f"Model is initialized! Model type is {calibrated_model_type}. Running on {cuda_available}"
    )
    print("Calibrated model is initialized!")

    print(f"Running test on {len(test_part)} images...")
    for i in progress(range(len(test_part)), desc="Running test"):
        image = test_part[i]
        phase = cv2.cvtColor(image[0], cv2.COLOR_GRAY2RGB)
        dapi = cv2.cvtColor(image[1], cv2.COLOR_GRAY2RGB)

        dapi_result = get_sliced_prediction(
            dapi,
            dapi_model,
            slice_height=Sahi_size,
            slice_width=Sahi_size,
            overlap_height_ratio=Sahi_overlap,
            overlap_width_ratio=Sahi_overlap,
            postprocess_type=Postprocess,
            postprocess_match_metric=Match_metric,
            postprocess_match_threshold=Intersection_threshold,
            verbose=0,
        )

        dapi_count = len(dapi_result.object_prediction_list)
        test_results["DAPI_count"].append(dapi_count)

        phase_result = get_sliced_prediction(
            phase,
            calibrated_model,
            slice_height=Sahi_size,
            slice_width=Sahi_size,
            overlap_height_ratio=Sahi_overlap,
            overlap_width_ratio=Sahi_overlap,
            postprocess_type=Postprocess,
            postprocess_match_metric=Match_metric,
            postprocess_match_threshold=Intersection_threshold,
            verbose=0,
        )

        phase_count = len(phase_result.object_prediction_list)
        test_results["Predicted_count"].append(phase_count)
    print("Test is complete!")
    viewer.window._status_bar._toggle_activity_dock(False)
    test_ds = pd.DataFrame.from_dict(test_results)

    test_ds["Error"] = (
        test_ds["DAPI_count"] - test_ds["Predicted_count"]
    ) / test_ds["DAPI_count"]
    MAPE = test_ds["Error"].abs().mean() * 100
    print(f"MAPE for this model is {MAPE:.2f}%")

    print("Drawing error plot...")
    sns.set(rc={"figure.dpi": 150, "savefig.dpi": 150})
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=test_ds, x="DAPI_count", y="Predicted_count"
    ).set_title(
        f"{Phase_model} \n {best_threshold:.3f} threshold on {Select_Phase_image}. MAPE is {MAPE:.2f}%"
    )
    sns.lineplot(np.arange(0, test_ds["DAPI_count"].max(), 1), color="r")
    subfolder = create_unique_subfolder(str(Save_folder), str(Experiment_name))
    file_name = os.path.join(subfolder, "Calibration error plot.png")
    fig.savefig(file_name)
    plt.close(fig)
    print(f"Error plot is saved at {subfolder}")

    print("Creating metadata file...")
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    metadata = f"""Experiment time: {current_date}
    Calibration method: from DAPI image
    Phase image: {Select_Phase_image}, {phase_pic.shape} pixels
    DAPI image: {Select_DAPI_image}, {dapi_pic.shape} pixels
    Phase model: {Phase_model}, {phase_model_type}
    DAPI model: {DAPI_model}, {dapi_model_type}
    Division size: {Division_size}, resulting in {len(stack)} small images
    Calibration proportion: {Calibration_proportion}, resulting in {len(calibration_part)} images for calibration and {len(test_part)} for testing.
    Random seed: {Random_seed}. Use this for exact reproduction of data
    DAPI confidence threshold: {DAPI_confidence_threshold}
    Postprocess algorithm: {Postprocess}
    Match metric: {Match_metric}
    Intersection threshold: {Intersection_threshold}
    SAHI size: {Sahi_size}
    SAHI overlap: {Sahi_overlap}
    Exact best threshold: {best_threshold}
    Exact result MAPE: {MAPE}%"""
    metadata_path = os.path.join(subfolder, "metadata.txt")
    with open(metadata_path, "w") as f:
        f.write(metadata)
    print("Metadata file is saved!")

    return f"Best threshold for {Phase_model} is {best_threshold:.3f} with MAPE {MAPE:.2f}%"
