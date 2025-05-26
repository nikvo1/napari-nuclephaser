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
from napari.layers import Image, Points
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
    Postprocess={
        "choices": ["GREEDYNMM", "NMS", "NMM"],
        "tooltip": "An algorithm to process overlapping detections. See obss/sahi library docs for more details.",
    },
    Match_metric={
        "choices": ["IOS", "IOU"],
        "tooltip": "A metric to determine when two detections are two different detections overlapping or is it a one detection. Sett obss/sahi library docs for more details",
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
def calibrate_with_points(
    Select_Phase_image: Image,
    Select_Points_layer: Points,
    viewer: napari.Viewer,
    Phase_model=first_model,
    Division_size=640,
    Calibration_proportion=0.1,
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
    """Takes a single-frame phase-contrast image (or other microscopy methods), corresponding Napari.Points layer with labeled nuclei,
    path to YOLO model to calibrate (Phase_model), SAHI options
    -> splits images into small chunks with Division_size size, then splits result stack into calibration and test subsets with given proportion (Calibration_proportion),
    then finds a Confidence threshold for Phase model that returns closest number of detected objects to number of points on according Points layer. Finds a best threshold for each image in calibration subset and averages them.
    Then initializes Phase model with calibrated confidence threshold and runs it on test subset against Points layer to evaluate accuracy.
    -> returns best threshold for Phase model, saves error scatterplot.png and metadata.txt files at given folder and subfolder name; creates new subfolder if given already exists
    Can be used to calibrate model on images labeled by human or on images with machine predictions checked and corrected by human
    """
    phase_pic = Select_Phase_image.data
    if len(phase_pic.shape) > 3 or (
        len(phase_pic.shape) == 3 and phase_pic.shape[-1] not in (1, 3, 4)
    ):  # Check whether image is single-frame, otherwise return error and stop the function
        show_error(
            "Phase image is not a single frame! Can't calibrate on a stack of images"
        )
        print(
            "Phase image is not a single frame! Can't calibrate on a stack of images"
        )
        return None
    if len(phase_pic.shape) == 2:
        phase_pic = cv2.cvtColor(phase_pic, cv2.COLOR_GRAY2RGB)
    if phase_pic.dtype == np.uint16:
        phase_pic = cv2.convertScaleAbs(phase_pic, alpha=255 / 65535)
        phase_pic = phase_pic.astype(np.uint8)

    points = Select_Points_layer.data

    print(len(points))
    if len(points) == 0:
        show_error("Points layer is empty! Can't proceed further")
        print("Points layer is empty! Can't proceed further")
        return None

    def split_image_and_points(image, points, window_size):
        height, width, _ = image.shape
        num_tiles_height = height // window_size
        num_tiles_width = width // window_size
        cropped_images = []
        points_per_tile = []

        for i in range(num_tiles_height):
            for j in range(num_tiles_width):
                # Calculate the current tile's boundaries
                left = j * window_size
                upper = i * window_size
                right = left + window_size
                lower = upper + window_size

                # Crop the image
                cropped = image[upper:lower, left:right, :]
                cropped_images.append(cropped)

                # Determine which points fall into this tile
                tile_points = []
                for point in points:
                    y, x = point[0], point[1]
                    if left <= x < right and upper <= y < lower:
                        # Adjust coordinates relative to the tile
                        adjusted_x = x - left
                        adjusted_y = y - upper
                        tile_points.append([adjusted_x, adjusted_y])
                points_per_tile.append(len(tile_points))

        return np.array(cropped_images), np.array(points_per_tile)

    stack, points = split_image_and_points(phase_pic, points, Division_size)

    n = int(len(stack) * Calibration_proportion)

    # Randomly select the indices for the first part
    np.random.seed(Random_seed)
    indices = np.random.choice(len(stack), n, replace=False)  # ADD random seed

    # Split the array into two parts
    calibration_part = stack[indices]
    calibration_points = points[indices]
    test_part = np.delete(stack, indices, axis=0)
    test_points = np.delete(points, indices, axis=0)
    print(
        f"Images initialized successfully! With {Division_size} window size image is split into {len(stack)} small ones"
    )

    print("Initializing model...")
    phase_model, model_type = initialize_model(
        rf"{Phase_model}", 0.01, cuda_available
    )
    print(
        f"Model is initialized! Model type is {model_type}. Running on {cuda_available}"
    )

    print(f"Running calibration on {len(calibration_part)} images...")
    thresholds = []
    viewer.window._status_bar._toggle_activity_dock(True)
    for i in progress(
        range(len(calibration_part)), desc="Running calibration"
    ):

        image = calibration_part[i]
        points = calibration_points[i]

        ground_truth = points

        phase_result = get_sliced_prediction(
            image,
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

        for box in phase_result.object_prediction_list:
            detection_confidences.append(box.score.value)

        best_threshold = 0
        best_difference = 100000

        if len(detection_confidences) == 0:
            continue

        for i in np.arange(0.01, 1, 0.01):
            phase_count = sum(x > i for x in detection_confidences)
            difference = abs(phase_count - ground_truth)
            if difference < best_difference:
                best_difference = difference
                best_threshold = round(i, 2)
        thresholds.append(best_threshold)

    if len(thresholds) == 0:
        print("Couldn't calibrate! Model didn't detect any objects")
        show_error("Couldn't calibrate! Model didn't detect any objects")
        return None

    best_threshold = np.array(thresholds).mean()
    print(
        f"Calibration is complete! Best threshold for {Phase_model} is {best_threshold:.3f}"
    )

    if len(test_part) == 0:
        print("There are no images in the test part! Skipping tests...")
        print(f"Best threshold for {Phase_model} is {best_threshold:.3f}")
        return None

    test_results = {"Predicted_count": [], "Ground_truth_count": []}

    print("Running test. Initializing calibrated model for testing...")
    calibrated_model, model_type = initialize_model(
        rf"{Phase_model}", best_threshold, cuda_available
    )
    print(
        f"Model is initialized! Model type is {model_type}. Running on {cuda_available}"
    )

    print(f"Running test on {len(test_part)} images...")
    for i in progress(range(len(test_part)), desc="Running test"):

        image = test_part[i]
        points = test_points[i]

        ground_truth = points
        test_results["Ground_truth_count"].append(ground_truth)

        phase_result = get_sliced_prediction(
            image,
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
    viewer.window._status_bar._toggle_activity_dock(False)
    test_ds = pd.DataFrame.from_dict(test_results)

    test_ds["Error"] = (
        test_ds["Ground_truth_count"] - test_ds["Predicted_count"]
    ) / test_ds["Ground_truth_count"]
    MAPE = test_ds["Error"].abs().mean() * 100
    print(f"Test is complete! MAPE for this model is {MAPE:.2f}%")

    def generate_plot_name(filename):
        # Split the filename into name and extension
        name, extension = os.path.splitext(filename)

        # Initialize a counter for the filename
        counter = 1

        # Check if the file exists and modify the filename if necessary
        while os.path.exists(filename):
            filename = f"{name}{counter}{extension}"
            counter += 1

        return filename

    print("Drawing error plot...")
    sns.set(rc={"figure.dpi": 150, "savefig.dpi": 150})
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=test_ds, x="Ground_truth_count", y="Predicted_count"
    ).set_title(
        f"{Phase_model} \n {best_threshold:.3f} threshold on {Select_Phase_image}. MAPE is {MAPE:.2f}%"
    )
    sns.lineplot(
        np.arange(0, test_ds["Ground_truth_count"].max(), 1), color="r"
    )
    subfolder = create_unique_subfolder(str(Save_folder), str(Experiment_name))
    file_name = os.path.join(subfolder, "Calibration error plot.png")
    fig.savefig(file_name)
    plt.close(fig)
    print(f"Error plot is saved at {subfolder}")

    print("Saving points...")
    Select_Points_layer.save(os.path.join(subfolder, "reference points.csv"))
    print("Points used for calibration are saved!")

    print("Creating metadata file...")
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    metadata = f"""Experiment time: {current_date}
    Calibration method: from points
    Phase image: {Select_Phase_image}, {phase_pic.shape} pixels
    Phase model: {Phase_model}, {model_type}
    Division size: {Division_size}, resulting in {len(stack)} small images
    Calibration proportion: {Calibration_proportion}, resulting in {len(calibration_part)} images for calibration and {len(test_part)} for testing.
    Random seed: {Random_seed}. Use this for exact reproduction of data
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
