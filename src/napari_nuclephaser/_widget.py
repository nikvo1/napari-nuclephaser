import os
import pathlib
import time
from datetime import datetime

import cv2
import matplotlib
import napari.types
import numpy as np
import pandas as pd
import seaborn as sns
from magicgui import magic_factory
from matplotlib import pyplot as plt
from napari.layers import Image, Points
from napari.utils import progress
from napari.utils.notifications import show_error, show_info
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from torch import cuda

matplotlib.use("Agg")
# cuda device check
cuda_available = "cuda:0" if cuda.is_available() else "cpu"

# find default models folder
models_folder = pathlib.Path(pathlib.Path(__file__).parent / "models")
first_model = next((x for x in models_folder.iterdir() if x.is_file()), None)
model_type_list = ("yolov5", "ultralytics", "yolov8", "yolov11", "yolo11")


def initialize_model(model_path, confidence_threshold, device):
    """Takes a YOLO model path, confidence threshold and device and returns an initialized model
    without explicitly passing model type"""
    for model_type in model_type_list:
        try:
            detection_model = AutoDetectionModel.from_pretrained(
                model_type=model_type,
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                device=device,
            )
            return (
                detection_model,
                model_type,
            )  # Return the successfully initialized model
        except TypeError:
            continue  # Continue to the next model type

    # Raise an error if all attempts fail
    raise RuntimeError(
        "Failed to initialize model from the provided file path."
    )


def create_unique_subfolder(parent_folder, subfolder_name):
    """Takes a root folder path and subfolder name and returns a straight up subfolder path if
    one doesn't exist, and returns same path with subfolder1/2/3... otherwise
    """
    base_path = os.path.join(parent_folder, subfolder_name)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        return base_path

    counter = 1
    while True:
        new_name = f"{subfolder_name}{counter}"
        new_path = os.path.join(parent_folder, new_name)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            return new_path
        counter += 1


@magic_factory(
    Postprocess={
        "choices": ["GREEDYNMM", "NMS", "NMM"],
        "tooltip": "An algorithm to process overlapping detections. See obss/sahi library docs for more details.",
    },
    Match_metric={
        "choices": ["IOS", "IOU"],
        "tooltip": "A metric to determine when two detections are two different detections overlapping or is it a one detection. Sett obss/sahi library docs for more details",
    },
    Generate_points={
        "tooltip": "If chosen, Points layer will be created with point at the center of bounding box for each detection"
    },
    Generate_bbox={
        "tooltip": "If chosen, Shapes layer will be created with rectangle representing bounding box of each detection"
    },
    Show_confidence={
        "tooltip": "If chosen, each rectangle in Shapes layer will have confidence score of each detection printed above it"
    },
    Confidence_threshold={
        "tooltip": "Parameter that determines how many detections will model return. Use calibration widgets to determine optimal threshold for your use case."
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
    Points_size={
        "tooltip": "Points size in results Points layer. Can be changed later by pressing Ctrl+A and moving Size slider in the layer itself"
    },
    Bbox_thickness={
        "tooltip": "Thickness of the side of rectangles in Shapes layer if Generate bbox is chosen"
    },
    Score_text_size={
        "tooltip": "Font size of confidence score text if Show confidence parameter is chosen"
    },
    call_button="Predict",
    auto_call=False,
    result_widget=False,
)
def make_points(
    Select_image: Image,
    viewer: napari.Viewer,
    Select_model=first_model,
    Postprocess="GREEDYNMM",
    Match_metric="IOS",
    Generate_points=True,
    Generate_bbox=False,
    Show_confidence=False,
    Confidence_threshold: float = 0.5,
    Sahi_size=640,
    Sahi_overlap: float = 0.2,
    Intersection_threshold=0.3,
    Points_size=10,
    Bbox_thickness=5,
    Score_text_size=3,
) -> napari.types.LayerDataTuple:
    """Takes a single-frame image of any size, YOLO object detection model (v5, v8 or v11) and SAHI parameters ->
    returns a detection in formats of Points layer and/or Shapes layer with boxes ahd corresponding confidence scores
    """
    pic = Select_image.data
    if (
        len(pic.shape) == 2
    ):  # Check if image is single channel. YOLO models work only with RGB images.
        pic = cv2.cvtColor(pic, cv2.COLOR_GRAY2RGB)
    if len(pic.shape) > 3 or (
        len(pic.shape) == 3 and pic.shape[-1] not in (1, 3, 4)
    ):  # Check whether image is single-frame, otherwise return error and stop the function
        show_error(
            "Image is not a single frame! Choose different widget for processing stacks of images"
        )
        return None
    name = Select_image.name  # Fetch image name for further purposes
    if pic.dtype == np.uint16:
        pic = cv2.convertScaleAbs(pic, alpha=255 / 65535)
        pic = pic.astype(np.uint8)

    print("Initializing model...")
    detection_model, model_type = initialize_model(
        rf"{Select_model}", Confidence_threshold, cuda_available
    )
    print(
        f"Model is initialized! Model type is {model_type}. Running on {cuda_available}"
    )

    print("Performing sliced prediction...")
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
    )  # Standard SAHI sliced prediction code
    result = result.to_coco_predictions()
    print("Prediction is done!")

    def create_points(result):
        # Function for converting prediction results from COCO format into napari.layers.Points layer
        points = []
        for instance in result:
            bbox = instance["bbox"]
            Y, X = int(bbox[0] + (bbox[2] // 2)), int(bbox[1] + (bbox[3] // 2))
            points.append([X, Y])
        n_cells = len(points)
        points = np.array(points)

        viewer.add_points(
            points, size=Points_size, name=f"{n_cells} points {name}"
        )
        return points, n_cells

    def create_bbox(result):
        # Function for converting prediction results from COCO format into napari.layers.Shapes layer
        bboxes = []
        scores = []
        for instance in result:
            bbox = instance["bbox"]
            score = instance["score"]
            Y1, X1, Y2, X2 = (
                int(bbox[0]),
                int(bbox[1]),
                int(bbox[0] + (bbox[2])),
                int(bbox[1] + (bbox[3])),
            )
            bboxes.append(np.array([[X1, Y1], [X1, Y2], [X2, Y2], [X2, Y1]]))
            scores.append(score)
        n_cells = len(scores)
        # bboxes, scores = np.array(bboxes), np.array(scores)

        # create the properties dictionary
        properties = {"score": scores}

        # specify the display parameters for the text

        if Show_confidence:
            text_parameters = {
                "string": "{score:.2f}",
                "size": Score_text_size,
                "color": "red",
                "anchor": "upper_left",
                "translation": [-3, 0],
            }

            viewer.add_shapes(
                bboxes,
                face_color="transparent",
                edge_color="red",
                edge_width=Bbox_thickness,
                properties=properties,
                text=text_parameters,
                name=f"{n_cells} bounding boxes {name}",
            )
        else:
            viewer.add_shapes(
                bboxes,
                face_color="transparent",
                edge_color="red",
                edge_width=Bbox_thickness,
                properties=properties,
                name=f"{n_cells} bounding boxes {name}",
            )
        return bboxes, scores, n_cells

    if Generate_points:
        print("Generating points...")
        create_points(result)
        print("Points are generated!")
    if Generate_bbox:
        print("Generating boxes...")
        create_bbox(result)
        print("Boxes are generated!")
    if not Generate_points and not Generate_bbox:
        print("None of the options are chosen, generating points as a default")
        create_points(result)
        print("Points are generated!")
    return None


@magic_factory(
    Postprocess={
        "choices": ["GREEDYNMM", "NMS", "NMM"],
        "tooltip": "An algorithm to process overlapping detections. See obss/sahi library docs for more details.",
    },
    Match_metric={
        "choices": ["IOS", "IOU"],
        "tooltip": "A metric to determine when two detections are two different detections overlapping or is it a one detection. Sett obss/sahi library docs for more details",
    },
    Confidence_threshold={
        "tooltip": "Parameter that determines how many detections will model return. Use calibration widgets to determine optimal threshold for your use case."
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
    Points_size={
        "tooltip": "Points size in results Points layer. Can be changed later by pressing Ctrl+A and moving Size slider in the layer itself"
    },
    Save_result={
        "tooltip": "If chosen, a folder will be created with .csv or .xlsx file containing quantification of objects for each frame"
    },
    Experiment_name={
        "tooltip": "Name of the subfolder that will be created for the results"
    },
    Save_csv={
        "tooltip": "If chosen, .csv format file with counting results will be saved at given folder"
    },
    Save_xlsx={
        "tooltip": "If chosen, .xlsx format file with counting results will be saved at given folder"
    },
    call_button="Predict",
    Save_folder={"mode": "d"},
    auto_call=False,
    result_widget=False,
)
def predict_on_stack(
    Select_stack: Image,
    viewer: napari.Viewer,
    Select_model=first_model,
    Postprocess="GREEDYNMM",
    Match_metric="IOS",
    Confidence_threshold: float = 0.5,
    Sahi_size=640,
    Sahi_overlap: float = 0.2,
    Intersection_threshold=0.3,
    Points_size=30,
    Save_result=True,
    Save_folder=pathlib.Path(),
    Experiment_name="Experiment",
    Save_csv=False,
    Save_xlsx=True,
):
    """Takes a 1-dimensional stack of images (grayscale of RGB), YOLO object detection model (v5, v8 or v11) and SAHI parameters ->
    returns a detection in formats of one-dimensional stack of Points layers and saves count results in .csv/.xlsx format and metadata in .txt format
    in given folder with given subfolder name. Will create new subfolder if one with given name already exists
    """
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
    print("Images stack is initialized successfuly!")

    print("Initializing model...")
    detection_model, model_type = initialize_model(
        rf"{Select_model}", Confidence_threshold, cuda_available
    )
    print(
        f"Model is initialized! Model type is {model_type}. Running on {cuda_available}"
    )

    points = []
    result_table = {"Frame": [], "Count": []}

    print("Running predictions...")
    viewer.window._status_bar._toggle_activity_dock(True)
    for i in progress(range(len(pic)), desc="Running predictions"):
        if (
            i == 0
        ):  # Clock the starting time for the first frame to assess the whole stack processing time
            start_time = time.time()
        frame = pic[i]
        if (
            type(frame).__module__ == "dask.array.core"
            and type(frame).__name__ == "Array"
        ):
            frame = (
                frame.compute()
            )  # Code to translate image from dask array to numpy array
        if frame.dtype == np.uint16:
            frame = cv2.convertScaleAbs(frame, alpha=255 / 65535)
            frame = frame.astype(np.uint8)
        if is_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
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
        )
        result = result.to_coco_predictions()
        for instance in result:
            bbox = instance["bbox"]
            Y, X = int(bbox[0] + (bbox[2] // 2)), int(bbox[1] + (bbox[3] // 2))
            points.append([i, X, Y])
        result_table["Frame"].append(i)
        result_table["Count"].append(len(result))

        # Clock the end of the first frame processing and assess the whole stack processing time
        if i == 0:
            finish_time = time.time()
            frame_time = round(finish_time - start_time)
            print(f"First slice took {frame_time} seconds to process.")
            print(
                f"Processing whole stack will take approximately {frame_time * len(pic)} seconds"
            )
        print(f"Slice {i} is done!")
    viewer.add_points(points, size=Points_size, name=f"Points for {name}")
    viewer.window._status_bar._toggle_activity_dock(False)
    print("Prediction is complete!")

    if Save_result:
        print("Saving results...")
        subfolder = create_unique_subfolder(
            str(Save_folder), str(Experiment_name)
        )
        df = pd.DataFrame.from_dict(result_table)
        if Save_csv:
            df.to_csv(
                os.path.join(subfolder, f"{name} count results.csv"),
                index=False,
            )
            print(".csv file created successfuly")
        if Save_xlsx:
            df.to_excel(
                os.path.join(subfolder, f"{name} count results.xlsx"),
                index=False,
            )
            print(".xlsx file created successfuly")
        if not Save_csv and not Save_xlsx:
            df.to_csv(
                os.path.join(subfolder, f"{name} count results.csv"),
                index=False,
            )
            print(
                "None of the options are chosen, creating .csv file as a default"
            )

        print("Creating metadata file...")
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        metadata = f"""Experiment time: {current_date}
        Prediction on 1-stack
        Stack napari name: {name}
        Detection_model: {Select_model}
        Model type: {model_type}
        Confidence threshold: {Confidence_threshold}
        Postprocess algorithm: {Postprocess}
        Match metric: {Match_metric}
        Intersection threshold: {Intersection_threshold}
        SAHI size: {Sahi_size}
        SAHI overlap: {Sahi_overlap}"""
        metadata_path = os.path.join(subfolder, f"{name} count metadata.txt")

        with open(metadata_path, "w") as f:
            f.write(metadata)
        print("Metadata file is saved!")

    show_info("Made predictions for stack successfully!")


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
    Random_seed=42,
    DAPI_confidence_threshold=0.5,
    Postprocess="GREEDYNMM",
    Match_metric="IOS",
    Intersection_threshold=0.3,
    Sahi_size=640,
    Sahi_overlap: float = 0.2,
    Save_folder=pathlib.Path(),
    Experiment_name="Experiment",
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
    Select_points_layer: Points,
    viewer: napari.Viewer,
    Phase_model=first_model,
    Division_size=640,
    Calibration_proportion=0.1,
    Random_seed=42,
    Postprocess="GREEDYNMM",
    Match_metric="IOS",
    Intersection_threshold=0.3,
    Sahi_size=640,
    Sahi_overlap: float = 0.2,
    Save_folder=pathlib.Path(),
    Experiment_name="Experiment",
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

    points = Select_points_layer.data

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
    for i in progress(range(len(test_part)), desc="Running calibration"):

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
    Select_points_layer.save(os.path.join(subfolder, "reference points.csv"))
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


@magic_factory(
    auto_call=False,
    call_button="Convert",
    result_widget=False,
    Points_layer={"label": "Select points layer"},
    Reference_image={"label": "Select reference image"},
)
def convert_points_to_labels(
    Points_layer: Points,
    Reference_image: Image,
    viewer: napari.Viewer,
    Label_size=10,
):
    """Takes a Napari.Points layer (stacks are allowed) and reference image and returns points in Napari.Labels format.
    Main purpose is tracking with napari.btrack plugin (which accepts only Labels) or other plugins
    """
    points = Points_layer.data
    zeros = np.zeros(
        shape=(
            Reference_image.data.shape
            if len(Reference_image.data.shape) == 3
            else Reference_image.data.shape[:3]
        ),
        dtype=np.uint8,
    )
    for point in points:
        cv2.circle(
            zeros[int(point[0])],
            (int(point[2]), int(point[1])),
            Label_size,
            256,
            -1,
        )
    viewer.add_labels(
        zeros.astype(np.uint8),
        name=Points_layer.name + " labels",
        depiction="plane",
    )


@magic_factory(
    auto_call=False,
    call_button="Count Up",
    result_widget=True,
    Points_layer={"label": "Select points layer"},
)
def give_num_points(Points_layer: Points):
    # Count up the points from the label layer
    points = Points_layer
    return len(points.data)


#
# napari.Viewer()
# calculate_av_size()
viewer = napari.viewer.current_viewer()
# napari.run()
