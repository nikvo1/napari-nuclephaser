import os
import pathlib
import time
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

from napari_nuclephaser.utils import create_unique_subfolder, initialize_model

# cuda device check
cuda_available = "cuda:0" if cuda.is_available() else "cpu"

# find default models folder
models_folder = pathlib.Path(pathlib.Path(__file__).parent / "models")
first_model = next((x for x in models_folder.iterdir() if x.is_file()), None)


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
    Confidence_threshold: float = 0.5,
    Save_result=True,
    Save_folder=pathlib.Path(),
    Experiment_name="Experiment",
    ADVANCED_SETTINGS="",
    Postprocess="GREEDYNMM",
    Match_metric="IOS",
    Sahi_size=640,
    Sahi_overlap: float = 0.2,
    Intersection_threshold=0.3,
    Points_size=30,
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
